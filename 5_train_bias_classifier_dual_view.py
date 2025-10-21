#!/usr/bin/env python3
''' Dual-View Hierarchical GNN for Bias Detection - Multi-GPU 10-Fold CV '''

import os
import sys
import argparse
import torch
import json
import numpy as np
import random
import time
import datetime
from pathlib import Path
import math

# --- Initial Device Setup (Minimal to avoid scheduler conflict) ---

from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer, LongformerModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

# --- 1. Command-line arguments for multi-GPU execution ---

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, required=True, help='Which fold to run (0-9)')
parser.add_argument('--gpu', type=int, required=True, choices=[0, 1, 2, 3], help='Which physical GPU index (0-3) to map this process to')
# CRITICAL FIX: Directing output to scratch path
parser.add_argument('--output_dir', type=str, default='/scratch/atharv.johar/Media-Bias-Analysis/results_dual_view', help='Output directory')
args = parser.parse_args()

# CRITICAL FIX: Isolate the process to the assigned GPU index
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"[Fold {args.fold}] Assigned to GPU {args.gpu}: {torch.cuda.get_device_name(0)}")
else:
    print(f"[Fold {args.fold}] Assigned to CPU (GPU {args.gpu} not available).")


# --- 2. Hyper-parameters ---

# VRAM FIX: Using 1024 tokens for stable VRAM use.
MAX_LEN = 2048
# HYPERPARAMETER TWEAK: Increased epochs to allow GNN/Classifier more time to learn.
num_epochs = 10 
batch_size = 1 
check_times = 4 * num_epochs

# HYPERPARAMETER TWEAK: Weighted loss for imbalanced classification (Non-Bias:Bias is roughly 4:1)
# Class 0 (Non-Bias) receives weight 1.0; Class 1 (Bias) receives weight 4.0
CLASS_WEIGHTS = torch.tensor([1.0, 4.0]).to(device) 

lambda_event = 1.0
no_decay = ['bias', 'LayerNorm.weight']
longformer_weight_decay = 1e-2
non_longformer_weight_decay = 1e-2
warmup_proportion = 0.0
longformer_lr = 1e-5
non_longformer_lr = 2e-5

# --- 3. Utility Functions for Dataset/Splits ---

def get_available_classified_files():
    """Get list of available classified files"""
    classified_path = "./BASIL_event_graph_classified"
    all_files = [f for f in os.listdir(classified_path) if f.endswith('_classified.json')]
    
    file_info = []
    for f in all_files:
        parts = f.replace('_event_graph_classified.json', '').split('_')
        if len(parts) == 3 and parts[0] == 'basil':
            try:
                triplet_num = int(parts[1])
                media = parts[2]
                file_info.append({
                    'file': f,
                    'triplet': triplet_num,
                    'media': media,
                    'path': f"{classified_path}/{f}"
                })
            except ValueError:
                continue
    
    return file_info

def create_cv_folds_triplet_aware(file_list, n_folds=10, seed=42):
    """
    Create folds ensuring triplets (same story, different media) stay together
    This prevents data leakage
    """
    random.seed(seed)
    
    # Group by triplet
    triplets = {}
    for info in file_list:
        t = info['triplet']
        if t not in triplets:
            triplets[t] = []
        triplets[t].append(info)
    
    # Shuffle triplet numbers
    triplet_nums = list(triplets.keys())
    random.shuffle(triplet_nums)
    
    # Distribute triplets across folds
    folds = [[] for _ in range(n_folds)]
    for i, triplet_num in enumerate(triplet_nums):
        fold_idx = i % n_folds
        folds[fold_idx].extend([info['path'] for info in triplets[triplet_num]])
    
    return folds

# --- 4. Custom Dataset (Adapted for Dual-View and Hierarchy) ---

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

class custom_dataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with open(file_path, "r", encoding='utf-8') as in_json:
            article_json = json.load(in_json)

        input_ids = []
        attention_mask = []
        label_sentence = []
        event_words = []
        
        # --- Paragraph Detection Heuristic (Simplified) ---
        current_paragraph_id = 0
        
        for sent_i in range(len(article_json['sentences'])):
            sent_data = article_json['sentences'][sent_i]
            
            # Heuristic for paragraph ID 
            if 'sentence_id' in sent_data and isinstance(sent_data['sentence_id'], int) and sent_data['sentence_id'] > 0 and sent_data['sentence_id'] % 5 == 1:
                current_paragraph_id += 1
            elif sent_i == 0:
                current_paragraph_id = 0
            
            if len(sent_data.get('sentence_text', '')) > 1:
                start = len(input_ids)
                input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['input_ids'])
                attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['attention_mask'])
                end = len(input_ids)
                
                # label_sentence: [start, end, index_of_sentence, label_bias, paragraph_id]
                if end < MAX_LEN:
                    label_sentence.append([
                        start, 
                        end, 
                        sent_i, 
                        sent_data.get('label_info_lex_bias', -1), 
                        current_paragraph_id 
                    ])
                
                # Token processing
                for token_i in range(len(sent_data['tokens'])):
                    token_data = sent_data['tokens'][token_i]
                    token_text = token_data['token_text']
                    start = len(input_ids)
                    word_encoding = tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)
                    
                    if len(input_ids) + len(word_encoding['input_ids']) >= MAX_LEN:
                         # Stop adding tokens if MAX_LEN is reached
                        break
                        
                    input_ids.extend(word_encoding['input_ids'])
                    attention_mask.extend(word_encoding['attention_mask'])
                    end = len(input_ids)
                    
                    if token_data.get('label_event', 0) == 1:
                        
                        fi_class = token_data.get('fi_classification', 'FACTUAL')
                        fi_label = 1 if fi_class == 'INTERPRETIVE' else 0 # Factual(0)/Interpretive(1)
                        
                        event_words.append([
                            start, end, sent_i,
                            token_data['index_of_token'],
                            token_data.get('prob_event', [0])[0], 
                            token_data.get('prob_event', [0])[1], 
                            token_data['label_event'], 
                            fi_label, 
                            current_paragraph_id 
                        ])

                # Check if we should stop processing the rest of the document
                if len(input_ids) >= MAX_LEN:
                    break

                # End of sentence tokens (only if there is still space)
                end_sent_tokens = tokenizer.encode_plus('</s>', add_special_tokens=False)
                if len(input_ids) + len(end_sent_tokens['input_ids']) < MAX_LEN:
                    input_ids.extend(end_sent_tokens['input_ids'])
                    attention_mask.extend(end_sent_tokens['attention_mask'])
                else:
                    break # Stop if we can't fit the sentence end marker

        
        # Padding/Truncation
        num_pad = MAX_LEN - len(input_ids)
        if num_pad > 0:
            input_ids.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['input_ids'])
            attention_mask.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['attention_mask'])

        input_ids = torch.tensor(input_ids[:MAX_LEN])
        attention_mask = torch.tensor(attention_mask[:MAX_LEN])
        label_sentence = torch.tensor(label_sentence)
        event_words = torch.tensor(event_words) if len(event_words) > 0 else torch.zeros((0, 9), dtype=torch.long)

        # Filter event_words that were truncated past MAX_LEN
        event_words = event_words[event_words[:, 1] < MAX_LEN]
        
        # Event pairs and relations (Mapped to rows in event_words, filtered later)
        event_pairs_raw = []
        label_coreference_raw = []
        label_temporal_raw = []
        label_causal_raw = []
        label_subevent_raw = []

        if 'relation_label' in article_json and len(article_json.get('relation_label', [])) > 0:
            # Create a dictionary to map old token index to new row index in event_words
            token_to_row = {int(event_words[i, 3]): i for i in range(event_words.size(0))}
            
            for event_pair_i in range(len(article_json['relation_label'])):
                event_1_index = article_json['relation_label'][event_pair_i]['event_1']['index_of_token']
                event_2_index = article_json['relation_label'][event_pair_i]['event_2']['index_of_token']
                
                # Only include pairs where both events survived truncation
                if event_1_index in token_to_row and event_2_index in token_to_row:
                    event_1_row = token_to_row[event_1_index]
                    event_2_row = token_to_row[event_2_index]
                    
                    event_pairs_raw.append([event_1_row, event_2_row])
                    
                    label_coreference_raw.append(article_json['relation_label'][event_pair_i]['prob_coreference'])
                    label_temporal_raw.append(article_json['relation_label'][event_pair_i]['prob_temporal'])
                    label_causal_raw.append(article_json['relation_label'][event_pair_i]['prob_causal'])
                    label_subevent_raw.append(article_json['relation_label'][event_pair_i]['prob_subevent'])


        if len(event_pairs_raw) == 0:
            event_pairs = torch.zeros((0, 2), dtype=torch.long)
            label_coreference = torch.zeros((0, 2), dtype=torch.float)
            label_temporal = torch.zeros((0, 4), dtype=torch.float)
            label_causal = torch.zeros((0, 3), dtype=torch.float)
            label_subevent = torch.zeros((0, 3), dtype=torch.float)
        else:
            event_pairs = torch.tensor(event_pairs_raw)
            label_coreference = torch.tensor(label_coreference_raw)
            label_temporal = torch.tensor(label_temporal_raw)
            label_causal = torch.tensor(label_causal_raw)
            label_subevent = torch.tensor(label_subevent_raw)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_sentence": label_sentence, # [start, end, sent_idx, bias_label, paragraph_id]
            "event_words": event_words, # [start, end, sent_idx, token_idx, prob_ne, prob_e, label_e, fi_label, paragraph_id]
            "event_pairs": event_pairs,
            "label_coreference": label_coreference,
            "label_temporal": label_temporal,
            "label_causal": label_causal,
            "label_subevent": label_subevent
        }

# --- 5. Custom Layers for Dual-View R-GAT (FINAL FIX) ---

class R_GAT_Layer(nn.Module):
    """Relation-Aware Graph Attention Network (Simplified to Coreference only for POC)"""
    def __init__(self, feature_dim, relation_types):
        super(R_GAT_Layer, self).__init__()
        self.feature_dim = feature_dim
        # CRITICAL: We only use the FIRST relation type ('coref') from the list for the forward pass logic.
        self.coref_relation_name = relation_types[0] # 'coref'
        
        self.W_Q = nn.Linear(feature_dim, feature_dim)
        self.W_K = nn.Linear(feature_dim, feature_dim)
        self.W_V = nn.Linear(feature_dim, feature_dim)

        # Initialize all relation heads just to load the model definition (only 'coref' used in forward)
        self.W_R = nn.ModuleDict({
            r: nn.Linear(feature_dim, feature_dim, bias=False) 
            for r in relation_types
        })
        self.a_R = nn.ModuleDict({
            r: nn.Linear(feature_dim * 2, 1) 
            for r in relation_types
        })
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, event_embeddings, event_pairs, coreference_labels):
        """
        Input: 
            event_embeddings: (N_events, feature_dim)
            event_pairs: (N_pairs, 2) - indices of event_embeddings (Local to the view)
            coreference_labels: (N_pairs, 1) - **Probability of Coreference (index 1 of original tensor)**
        """
        N = event_embeddings.size(0)
        N_pairs = event_pairs.size(0)
        if N == 0 or N_pairs == 0:
            return event_embeddings.clone()
        
        V = self.W_V(event_embeddings)
        H_initial = event_embeddings 
        
        # 1. Prepare Coreference-only Neighbor Mask
        # Filter pairs where coreference probability > 0.5 (hard decision)
        coref_mask = (coreference_labels[:, 0] > 0.5).nonzero(as_tuple=True)[0]
        
        if coref_mask.size(0) == 0:
            return event_embeddings.clone() 
        
        coref_pairs = event_pairs[coref_mask]
        
        new_embeddings = torch.zeros_like(event_embeddings).to(event_embeddings.device)
        
        # Use the specific coreference layer (defined at index 0)
        W_R_coref = self.W_R[self.coref_relation_name]
        a_R_coref = self.a_R[self.coref_relation_name]
        
        for i in range(N):
            # Find coreferent neighbors pointing TO event i
            source_indices = (coref_pairs[:, 1] == i).nonzero(as_tuple=True)[0]
            
            if len(source_indices) == 0:
                new_embeddings[i] = event_embeddings[i] 
                continue
                
            neighbor_indices = coref_pairs[source_indices, 0]
            
            # --- Attention Calculation for the 'coref' relation ---
            neighbor_embeddings = event_embeddings[neighbor_indices]
            
            # Relation-aware transformation (W_r * H_j)
            H_neighbor_r = W_R_coref(neighbor_embeddings)
            
            # Attention input (H_i || W_r * H_j)
            H_i_r = H_initial[i].repeat(H_neighbor_r.size(0), 1)
            attention_input = torch.cat([H_i_r, H_neighbor_r], dim=1)
            e_r = a_R_coref(attention_input) 
            
            # Apply Softmax to get attention weights (alpha_ij)
            attention_weights = F.softmax(self.leakyrelu(e_r), dim=0)
            
            # Aggregation (sum of alpha_ij * V_j)
            all_neighbor_V = V[neighbor_indices]
            aggregated_msg = torch.sum(attention_weights * all_neighbor_V, dim=0)
            
            # Update (with skip connection)
            new_embeddings[i] = aggregated_msg + event_embeddings[i]
                
        return new_embeddings


class Cross_View_Attention(nn.Module):
    """Level 1: Paragraph-Level, Cross-View Interaction"""
    def __init__(self, feature_dim):
        super(Cross_View_Attention, self).__init__()
        self.feature_dim = feature_dim
        
        # Cross-attention weights (W_Q, W_K, W_V)
        self.W_Q_F = nn.Linear(feature_dim, feature_dim) 
        self.W_K_I = nn.Linear(feature_dim, feature_dim) 
        self.W_V_I = nn.Linear(feature_dim, feature_dim) 

        self.W_Q_I = nn.Linear(feature_dim, feature_dim) 
        self.W_K_F = nn.Linear(feature_dim, feature_dim) 
        self.W_V_F = nn.Linear(feature_dim, feature_dim) 
        
        self.W_F_out = nn.Linear(feature_dim * 2, feature_dim)
        self.W_I_out = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, F_events, I_events):
        if F_events.size(0) == 0 and I_events.size(0) == 0:
             return F_events, I_events 
        if F_events.size(0) == 0:
             return F_events, I_events.clone() 
        if I_events.size(0) == 0:
             return F_events.clone(), I_events 
        
        # Factual <- Interpretive
        Q_F = self.W_Q_F(F_events)
        K_I = self.W_K_I(I_events)
        V_I = self.W_V_I(I_events)
        
        attention_weights_F_I = F.softmax(Q_F @ K_I.transpose(0, 1) / math.sqrt(self.feature_dim), dim=1)
        I_context_for_F = attention_weights_F_I @ V_I

        # Interpretive <- Factual
        Q_I = self.W_Q_I(I_events)
        K_F = self.W_K_F(F_events)
        V_F = self.W_V_F(F_events)
        
        attention_weights_I_F = F.softmax(Q_I @ K_F.transpose(0, 1) / math.sqrt(self.feature_dim), dim=1)
        F_context_for_I = attention_weights_I_F @ V_F
        
        # Final update 
        F_updated = F.relu(self.W_F_out(torch.cat([F_events, I_context_for_F], dim=1)))
        I_updated = F.relu(self.W_I_out(torch.cat([I_events, F_context_for_I], dim=1)))
        
        return F_updated, I_updated

class GAT_Layer(nn.Module):
    """Standard Graph Attention Network (Level 2: Document-Level, Paragraph Nodes)"""
    def __init__(self, feature_dim):
        super(GAT_Layer, self).__init__()
        self.feature_dim = feature_dim
        self.W = nn.Linear(feature_dim, feature_dim)
        self.a = nn.Linear(feature_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, node_embeddings, adj_matrix):
        N = node_embeddings.size(0)
        if N <= 1: # Handle singleton document graph
            return node_embeddings

        H = self.W(node_embeddings) 
        
        # Compute attention scores (e_ij)
        H_i = H.repeat(1, N).view(N * N, self.feature_dim)
        H_j = H.repeat(N, 1)
        
        attention_input = torch.cat([H_i, H_j], dim=1) 
        
        e = self.a(attention_input).view(N, N) 
        
        # Mask and Softmax
        e = e.masked_fill(adj_matrix == 0, float('-inf'))
        attention_weights = F.softmax(self.leakyrelu(e), dim=1) 
        
        # Aggregate
        new_embeddings = attention_weights @ H 

        return new_embeddings


# --- 6. The Complete Dual-View Hierarchical Model ---

class Dual_View_Model(nn.Module):
    def __init__(self):
        super(Dual_View_Model, self).__init__()
        feature_dim = 768
        self.feature_dim = feature_dim
        self.relation_types = ['coref', 'before', 'after', 'overlap', 'cause', 'caused', 'contain', 'contained']
        
        # Longformer base model for encoding
        self.token_embedding = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)
        self.bilstm_token = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim//2, batch_first=True, bidirectional=True)
        
        # Level 1: Paragraph-Level Dual-View GNN
        self.R_GAT_Factual = R_GAT_Layer(feature_dim, self.relation_types)
        self.R_GAT_Interpretive = R_GAT_Layer(feature_dim, self.relation_types)
        self.Cross_View_Attn = Cross_View_Attention(feature_dim)
        
        # Paragraph Aggregation 
        self.paragraph_agg = nn.Linear(feature_dim * 2, feature_dim)

        # Level 2: Document-Level GNN
        self.GAT_Document = GAT_Layer(feature_dim)
        
        # Final Sentence Classification Head (Feature Fusion)
        self.bias_sentence_head_1 = nn.Linear(feature_dim * 3, feature_dim)
        self.bias_sentence_head_2 = nn.Linear(feature_dim, 2)
        
        self.relu = nn.ReLU()
        # HYPERPARAMETER TWEAK: Use weighted CrossEntropyLoss
        self.crossentropyloss_sum = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, reduction='sum')
        
        # Distillation/Auxiliary Heads
        self.event_head_1 = nn.Linear(feature_dim, feature_dim)
        self.event_head_2 = nn.Linear(feature_dim, 2)
        
    def _get_token_embeddings(self, input_ids, attention_mask):
        """Extracts token embeddings from Longformer and passes through BiLSTM"""
        outputs = self.token_embedding(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = outputs[2]
        token_embeddings_layers = torch.stack(hidden_states, dim=0) 
        token_embeddings_layers = token_embeddings_layers[:, 0, :, :] 
        token_embeddings = torch.sum(token_embeddings_layers[-4:, :, :], dim=0) # (N_tokens, 768)
        
        # Token-level BiLSTM
        token_embeddings = token_embeddings.view(1, token_embeddings.shape[0], token_embeddings.shape[1])
        h0_token = torch.zeros(2, 1, self.feature_dim//2).to(device).requires_grad_()
        c0_token = torch.zeros(2, 1, self.feature_dim//2).to(device).requires_grad_()
        token_embeddings, (_, _) = self.bilstm_token(token_embeddings, (h0_token, c0_token))
        return token_embeddings[0, :, :] # (N_tokens, 768)

    def forward(self, batch):
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label_sentence = batch['label_sentence'][0].to(device)
        event_words = batch['event_words'][0].to(device)
        event_pairs = batch['event_pairs'][0].to(device)
        
        # --- Stage 1 & 2: Encoding and Initial Event/Sentence Embeddings ---
        
        token_embeddings = self._get_token_embeddings(input_ids, attention_mask)
        
        # A) Sentence Embeddings (using <s> token representation)
        sent_start_indices = label_sentence[:, 0]
        sentence_embeddings = token_embeddings[sent_start_indices] # (N_sentences, 768)
        
        # B) Event Embeddings 
        if event_words.size(0) > 0:
            event_embeddings = []
            for i in range(event_words.size(0)):
                start = event_words[i, 0].long()
                end = event_words[i, 1].long()
                event_embeddings.append(torch.mean(token_embeddings[start:end, :], dim=0)) 
            event_embeddings = torch.stack(event_embeddings) # (N_events, 768)
            
            # Aux Event loss (Knowledge Distillation)
            event_scores = self.event_head_2(self.relu(self.event_head_1(event_embeddings)))
            event_loss = self.crossentropyloss_sum(event_scores, event_words[:, 4:6])
        else:
            event_embeddings = torch.zeros((0, self.feature_dim)).to(device)
            event_loss = torch.tensor(0.0).to(device)

        # --- Stage 3: Two-Level Dual-View GNN ---
        
        unique_paragraph_ids = torch.unique(label_sentence[:, 4])
        paragraph_representations = torch.zeros((len(unique_paragraph_ids), self.feature_dim)).to(device)
        
        # CRITICAL: Simplify the relation labels tensor to pass only coreference to R-GAT
        # This tensor's shape is (N_pairs, 1)
        coreference_labels = batch['label_coreference'][0].to(device)[:, 1].unsqueeze(1) 


        # A) Level 1: Paragraph-Level Processing
        for p_idx, p_id in enumerate(unique_paragraph_ids):
            p_mask = (event_words[:, 8] == p_id).nonzero(as_tuple=True)[0]
            
            if p_mask.size(0) == 0:
                sent_mask = (label_sentence[:,4] == p_id).nonzero(as_tuple=True)[0]
                paragraph_representations[p_idx] = torch.mean(sentence_embeddings[sent_mask], dim=0) if sent_mask.size(0) > 0 else torch.zeros(self.feature_dim).to(device)
                continue
            
            p_events = event_embeddings[p_mask]
            
            # 2. Separate Factual (0) and Interpretive (1) Views
            F_mask = (event_words[p_mask, 7] == 0).nonzero(as_tuple=True)[0] 
            I_mask = (event_words[p_mask, 7] == 1).nonzero(as_tuple=True)[0] 
            
            F_events_embed = p_events[F_mask]
            I_events_embed = p_events[I_mask]

            # 3. Within-View R-GAT Processing (Passing only coreference labels)
            F_updated = self.R_GAT_Factual(F_events_embed, event_pairs, coreference_labels) 
            I_updated = self.R_GAT_Interpretive(I_events_embed, event_pairs, coreference_labels) 
            
            # 4. Cross-View Attention
            F_final, I_final = self.Cross_View_Attn(F_updated, I_updated)
            
            # 5. Paragraph Aggregation
            F_summary = torch.mean(F_final, dim=0) if F_final.size(0) > 0 else torch.zeros(self.feature_dim).to(device)
            I_summary = torch.mean(I_final, dim=0) if I_final.size(0) > 0 else torch.zeros(self.feature_dim).to(device)
            
            paragraph_representations[p_idx] = F.relu(self.paragraph_agg(torch.cat([F_summary, I_summary], dim=0)))
            
        # B) Level 2: Document-Level Processing
        N_para = paragraph_representations.size(0)
        adj_doc = torch.zeros((N_para, N_para)).to(device)
        
        # Document adjacency: Sequential adjacency (P_i -> P_i+1)
        for i in range(N_para - 1):
            adj_doc[i, i+1] = 1
            adj_doc[i+1, i] = 1
        adj_doc += torch.eye(N_para).to(device) # Self-loop
        
        # Apply Document GAT
        updated_paragraph_representations = self.GAT_Document(paragraph_representations, adj_doc)
        
        # --- Stage 4: Final Sentence Classification (Feature Fusion) ---
        
        # 1. Map updated paragraph reps back to sentences
        sent_to_para_indices = label_sentence[:, 4].long() 
        mapped_paragraph_context = updated_paragraph_representations[sent_to_para_indices]
        
        # 2. Event Aggregation (Mean of all initial event embeddings in sentence)
        event_aggregation_for_sent = torch.zeros_like(sentence_embeddings).to(device)
        
        for sent_idx in range(sentence_embeddings.size(0)):
            event_mask = (event_words[:, 2] == sent_idx).nonzero(as_tuple=True)[0]
            if event_mask.size(0) > 0:
                event_aggregation_for_sent[sent_idx] = torch.mean(event_embeddings[event_mask], dim=0) 
        
        # 3. Concatenation 
        final_sentence_embedding = torch.cat([
            sentence_embeddings, 
            mapped_paragraph_context, 
            event_aggregation_for_sent 
        ], dim=1) # (N_sentences, 3 * 768)

        # 4. Final Classification
        label_bias_sentence = label_sentence[:, 3].long()
        
        # --- SANITIZATION STEP ---
        # Ensure labels are valid (0 or 1) and filter out any sentence headers (label -1)
        valid_indices = (label_bias_sentence >= 0)
        
        if valid_indices.sum() == 0:
            # Handle case where no sentences have valid labels (e.g., if we only load the title)
            return torch.zeros((1, 2)).to(device), label_bias_sentence, torch.tensor(0.0).to(device), event_loss
        
        filtered_embeddings = final_sentence_embedding[valid_indices]
        filtered_labels = label_bias_sentence[valid_indices]
        
        # Ensure labels are strictly 0 or 1 for CrossEntropyLoss
        if (filtered_labels > 1).any() or (filtered_labels < 0).any():
             print(f"[Fold {args.fold}] Warning: Invalid label detected: {filtered_labels.unique()}. Setting loss to zero.")
             return torch.zeros((1, 2)).to(device), filtered_labels, torch.tensor(0.0).to(device), event_loss

        # Use filtered data for loss calculation
        bias_sentence_scores = self.bias_sentence_head_2(self.relu(self.bias_sentence_head_1(filtered_embeddings)))
        bias_sentence_loss = self.crossentropyloss_sum(bias_sentence_scores, filtered_labels)

        # Return scores and labels of the filtered set
        return bias_sentence_scores, filtered_labels, bias_sentence_loss, event_loss


# --- 7. Evaluation, Training Loop, and Execution ---

def evaluate(model, eval_dataloader):
    model.eval()
    all_decisions = []
    all_labels = []

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            # Model output returns the *filtered* label tensor
            bias_sentence_scores, label_bias_sentence_filtered, _, _ = model(batch)

        decision = torch.argmax(bias_sentence_scores, dim=1)
        all_decisions.extend(decision.cpu().numpy())
        all_labels.extend(label_bias_sentence_filtered.cpu().numpy())

    all_decisions = np.array(all_decisions)
    all_labels = np.array(all_labels)
    
    if all_labels.size == 0:
        return 0.0, 0.0, all_decisions, all_labels

    macro_F = precision_recall_fscore_support(all_labels, all_decisions, average='macro')[2]
    biased_F = precision_recall_fscore_support(all_labels, all_decisions, average='binary', zero_division=0)[2]

    return macro_F, biased_F, all_decisions, all_labels

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def run_training_and_testing():
    # Set seed
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Get fold splits (based on triplet-aware CV)
    available_files = get_available_classified_files()
    folders = create_cv_folds_triplet_aware(available_files, n_folds=10, seed=42)
    
    test_folder_index = args.fold
    dev_folder_index = (test_folder_index - 1) % 10

    test_file_paths = folders[test_folder_index]
    dev_file_paths = folders[dev_folder_index]

    train_file_paths = []
    for j in range(10):
        if j != dev_folder_index and j != test_folder_index:
            train_file_paths.extend(folders[j])

    print(f"[Fold {args.fold}] Train: {len(train_file_paths)}, Dev: {len(dev_file_paths)}, Test: {len(test_file_paths)}")

    # Initialize model
    model = Dual_View_Model()
    model.to(device)

    param_all = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('token_embedding' in n))], 
         'lr': longformer_lr, 'weight_decay': longformer_weight_decay},
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and (not 'token_embedding' in n))], 
         'lr': non_longformer_lr, 'weight_decay': non_longformer_weight_decay},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('token_embedding' in n))], 
         'lr': longformer_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and (not 'token_embedding' in n))], 
         'lr': non_longformer_lr, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)

    train_dataset = custom_dataset(train_file_paths)
    dev_dataset = custom_dataset(dev_file_paths)
    test_dataset = custom_dataset(test_file_paths)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_train_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(warmup_proportion * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    best_dev_biased_F = 0
    training_stats = []

    print(f"\n[Fold {args.fold}] Starting training...")
    print("="*70)

    for epoch_i in range(num_epochs):
        print(f"\n[Fold {args.fold}] Epoch {epoch_i+1}/{num_epochs}")
        
        model.train()
        t0 = time.time()
        total_bias_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            
            optimizer.zero_grad()

            bias_sentence_scores, label_bias_sentence, bias_sentence_loss, event_loss = model(batch)

            total_loss = bias_sentence_loss + lambda_event * event_loss
            
            # Use torch.nan_to_num to prevent NaN/Inf errors from propagating
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                 print(f"[Fold {args.fold}] Warning: NaN/Inf loss encountered in step {step}. Skipping step.")
                 continue
            
            total_bias_loss += total_loss.item()

            # Backward
            try:
                total_loss.backward()
            except RuntimeError as e:
                # This block catches OOM/memory errors during backward pass
                print(f"[Fold {args.fold}] Warning: Gradient accumulation failed in step {step}. Skipping step. Error: {e}")
                continue 

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % (len(train_dataloader) // 4) == 0 and step > 0:
                elapsed = format_time(time.time() - t0)
                print(f"  [{step}/{len(train_dataloader)}] Total Loss: {total_bias_loss/(step+1):.3f} | Bias Loss: {bias_sentence_loss.item():.3f} | {elapsed}")

        # Evaluate on dev
        macro_F, biased_F, _, _ = evaluate(model, dev_dataloader)
        elapsed = format_time(time.time() - t0)
        print(f"  Epoch {epoch_i+1} complete | Dev Macro-F1: {macro_F:.4f} | Biased-F1: {biased_F:.4f} | {elapsed}")

        training_stats.append({
            'epoch': epoch_i + 1,
            'dev_macro_F': macro_F,
            'dev_biased_F': biased_F
        })

        if biased_F > best_dev_biased_F:
            best_dev_biased_F = biased_F
            torch.save(model.state_dict(), f"{args.output_dir}/fold_{args.fold}_best.ckpt")
            print(f"  ✓ New best model saved (Biased-F1: {biased_F:.4f})")

    # Test on best model
    print(f"\n[Fold {args.fold}] Testing on held-out fold...")
    # Add map_location to ensure checkpoint loading works even if VRAM is tight
    model.load_state_dict(torch.load(f"{args.output_dir}/fold_{args.fold}_best.ckpt", map_location=device))
    macro_F, biased_F, test_decisions, test_labels = evaluate(model, test_dataloader)

    print(f"\n[Fold {args.fold}] Test Results:")
    print(f"  Macro-F1: {macro_F:.4f}")
    print(f"  Biased-F1: {biased_F:.4f}")
    print("\nClassification Report (Bias Class is 1):")
    print(classification_report(test_labels, test_decisions, digits=4, zero_division=0))

    # Save results
    results = {
        'fold': args.fold,
        'gpu': args.gpu,
        'best_dev_biased_F': best_dev_biased_F,
        'test_macro_F': macro_F,
        'test_biased_F': biased_F,
        'training_stats': training_stats,
        'classification_report': classification_report(test_labels, test_decisions, digits=4, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(test_labels, test_decisions).tolist()
    }

    with open(f"{args.output_dir}/fold_{args.fold}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Fold {args.fold}] Results saved to {args.output_dir}/fold_{args.fold}_results.json")
    print("="*70)
    
if __name__ == "__main__":
    run_training_and_testing()