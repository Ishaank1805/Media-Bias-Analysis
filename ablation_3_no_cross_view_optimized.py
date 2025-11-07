#!/usr/bin/env python3
"""
Ablation 3: No Cross-View Attention
Two-Level Dual-View hierarchy WITHOUT cross-view attention between F and I subgraphs
"""

import os
import sys
import torch
import json
import numpy as np
import random
import time
import datetime
import argparse
from pathlib import Path
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer, LongformerModel
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

# ==================== ARGUMENTS ====================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Ablation 3: No Cross-View Attention',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--edge_dropout', type=float, default=0.1)
    parser.add_argument('--contrastive_weight', type=float, default=0.3)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='./BASIL_event_graph_classified')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.debug:
        print("\n🐛 DEBUG MODE")
        args.n_folds = 1
        args.epochs = 3
        args.max_files = 30
    
    return args

args = parse_arguments()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = args.results_dir
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

print("="*70)
print("ABLATION 3: NO CROSS-VIEW ATTENTION")
print("="*70)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Folds: {args.n_folds} | Epochs: {args.epochs}")
if args.use_amp:
    print("⚡ Mixed Precision: ON")
print(f"Results: {RESULTS_DIR}")
print("="*70 + "\n")

# ==================== HYPERPARAMETERS ====================

MAX_LEN = args.max_len
num_epochs = args.epochs
batch_size = args.batch_size
CLASS_WEIGHTS = torch.tensor([1.0, 3.0]).to(device)

lambda_event = 1.0
lambda_coreference = 1.0
lambda_temporal = 1.0
lambda_causal = 1.0
lambda_subevent = 1.0
lambda_contrastive = args.contrastive_weight

no_decay = ['bias', 'LayerNorm.weight']
longformer_weight_decay = 1e-2
non_longformer_weight_decay = 1e-2
warmup_proportion = 0.1
longformer_lr = 1e-5
non_longformer_lr = 2e-5
edge_dropout_rate = args.edge_dropout

# ==================== UTILITIES ====================

def get_available_classified_files(max_files=None):
    classified_path = args.data_dir
    if not os.path.exists(classified_path):
        print(f"❌ ERROR: {classified_path} not found")
        sys.exit(1)
    
    all_files = sorted([f for f in os.listdir(classified_path) if f.endswith('_classified.json')])
    if max_files:
        all_files = all_files[:max_files]
    
    file_info = []
    for f in all_files:
        parts = f.replace('_event_graph_classified.json', '').split('_')
        if len(parts) == 3 and parts[0] == 'basil':
            try:
                file_info.append({
                    'triplet': int(parts[1]),
                    'path': f"{classified_path}/{f}"
                })
            except ValueError:
                continue
    
    print(f"Found {len(file_info)} files ({len(set([f['triplet'] for f in file_info]))} triplets)\n")
    return file_info

def create_cv_folds_triplet_aware(file_list, n_folds=10, seed=42):
    random.seed(seed)
    triplets = {}
    for info in file_list:
        t = info['triplet']
        if t not in triplets:
            triplets[t] = []
        triplets[t].append(info)
    
    triplet_nums = list(triplets.keys())
    random.shuffle(triplet_nums)
    
    folds = [[] for _ in range(n_folds)]
    for i, triplet_num in enumerate(triplet_nums):
        folds[i % n_folds].extend([info['path'] for info in triplets[triplet_num]])
    
    return folds

def detect_paragraphs(sentences):
    current_para = 0
    prev_id = None
    for i, sent in enumerate(sentences):
        if sent.get('sentence_id') == 'title':
            sent['paragraph_id'] = -1
            continue
        if 'sentence_id' in sent and isinstance(sent['sentence_id'], int):
            curr_id = sent['sentence_id']
            if prev_id is not None and isinstance(prev_id, int) and curr_id - prev_id > 1:
                current_para += 1
            prev_id = curr_id
        elif i > 0 and i % 5 == 0:
            current_para += 1
        sent['paragraph_id'] = current_para
    return sentences

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

# ==================== DATASET ====================

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

class custom_dataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], "r") as f:
            article = json.load(f)
        
        article['sentences'] = detect_paragraphs(article['sentences'])
        
        token_to_sent = {}
        token_to_para = {}
        for sent_i, sent in enumerate(article['sentences']):
            para_id = sent.get('paragraph_id', 0)
            if para_id == -1:
                para_id = 0
            for tok in sent['tokens']:
                token_to_sent[tok['index_of_token']] = sent_i
                token_to_para[tok['index_of_token']] = para_id

        input_ids = []
        attention_mask = []
        label_sentence = []
        
        for sent_i, sent_data in enumerate(article['sentences']):
            if len(sent_data.get('sentence_text', '')) > 1:
                start = len(input_ids)
                input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['input_ids'])
                attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['attention_mask'])
                end = len(input_ids)
                
                if end < MAX_LEN:
                    para_id = sent_data.get('paragraph_id', 0)
                    if para_id == -1:
                        para_id = 0
                    label_sentence.append([start, end, sent_i, sent_data.get('label_info_lex_bias', -1), para_id])
                
                for token_data in sent_data['tokens']:
                    word_enc = tokenizer.encode_plus(' ' + token_data['token_text'], add_special_tokens=False)
                    if len(input_ids) + len(word_enc['input_ids']) >= MAX_LEN:
                        break
                    input_ids.extend(word_enc['input_ids'])
                    attention_mask.extend(word_enc['attention_mask'])
                
                if len(input_ids) >= MAX_LEN:
                    break
                
                end_sent = tokenizer.encode_plus('</s>', add_special_tokens=False)
                if len(input_ids) + len(end_sent['input_ids']) < MAX_LEN:
                    input_ids.extend(end_sent['input_ids'])
                    attention_mask.extend(end_sent['attention_mask'])
                else:
                    break
        
        position_map = {}
        current_pos = 0
        for sent_i, sent_data in enumerate(article['sentences']):
            if len(sent_data.get('sentence_text', '')) <= 1:
                continue
            current_pos += 1
            for token_data in sent_data['tokens']:
                start_pos = current_pos
                word_enc = tokenizer.encode_plus(' ' + token_data['token_text'], add_special_tokens=False)
                current_pos += len(word_enc['input_ids'])
                end_pos = current_pos
                if end_pos >= MAX_LEN:
                    break
                position_map[token_data['index_of_token']] = (start_pos, end_pos)
            current_pos += 1
            if current_pos >= MAX_LEN:
                break
        
        event_words = []
        if 'event_tokens' in article:
            for event_token in article['event_tokens']:
                token_idx = event_token['index_of_token']
                if token_idx not in position_map:
                    continue
                start_pos, end_pos = position_map[token_idx]
                sent_idx = token_to_sent.get(token_idx, 0)
                para_id = token_to_para.get(token_idx, 0)
                fi_label = 1 if event_token.get('fi_classification', 'FACTUAL') == 'INTERPRETIVE' else 0
                event_words.append([
                    start_pos, end_pos, sent_idx, token_idx,
                    event_token['prob_event'][0], event_token['prob_event'][1],
                    event_token['label_event'], fi_label, para_id
                ])
        
        num_pad = MAX_LEN - len(input_ids)
        if num_pad > 0:
            input_ids.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['input_ids'])
            attention_mask.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['attention_mask'])

        input_ids = torch.tensor(input_ids[:MAX_LEN])
        attention_mask = torch.tensor(attention_mask[:MAX_LEN])
        label_sentence = torch.tensor(label_sentence)
        event_words = torch.tensor(event_words, dtype=torch.float) if len(event_words) > 0 else torch.zeros((0, 9), dtype=torch.float)
        event_words = event_words[event_words[:, 1] < MAX_LEN]
        
        event_pairs_raw = []
        label_coref_raw = []
        label_temp_raw = []
        label_caus_raw = []
        label_sub_raw = []

        if event_words.size(0) > 0 and 'relation_label' in article:
            token_to_row = {int(event_words[i, 3]): i for i in range(event_words.size(0))}
            for rel in article['relation_label']:
                e1 = rel['event_1']['index_of_token']
                e2 = rel['event_2']['index_of_token']
                if e1 in token_to_row and e2 in token_to_row:
                    event_pairs_raw.append([token_to_row[e1], token_to_row[e2]])
                    label_coref_raw.append([rel['prob_coreference'][0], rel['prob_coreference'][1], rel['label_coreference']])
                    label_temp_raw.append([rel['prob_temporal'][0], rel['prob_temporal'][1], rel['prob_temporal'][2], rel['prob_temporal'][3], rel['label_temporal']])
                    label_caus_raw.append([rel['prob_causal'][0], rel['prob_causal'][1], rel['prob_causal'][2], rel['label_causal']])
                    label_sub_raw.append([rel['prob_subevent'][0], rel['prob_subevent'][1], rel['prob_subevent'][2], rel['label_subevent']])

        if len(event_pairs_raw) == 0:
            event_pairs = torch.zeros((0, 2), dtype=torch.long)
            label_coref = torch.zeros((0, 3), dtype=torch.float)
            label_temp = torch.zeros((0, 5), dtype=torch.float)
            label_caus = torch.zeros((0, 4), dtype=torch.float)
            label_sub = torch.zeros((0, 4), dtype=torch.float)
        else:
            event_pairs = torch.tensor(event_pairs_raw, dtype=torch.long)
            label_coref = torch.tensor(label_coref_raw, dtype=torch.float)
            label_temp = torch.tensor(label_temp_raw, dtype=torch.float)
            label_caus = torch.tensor(label_caus_raw, dtype=torch.float)
            label_sub = torch.tensor(label_sub_raw, dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_sentence": label_sentence,
            "event_words": event_words,
            "event_pairs": event_pairs,
            "label_coreference": label_coref,
            "label_temporal": label_temp,
            "label_causal": label_caus,
            "label_subevent": label_sub
        }

# ==================== INNOVATIONS ====================

class AdaptiveEdgeDropout(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, event_pairs, relation_probs_dict, training=True):
        if not training or event_pairs.size(0) == 0:
            return event_pairs, relation_probs_dict
        
        importance_scores = []
        if 'coref' in relation_probs_dict and relation_probs_dict['coref'].size(0) > 0:
            importance_scores.append(relation_probs_dict['coref'][:, :2].max(dim=1)[0])
        if 'temporal' in relation_probs_dict and relation_probs_dict['temporal'].size(0) > 0:
            importance_scores.append(relation_probs_dict['temporal'][:, :4].max(dim=1)[0])
        if 'causal' in relation_probs_dict and relation_probs_dict['causal'].size(0) > 0:
            importance_scores.append(relation_probs_dict['causal'][:, :3].max(dim=1)[0])
        if 'subevent' in relation_probs_dict and relation_probs_dict['subevent'].size(0) > 0:
            importance_scores.append(relation_probs_dict['subevent'][:, :3].max(dim=1)[0])
        
        if len(importance_scores) == 0:
            return event_pairs, relation_probs_dict
        
        avg_importance = torch.stack(importance_scores).mean(dim=0)
        threshold = avg_importance.quantile(self.dropout_rate)
        keep_mask = avg_importance > threshold
        
        filtered_pairs = event_pairs[keep_mask]
        filtered_probs = {key: val[keep_mask] for key, val in relation_probs_dict.items()}
        return filtered_pairs, filtered_probs

class MultiScaleAttentionPooling(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.W_Q = nn.Linear(feature_dim, feature_dim)
        self.W_K = nn.Linear(feature_dim, feature_dim)
        self.W_V = nn.Linear(feature_dim, feature_dim)
        self.W_out = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, embeddings):
        if embeddings.size(0) == 0:
            return torch.zeros(embeddings.size(1) if len(embeddings.shape) > 1 else 768).to(embeddings.device)
        query = embeddings.mean(dim=0, keepdim=True)
        seq_len = embeddings.size(0)
        Q = self.W_Q(query).view(1, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.W_K(embeddings).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.W_V(embeddings).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=2)
        out = torch.matmul(attention, V).transpose(0, 1).contiguous().view(1, -1)
        return self.W_out(out).squeeze(0)

class DualViewContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, F_summaries, I_summaries, sentence_labels=None):
        if F_summaries.size(0) == 0 or I_summaries.size(0) == 0:
            return torch.tensor(0.0).to(F_summaries.device)
        F_norm = F.normalize(F_summaries, dim=1)
        I_norm = F.normalize(I_summaries, dim=1)
        similarity = torch.matmul(F_norm, I_norm.t()) / self.temperature
        labels = torch.arange(F_summaries.size(0)).to(similarity.device)
        return F.cross_entropy(similarity, labels)

# ==================== MODEL COMPONENTS ====================

class Token_Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.longformermodel = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)
        if args.use_amp:
            self.longformermodel.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        dev = next(self.longformermodel.parameters()).device
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        outputs = self.longformermodel(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[2]
        token_embeddings_layers = torch.stack(hidden_states, dim=0)[:, 0, :, :]
        return torch.sum(token_embeddings_layers[-4:, :, :], dim=0)

class R_GAT_Layer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.relation_types = ['coref', 'before', 'after', 'overlap', 'cause', 'caused', 'contain', 'contained']
        self.W_Q = nn.Linear(feature_dim, feature_dim)
        self.W_K = nn.Linear(feature_dim, feature_dim)
        self.W_V = nn.Linear(feature_dim, feature_dim)
        self.W_R = nn.ModuleDict({r: nn.Linear(feature_dim, feature_dim, bias=False) for r in self.relation_types})
        self.a_R = nn.ModuleDict({r: nn.Linear(feature_dim * 2, 1) for r in self.relation_types})
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, event_embeddings, event_pairs, relation_probs_dict):
        N = event_embeddings.size(0)
        if N == 0 or event_pairs.size(0) == 0:
            return event_embeddings.clone()
        
        V = self.W_V(event_embeddings)
        H_initial = event_embeddings
        relation_masks = {}
        
        if 'coref' in relation_probs_dict and relation_probs_dict['coref'].size(0) > 0:
            relation_masks['coref'] = (relation_probs_dict['coref'][:, 2] == 1).nonzero(as_tuple=True)[0]
        if 'temporal' in relation_probs_dict and relation_probs_dict['temporal'].size(0) > 0:
            temp_labels = relation_probs_dict['temporal'][:, 4].long()
            relation_masks['before'] = (temp_labels == 1).nonzero(as_tuple=True)[0]
            relation_masks['after'] = (temp_labels == 2).nonzero(as_tuple=True)[0]
            relation_masks['overlap'] = (temp_labels == 3).nonzero(as_tuple=True)[0]
        if 'causal' in relation_probs_dict and relation_probs_dict['causal'].size(0) > 0:
            causal_labels = relation_probs_dict['causal'][:, 3].long()
            relation_masks['cause'] = (causal_labels == 1).nonzero(as_tuple=True)[0]
            relation_masks['caused'] = (causal_labels == 2).nonzero(as_tuple=True)[0]
        if 'subevent' in relation_probs_dict and relation_probs_dict['subevent'].size(0) > 0:
            sub_labels = relation_probs_dict['subevent'][:, 3].long()
            relation_masks['contain'] = (sub_labels == 1).nonzero(as_tuple=True)[0]
            relation_masks['contained'] = (sub_labels == 2).nonzero(as_tuple=True)[0]
        
        new_embeddings = torch.zeros_like(event_embeddings).to(event_embeddings.device)
        for i in range(N):
            aggregated_messages = []
            for rel_type, mask in relation_masks.items():
                if mask.size(0) == 0:
                    continue
                rel_pairs = event_pairs[mask]
                source_indices = (rel_pairs[:, 1] == i).nonzero(as_tuple=True)[0]
                if len(source_indices) == 0:
                    continue
                neighbor_indices = rel_pairs[source_indices, 0]
                neighbor_embeddings = event_embeddings[neighbor_indices]
                H_neighbor_r = self.W_R[rel_type](neighbor_embeddings)
                H_i_r = H_initial[i].repeat(H_neighbor_r.size(0), 1)
                attention_input = torch.cat([H_i_r, H_neighbor_r], dim=1)
                e_r = self.a_R[rel_type](attention_input)
                attention_weights = F.softmax(self.leakyrelu(e_r), dim=0)
                aggregated_msg = torch.sum(attention_weights * V[neighbor_indices], dim=0)
                aggregated_messages.append(aggregated_msg)
            if len(aggregated_messages) > 0:
                new_embeddings[i] = torch.stack(aggregated_messages).mean(dim=0) + event_embeddings[i]
            else:
                new_embeddings[i] = event_embeddings[i]
        return new_embeddings

class GAT_Layer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.W = nn.Linear(feature_dim, feature_dim)
        self.a = nn.Linear(feature_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, node_embeddings, adj_matrix):
        N = node_embeddings.size(0)
        if N <= 1:
            return node_embeddings
        H = self.W(node_embeddings)
        H_i = H.repeat(1, N).view(N * N, self.feature_dim)
        H_j = H.repeat(N, 1)
        e = self.a(torch.cat([H_i, H_j], dim=1)).view(N, N)
        e = e.masked_fill(adj_matrix == 0, float('-inf'))
        attention_weights = F.softmax(self.leakyrelu(e), dim=1)
        return attention_weights @ H

# ==================== MAIN MODEL ====================

class Dual_View_Model(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = 768
        self.feature_dim = feature_dim
        
        self.token_embedding = Token_Embedding()
        self.bilstm_token = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim//2, 
                                     batch_first=True, bidirectional=True)
        
        self.event_head_1 = nn.Linear(feature_dim, feature_dim, bias=True)
        self.event_head_2 = nn.Linear(feature_dim, 2, bias=True)
        self.coreference_head_1 = nn.Linear(feature_dim * 4, feature_dim, bias=True)
        self.coreference_head_2 = nn.Linear(feature_dim, 2, bias=True)
        self.temporal_head_1 = nn.Linear(feature_dim * 4, feature_dim, bias=True)
        self.temporal_head_2 = nn.Linear(feature_dim, 4, bias=True)
        self.causal_head_1 = nn.Linear(feature_dim * 4, feature_dim, bias=True)
        self.causal_head_2 = nn.Linear(feature_dim, 3, bias=True)
        self.subevent_head_1 = nn.Linear(feature_dim * 4, feature_dim, bias=True)
        self.subevent_head_2 = nn.Linear(feature_dim, 3, bias=True)
        
        for m in [self.event_head_1, self.event_head_2, self.coreference_head_1, self.coreference_head_2,
                  self.temporal_head_1, self.temporal_head_2, self.causal_head_1, self.causal_head_2,
                  self.subevent_head_1, self.subevent_head_2]:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        
        self.adaptive_dropout = AdaptiveEdgeDropout(edge_dropout_rate)
        self.F_pooling = MultiScaleAttentionPooling(feature_dim, num_heads=4)
        self.I_pooling = MultiScaleAttentionPooling(feature_dim, num_heads=4)
        self.contrastive_loss = DualViewContrastiveLoss(temperature=0.07)
        
        self.R_GAT_Factual = R_GAT_Layer(feature_dim)
        self.R_GAT_Interpretive = R_GAT_Layer(feature_dim)
        self.paragraph_agg = nn.Linear(feature_dim * 2, feature_dim)
        self.GAT_Document = GAT_Layer(feature_dim)
        
        self.bias_sentence_1 = nn.Linear(feature_dim * 3, feature_dim, bias=True)
        self.bias_sentence_2 = nn.Linear(feature_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.bias_sentence_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.bias_sentence_1.bias)
        nn.init.xavier_uniform_(self.bias_sentence_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.bias_sentence_2.bias)
        
        self.relu = nn.ReLU()
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='mean')
        self.crossentropyloss_sum = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, reduction='sum')
    
    def build_paragraph_adjacency_with_coref(self, N_para, event_words, event_pairs, label_coreference):
        dev = next(self.parameters()).device
        adj = torch.eye(N_para).to(dev)
        if N_para > 1:
            for i in range(N_para - 1):
                adj[i, i+1] = 1
                adj[i+1, i] = 1
        if event_pairs.size(0) > 0 and label_coreference.size(0) > 0:
            coref_mask = (label_coreference[:, 2] == 1).nonzero(as_tuple=True)[0]
            if coref_mask.size(0) > 0:
                coref_pairs = event_pairs[coref_mask]
                for pair in coref_pairs:
                    if pair[0] < event_words.size(0) and pair[1] < event_words.size(0):
                        para1 = int(event_words[pair[0], 8])
                        para2 = int(event_words[pair[1], 8])
                        if para1 < N_para and para2 < N_para and para1 != para2:
                            adj[para1, para2] = 1
                            adj[para2, para1] = 1
        return adj
    
    def forward(self, batch):
        dev = next(self.parameters()).device
        input_ids = batch['input_ids'].to(dev)
        attention_mask = batch['attention_mask'].to(dev)
        label_sentence = batch['label_sentence'][0].to(dev)
        event_words = batch['event_words'][0].to(dev)
        event_pairs = batch['event_pairs'][0].to(dev)
        label_coreference = batch['label_coreference'][0].to(dev)
        label_temporal = batch['label_temporal'][0].to(dev)
        label_causal = batch['label_causal'][0].to(dev)
        label_subevent = batch['label_subevent'][0].to(dev)
        
        dummy_loss = torch.tensor(0.0).to(dev)
        
        token_embeddings = self.token_embedding(input_ids, attention_mask)
        token_embeddings = token_embeddings.view(1, token_embeddings.shape[0], token_embeddings.shape[1])
        h0 = torch.zeros(2, 1, self.feature_dim//2, device=token_embeddings.device).requires_grad_()
        c0 = torch.zeros(2, 1, self.feature_dim//2, device=token_embeddings.device).requires_grad_()
        token_embeddings, _ = self.bilstm_token(token_embeddings, (h0, c0))
        token_embeddings = token_embeddings[0, :, :]
        
        sent_start_indices = label_sentence[:, 0].long()
        sentence_embeddings = token_embeddings[sent_start_indices]
        
        relation_probs = {
            'coref': label_coreference,
            'temporal': label_temporal,
            'causal': label_causal,
            'subevent': label_subevent
        }
        
        event_pairs_filtered, relation_probs_filtered = self.adaptive_dropout(
            event_pairs, relation_probs, training=self.training
        )
        
        if event_words.size(0) > 0:
            event_embeddings = token_embeddings[event_words[:, 0].long()]
            event_scores = self.event_head_2(self.relu(self.event_head_1(event_embeddings)))
            event_loss = self.crossentropyloss(event_scores, event_words[:, 4:6])
            
            if event_pairs.size(0) > 0:
                event_1_emb = event_embeddings[event_pairs[:, 0].long()]
                event_2_emb = event_embeddings[event_pairs[:, 1].long()]
                pair_emb = torch.cat([event_1_emb, event_2_emb, 
                                     torch.sub(event_1_emb, event_2_emb),
                                     torch.mul(event_1_emb, event_2_emb)], dim=1)
                
                coref_scores = self.coreference_head_2(self.relu(self.coreference_head_1(pair_emb)))
                coreference_loss = self.crossentropyloss(coref_scores, label_coreference[:, :2])
                temp_scores = self.temporal_head_2(self.relu(self.temporal_head_1(pair_emb)))
                temporal_loss = self.crossentropyloss(temp_scores, label_temporal[:, :4])
                causal_scores = self.causal_head_2(self.relu(self.causal_head_1(pair_emb)))
                causal_loss = self.crossentropyloss(causal_scores, label_causal[:, :3])
                sub_scores = self.subevent_head_2(self.relu(self.subevent_head_1(pair_emb)))
                subevent_loss = self.crossentropyloss(sub_scores, label_subevent[:, :3])
            else:
                coreference_loss = temporal_loss = causal_loss = subevent_loss = dummy_loss
        else:
            event_embeddings = torch.zeros((0, self.feature_dim)).to(dev)
            event_loss = coreference_loss = temporal_loss = causal_loss = subevent_loss = dummy_loss
        
        unique_paras = torch.unique(label_sentence[:, 4])
        para_reps = torch.zeros((len(unique_paras), self.feature_dim)).to(dev)
        sentence_F_summaries = torch.zeros((sentence_embeddings.size(0), self.feature_dim)).to(dev)
        sentence_I_summaries = torch.zeros((sentence_embeddings.size(0), self.feature_dim)).to(dev)
        
        for p_idx, p_id in enumerate(unique_paras):
            p_mask = (event_words[:, 8] == p_id).nonzero(as_tuple=True)[0]
            if p_mask.size(0) == 0:
                sent_mask = (label_sentence[:, 4] == p_id).nonzero(as_tuple=True)[0]
                if sent_mask.size(0) > 0:
                    para_reps[p_idx] = torch.mean(sentence_embeddings[sent_mask], dim=0)
                continue
            
            para_events = event_embeddings[p_mask]
            F_mask = (event_words[p_mask, 7] == 0).nonzero(as_tuple=True)[0]
            I_mask = (event_words[p_mask, 7] == 1).nonzero(as_tuple=True)[0]
            
            F_events = para_events[F_mask] if F_mask.size(0) > 0 else torch.zeros((0, self.feature_dim)).to(dev)
            I_events = para_events[I_mask] if I_mask.size(0) > 0 else torch.zeros((0, self.feature_dim)).to(dev)

            F_updated = self.R_GAT_Factual(F_events, event_pairs_filtered, relation_probs_filtered)
            I_updated = self.R_GAT_Interpretive(I_events, event_pairs_filtered, relation_probs_filtered)
            
            # ABLATION: NO CROSS-VIEW ATTENTION
            F_final, I_final = F_updated, I_updated
            
            F_summary = self.F_pooling(F_final)
            I_summary = self.I_pooling(I_final)
            para_reps[p_idx] = F.relu(self.paragraph_agg(torch.cat([F_summary, I_summary], dim=0)))
            
            sent_in_para = (label_sentence[:, 4] == p_id).nonzero(as_tuple=True)[0]
            for sent_idx in sent_in_para:
                sentence_F_summaries[sent_idx] = F_summary
                sentence_I_summaries[sent_idx] = I_summary
        
        contrastive_loss = self.contrastive_loss(sentence_F_summaries, sentence_I_summaries)
        
        N_para = para_reps.size(0)
        adj_doc = self.build_paragraph_adjacency_with_coref(N_para, event_words, event_pairs, label_coreference)
        updated_para_reps = self.GAT_Document(para_reps, adj_doc)
        
        sent_to_para = label_sentence[:, 4].long()
        para_context = updated_para_reps[sent_to_para]
        
        event_agg = torch.zeros_like(sentence_embeddings).to(dev)
        if event_words.size(0) > 0:
            min_sent = int(event_words[:, 2].min())
            max_sent = int(event_words[:, 2].max())
            for sent_idx in range(min_sent, max_sent + 1):
                if sent_idx >= event_agg.size(0):
                    break
                event_mask = (event_words[:, 2] == sent_idx).nonzero(as_tuple=True)[0]
                if event_mask.size(0) > 0:
                    event_agg[sent_idx] = torch.mean(event_embeddings[event_mask], dim=0)
        
        final_sent_emb = torch.cat([sentence_embeddings, para_context, event_agg], dim=1)
        label_bias = label_sentence[:, 3].long()
        if label_bias[0] == -1:
            label_bias = label_bias[1:]
            final_sent_emb = final_sent_emb[1:, :]
        
        if label_bias.size(0) == 0 or (label_bias < 0).any() or (label_bias > 1).any():
            return (torch.zeros((1, 2)).to(dev), label_bias, dummy_loss, event_loss,
                   coreference_loss, temporal_loss, causal_loss, subevent_loss, contrastive_loss)
        
        bias_scores = self.bias_sentence_2(self.relu(self.bias_sentence_1(final_sent_emb)))
        bias_loss = self.crossentropyloss_sum(bias_scores, label_bias)
        return (bias_scores, label_bias, bias_loss, event_loss, coreference_loss, 
                temporal_loss, causal_loss, subevent_loss, contrastive_loss)

# ==================== EVALUATION ====================

def evaluate(model, dataloader):
    model.eval()
    all_decisions = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        with torch.no_grad():
            (bias_scores, labels, _, _, _, _, _, _, _) = model(batch)
        decision = torch.argmax(bias_scores, dim=1)
        all_decisions.extend(decision.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_decisions = np.array(all_decisions)
    all_labels = np.array(all_labels)
    if all_labels.size == 0:
        return 0.0, 0.0, 0.0, 0.0, all_decisions, all_labels
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_decisions, average='binary', pos_label=1, zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        all_labels, all_decisions, average='macro', zero_division=0
    )[2]
    return precision, recall, f1, macro_f1, all_decisions, all_labels

# ==================== TRAINING ====================

def train_one_fold(fold_idx, folders):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx}/{args.n_folds-1}")
    print(f"{'='*70}\n")
    
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    n_total_folds = len(folders)
    test_idx = fold_idx
    dev_idx = (test_idx - 1) % n_total_folds
    test_files = folders[test_idx]
    dev_files = folders[dev_idx]
    train_files = []
    for j in range(n_total_folds):
        if j != dev_idx and j != test_idx:
            train_files.extend(folders[j])

    print(f"Train: {len(train_files)}, Dev: {len(dev_files)}, Test: {len(test_files)}")

    model = Dual_View_Model()
    model.to(device)

    param_all = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('longformer' in n))], 
         'lr': longformer_lr, 'weight_decay': longformer_weight_decay},
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and (not 'longformer' in n))], 
         'lr': non_longformer_lr, 'weight_decay': non_longformer_weight_decay},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('longformer' in n))], 
         'lr': longformer_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and (not 'longformer' in n))], 
         'lr': non_longformer_lr, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)

    train_dataset = custom_dataset(train_files)
    dev_dataset = custom_dataset(dev_files)
    test_dataset = custom_dataset(test_files)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_train_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(warmup_proportion * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    best_dev_f1 = 0

    for epoch_i in tqdm(range(num_epochs), desc=f"Fold {fold_idx}", unit="epoch"):
        np.random.shuffle(train_files)
        train_dataset = custom_dataset(train_files)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        model.train()
        t0 = time.time()
        total_bias = 0
        total_contrastive = 0
        num_batch = 0
        
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                   desc=f"Epoch {epoch_i+1}/{num_epochs}", leave=False)
        
        for step, batch in pbar:
            with torch.cuda.amp.autocast() if args.use_amp else torch.cuda.amp.autocast(enabled=False):
                (bias_scores, labels, bias_loss, event_loss, 
                 coref_loss, temp_loss, causal_loss, sub_loss, contrastive_loss) = model(batch)

            if torch.isnan(bias_loss) or torch.isinf(bias_loss):
                continue
            
            total_bias += bias_loss.item()
            total_contrastive += contrastive_loss.item()
            num_batch += 1

            total_loss = (lambda_event * event_loss + 
                         lambda_coreference * coref_loss +
                         lambda_temporal * temp_loss +
                         lambda_causal * causal_loss +
                         lambda_subevent * sub_loss +
                         lambda_contrastive * contrastive_loss +
                         bias_loss) / args.grad_accumulation
            
            if args.use_amp:
                scaler.scale(total_loss).backward()
                if (step + 1) % args.grad_accumulation == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                total_loss.backward()
                if (step + 1) % args.grad_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            pbar.set_postfix({'loss': f'{bias_loss.item():.2f}', 'contr': f'{contrastive_loss.item():.3f}'})
        
        pbar.close()

        precision, recall, dev_f1, _, _, _ = evaluate(model, dev_dataloader)
        elapsed = format_time(time.time() - t0)
        
        if epoch_i % 3 == 0 or epoch_i == num_epochs - 1:
            avg_bias = total_bias / num_batch if num_batch > 0 else 0
            avg_cont = total_contrastive / num_batch if num_batch > 0 else 0
            print(f"  Epoch {epoch_i+1:2d}/{num_epochs} | Loss: {avg_bias:.2f} | Contr: {avg_cont:.3f} | "
                  f"P: {precision:.4f} | R: {recall:.4f} | F1: {dev_f1:.4f} | {elapsed}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(model.state_dict(), f"{RESULTS_DIR}/fold_{fold_idx}_best.ckpt")

    model.load_state_dict(torch.load(f"{RESULTS_DIR}/fold_{fold_idx}_best.ckpt", map_location=device))
    precision, recall, test_f1, macro_f1, test_dec, test_lab = evaluate(model, test_dataloader)
    print(f"[Fold {fold_idx}] TEST: P={precision:.4f}, R={recall:.4f}, F1={test_f1:.4f}")

    return {
        'fold': fold_idx,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(test_f1),
        'macro_f1': float(macro_f1),
        'predictions': test_dec.tolist(),
        'labels': test_lab.tolist()
    }

# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ABLATION 3: NO CROSS-VIEW ATTENTION")
    print("="*70 + "\n")
    
    available_files = get_available_classified_files(max_files=args.max_files)
    n_folds_for_split = 3 if args.debug else 10
    folders = create_cv_folds_triplet_aware(available_files, n_folds=n_folds_for_split, seed=args.seed)
    
    all_fold_results = []
    all_predictions = []
    all_labels = []
    
    print(f"Running {args.n_folds} folds...\n")
    
    for fold_i in range(args.n_folds):
        fold_result = train_one_fold(fold_i, folders)
        all_fold_results.append(fold_result)
        
        if fold_i == 0:
            all_predictions = fold_result['predictions']
            all_labels = fold_result['labels']
        else:
            all_predictions = np.concatenate([all_predictions, fold_result['predictions']])
            all_labels = np.concatenate([all_labels, fold_result['labels']])
    
    precisions = [f['precision'] for f in all_fold_results]
    recalls = [f['recall'] for f in all_fold_results]
    f1s = [f['f1'] for f in all_fold_results]

    avg_p = np.mean(precisions)
    std_p = np.std(precisions)
    avg_r = np.mean(recalls)
    std_r = np.std(recalls)
    avg_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    
    overall_p, overall_r, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', pos_label=1, zero_division=0
    )

    print("\n" + "="*70)
    print(f"RESULTS ({args.n_folds} FOLDS)")
    print("="*70)
    print(f"\nAverage: P={avg_p:.2f}±{std_p:.2f}, R={avg_r:.2f}±{std_r:.2f}, F1={avg_f1:.2f}±{std_f1:.2f}")
    print(f"Accumulated: P={overall_p:.2f}, R={overall_r:.2f}, F1={overall_f1:.2f}")
    
    if args.n_folds == 10:
        print(f"\nBaseline: F1=52.00 | Ablation 3: F1={avg_f1:.2f} | Diff: {avg_f1-52.00:+.2f}")
    
    print(f"\n{'Fold':<6} {'P':<10} {'R':<10} {'F1':<10}")
    print("-"*40)
    for f in all_fold_results:
        print(f"{f['fold']:<6} {f['precision']:<10.4f} {f['recall']:<10.4f} {f['f1']:<10.4f}")
    print("-"*40)
    print(f"{'Mean':<6} {avg_p:<10.4f} {avg_r:<10.4f} {avg_f1:<10.4f}")

    summary = {
        'dataset': 'BASIL',
        'n_folds': args.n_folds,
        'method': 'Ablation 3: No Cross-View Attention',
        'aggregated_metrics': {
            'precision': {'mean': float(avg_p), 'std': float(std_p)},
            'recall': {'mean': float(avg_r), 'std': float(std_r)},
            'f1': {'mean': float(avg_f1), 'std': float(std_f1)}
        },
        'per_fold_results': all_fold_results
    }
    
    with open(f"{RESULTS_DIR}/ablation_3_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results: {RESULTS_DIR}/ablation_3_results.json")
    print(f"\nFINAL: P={avg_p:.2f}±{std_p:.2f}, R={avg_r:.2f}±{std_r:.2f}, F1={avg_f1:.2f}±{std_f1:.2f}")
    print("="*70 + "\n")