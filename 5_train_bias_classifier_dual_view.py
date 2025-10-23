import os
import torch
import numpy as np
import json
import random
import logging
import time
import datetime
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerTokenizer, LongformerModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

# --- Environment & Setup ---

# Check for GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'✅ There are {torch.cuda.device_count()} GPU(s) available.')
    print(f'✅ We will use the GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Set seed for reproducibility
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    print(f"🌱 Seed set to {seed_val}")

set_seed()

# --- Configuration & Hyperparameters ---
# Paths (using scratch as requested)
BASE_OUTPUT_DIR = "/scratch/Media-Bias-Analysis/results/"
DATA_DIR = "./BASIL_event_graph_classified/" # Make sure this path is correct

# Model Hyperparameters (Adjust as needed based on your report/experiments)
MAX_LEN = 2048 # Max sequence length for Longformer
HIDDEN_DIM = 768
GAT_HEADS = 4 # Example number of attention heads for GAT
DROPOUT = 0.1
ALPHA_LEAKY = 0.2 # Alpha for LeakyReLU in GAT

# Training Hyperparameters
NUM_EPOCHS = 5 # As per report [cite: 640]
BATCH_SIZE = 1 # Keep batch size 1 for simplicity with variable graph sizes
CHECK_TIMES = 4 * NUM_EPOCHS # Evaluate 4 times per epoch
NO_DECAY = ['bias', 'LayerNorm.weight']
LONGFORMER_WEIGHT_DECAY = 1e-2
NON_LONGFORMER_WEIGHT_DECAY = 1e-2
WARMUP_PROPORTION = 0.1 # Report uses 0.0, but 0.1 is common
LONGFORMER_LR = 1e-5
NON_LONGFORMER_LR = 2e-5

# --- Helper Functions ---
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# --- Dataset Definition ---
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

class DualViewDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        print(f"🔍 Initialized dataset with {len(file_paths)} files.")

    def __len__(self):
        return len(self.file_paths)

    def _get_paragraph_grouping(self, sentences):
        # Simple heuristic: group consecutive sentences.
        # More sophisticated methods could use topic modeling or document structure.
        # For BASIL, paragraphs aren't explicitly marked. Group all as one "paragraph".
        para_groups = {}
        for i in range(len(sentences)):
            para_id = 0 # Treat whole document as one paragraph for BASIL
            if para_id not in para_groups:
                para_groups[para_id] = []
            para_groups[para_id].append(i) # Add sentence index i to paragraph 0
        return para_groups

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                article_json = json.load(f)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return None # Skip this file

        # --- 1. Tokenization and Basic Info ---
        input_ids = []
        attention_mask = []
        token_to_sentence_map = [] # Map token index to sentence index
        sentence_labels = {} # Map sentence index to bias label
        sentence_start_end_tokens = {} # Map sentence index to (start_token_idx, end_token_idx)

        # Add CLS token equivalent for Longformer
        input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['input_ids'])
        attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['attention_mask'])
        token_to_sentence_map.append(-1) # CLS token doesn't belong to a sentence

        current_token_idx = len(input_ids)
        for sent_i, sentence in enumerate(article_json['sentences']):
            sentence_text = sentence['sentence_text']
            label = sentence.get('label_info_bias', -1) # Use label_info_bias for BASIL

            # Handle title (label == -1), skip if needed or assign default
            if label == -1 and sent_i == 0: # Assuming title is first
                 label = 0 # Assign non-biased label to title for simplicity

            sentence_labels[sent_i] = label

            sent_start_token = current_token_idx
            encoded = tokenizer.encode_plus(' ' + sentence_text, add_special_tokens=False) # Add space prefix
            sent_ids = encoded['input_ids']
            sent_mask = encoded['attention_mask']

            if current_token_idx + len(sent_ids) < MAX_LEN -1: # Ensure space for EOS
                input_ids.extend(sent_ids)
                attention_mask.extend(sent_mask)
                token_to_sentence_map.extend([sent_i] * len(sent_ids))
                current_token_idx += len(sent_ids)
                sentence_start_end_tokens[sent_i] = (sent_start_token, current_token_idx)
            else:
                 logging.warning(f"Sentence {sent_i} in {file_path} truncated due to MAX_LEN.")
                 remaining_len = MAX_LEN - 1 - current_token_idx
                 if remaining_len > 0:
                     input_ids.extend(sent_ids[:remaining_len])
                     attention_mask.extend(sent_mask[:remaining_len])
                     token_to_sentence_map.extend([sent_i] * remaining_len)
                     current_token_idx += remaining_len
                     sentence_start_end_tokens[sent_i] = (sent_start_token, current_token_idx)
                 break # Stop adding sentences

        # Add EOS token
        input_ids.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['input_ids'])
        attention_mask.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['attention_mask'])
        token_to_sentence_map.append(-1)

        # Padding
        num_pad = MAX_LEN - len(input_ids)
        if num_pad > 0:
            input_ids.extend(tokenizer.encode_plus('<pad>', add_special_tokens=False)['input_ids'] * num_pad)
            attention_mask.extend([0] * num_pad) # Use 0 for padding mask
            token_to_sentence_map.extend([-1] * num_pad)

        # --- 2. Event and Relation Processing ---
        factual_events = [] # List of (event_idx_in_article, token_idx_start, token_idx_end)
        interpretive_events = []
        event_map = {} # Map article event index -> {'type': 'F'/'I', 'local_idx': index within factual/interpretive list, 'sent_idx': sentence_idx}
        event_idx_counter = 0

        # Map original token index to new input_ids index
        original_idx_to_input_idx = {}
        current_input_idx = 1 # Start after <s>
        for sent_i, sentence in enumerate(article_json['sentences']):
            if sent_i not in sentence_start_end_tokens: continue # Skip truncated sentences
            for token_info in sentence['tokens']:
                 original_token_idx = token_info['index_of_token']
                 token_text = token_info['token_text']
                 encoded = tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)
                 token_len = len(encoded['input_ids'])
                 if current_input_idx + token_len <= MAX_LEN -1:
                     original_idx_to_input_idx[original_token_idx] = (current_input_idx, current_input_idx + token_len)
                     current_input_idx += token_len
                 else:
                     # Token truncated or not included
                     break
            if current_input_idx >= MAX_LEN -1: break


        for event_token in article_json.get('event_tokens', []):
            original_token_idx = event_token['index_of_token']
            if original_token_idx not in original_idx_to_input_idx:
                continue # Skip events whose tokens were truncated

            start_idx, end_idx = original_idx_to_input_idx[original_token_idx]
            sent_idx = token_to_sentence_map[start_idx] # Find sentence index from first token
            if sent_idx == -1: continue # Should not happen for events within sentences

            event_info = {
                'token_indices': list(range(start_idx, end_idx)),
                'sent_idx': sent_idx,
                'original_idx': original_token_idx # Keep original index for relation mapping
            }

            if event_token.get('fi_classification') == 'FACTUAL':
                event_info['type'] = 'F'
                event_info['local_idx'] = len(factual_events)
                factual_events.append(event_info)
                event_map[original_token_idx] = event_info
            elif event_token.get('fi_classification') == 'INTERPRETIVE':
                event_info['type'] = 'I'
                event_info['local_idx'] = len(interpretive_events)
                interpretive_events.append(event_info)
                event_map[original_token_idx] = event_info
            # else: skip unclassified events if any

        # --- 3. Graph Construction (Adjacency Lists/Edge Indices) ---
        num_factual = len(factual_events)
        num_interpretive = len(interpretive_events)

        # Store edges as tuples: (source_local_idx, target_local_idx, relation_type_idx)
        # Relation type indices: 0:coref, 1:temp_before, 2:temp_after, 3:temp_overlap,
        #                        4:causal_cause, 5:causal_caused, 6:sub_contains, 7:sub_contained
        #                       8:cross_interprets, 9:cross_supported_by
        factual_edges = []
        interpretive_edges = []
        cross_view_edges = []

        relation_map = {
            'coreference': 0, 'temporal': {1: 1, 2: 2, 3: 3},
            'causal': {1: 4, 2: 5}, 'subevent': {1: 6, 2: 7}
        }

        for rel in article_json.get('relation_label', []):
            event1_orig_idx = rel['event_1']['index_of_token']
            event2_orig_idx = rel['event_2']['index_of_token']

            if event1_orig_idx not in event_map or event2_orig_idx not in event_map:
                continue # Skip relations involving truncated/unclassified events

            ev1_info = event_map[event1_orig_idx]
            ev2_info = event_map[event2_orig_idx]
            ev1_type, ev1_local_idx = ev1_info['type'], ev1_info['local_idx']
            ev2_type, ev2_local_idx = ev2_info['type'], ev2_info['local_idx']

            rels_to_add = []
            if rel['label_coreference'] == 1: rels_to_add.append(relation_map['coreference'])
            if rel['label_temporal'] > 0: rels_to_add.append(relation_map['temporal'][rel['label_temporal']])
            if rel['label_causal'] > 0: rels_to_add.append(relation_map['causal'][rel['label_causal']])
            if rel['label_subevent'] > 0: rels_to_add.append(relation_map['subevent'][rel['label_subevent']])

            if ev1_type == 'F' and ev2_type == 'F':
                for rel_type_idx in rels_to_add:
                    factual_edges.append((ev1_local_idx, ev2_local_idx, rel_type_idx))
                    # Add reverse edges if symmetric (coref, overlap) or needed by GNN library
                    if rel_type_idx in [0, 3]:
                         factual_edges.append((ev2_local_idx, ev1_local_idx, rel_type_idx))
            elif ev1_type == 'I' and ev2_type == 'I':
                 for rel_type_idx in rels_to_add:
                    interpretive_edges.append((ev1_local_idx, ev2_local_idx, rel_type_idx))
                    if rel_type_idx in [0, 3]:
                         interpretive_edges.append((ev2_local_idx, ev1_local_idx, rel_type_idx))
            else: # Cross-view
                # Check sentence co-occurrence (simplified)
                if ev1_info['sent_idx'] == ev2_info['sent_idx']:
                    if ev1_type == 'F': # F -> I ("interprets")
                        cross_view_edges.append((ev1_local_idx, ev2_local_idx, 8))
                    else: # I -> F ("supported-by")
                        cross_view_edges.append((ev1_local_idx, ev2_local_idx, 9))

                # Also add edge if *any* relation exists between them [cite: 522]
                if len(rels_to_add) > 0:
                    if ev1_type == 'F': # F -> I ("interprets")
                         cross_view_edges.append((ev1_local_idx, ev2_local_idx, 8))
                    else: # I -> F ("supported-by")
                         cross_view_edges.append((ev1_local_idx, ev2_local_idx, 9))


        # Convert edges to tensor format (required by PyG/DGL or manual GNNs)
        # Shape: [2, num_edges] for source/target indices, [num_edges] for types
        def edges_to_tensor(edges):
             if not edges:
                 return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
             edge_array = np.array(edges)
             edge_index = torch.tensor(edge_array[:, :2].T, dtype=torch.long)
             edge_type = torch.tensor(edge_array[:, 2], dtype=torch.long)
             return edge_index, edge_type

        factual_edge_index, factual_edge_type = edges_to_tensor(factual_edges)
        interpretive_edge_index, interpretive_edge_type = edges_to_tensor(interpretive_edges)
        # Cross-view needs careful indexing based on source/target view
        cross_f_to_i_edges = [(s, t, type) for s, t, type in cross_view_edges if type == 8]
        cross_i_to_f_edges = [(s, t, type) for s, t, type in cross_view_edges if type == 9]
        cross_f_to_i_index, cross_f_to_i_type = edges_to_tensor(cross_f_to_i_edges)
        cross_i_to_f_index, cross_i_to_f_type = edges_to_tensor(cross_i_to_f_edges)


        # --- 4. Paragraph and Document Structure ---
        # For BASIL, treat the whole document as one paragraph
        paragraph_mapping = torch.zeros(len(sentence_labels), dtype=torch.long) # All sentences belong to para 0
        num_paragraphs = 1

        # Doc graph edges (simplified: only sequential for single paragraph case)
        doc_edge_index = torch.empty((2,0), dtype=torch.long) # No edges if only 1 paragraph


        # --- 5. Final Output Dict ---
        # Filter out sentences with label -1 before converting to tensor
        valid_sent_indices = [i for i, label in sentence_labels.items() if label != -1]
        final_labels = torch.tensor([sentence_labels[i] for i in valid_sent_indices], dtype=torch.long)
        # Need to map sentence indices to their position in the final_labels tensor
        sent_idx_to_label_idx = {sent_idx: label_idx for label_idx, sent_idx in enumerate(valid_sent_indices)}

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "sentence_labels": final_labels,
            "num_factual": num_factual,
            "num_interpretive": num_interpretive,
            "factual_events": factual_events, # List of dicts with token indices
            "interpretive_events": interpretive_events,
            "factual_edge_index": factual_edge_index,
            "factual_edge_type": factual_edge_type,
            "interpretive_edge_index": interpretive_edge_index,
            "interpretive_edge_type": interpretive_edge_type,
            "cross_f_to_i_index": cross_f_to_i_index, # Factual source, Interpretive target
            "cross_i_to_f_index": cross_i_to_f_index, # Interpretive source, Factual target
            "paragraph_mapping": paragraph_mapping, # Maps sentence index -> paragraph index
            "num_paragraphs": num_paragraphs,
            "doc_edge_index": doc_edge_index,
            "sentence_start_end_tokens": sentence_start_end_tokens, # Dict: sent_idx -> (start, end)
             "sent_idx_to_label_idx": sent_idx_to_label_idx # Map original sent idx to index in sentence_labels tensor
        }


# --- Model Architecture ---

# Base Encoder (Longformer + BiLSTM) - Adapted from reference
class BaseEncoder(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)
        self.bilstm_token = nn.LSTM(input_size=HIDDEN_DIM, hidden_size=HIDDEN_DIM // 2,
                                    batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        # Longformer encoding
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        # Sum last 4 layers [cite: 1] (adaptation from reference's sum)
        hidden_states = outputs.hidden_states # Tuple of (batch, seq_len, hidden_dim)
        token_embeddings_layers = torch.stack(hidden_states[-4:], dim=0)
        token_embeddings = torch.sum(token_embeddings_layers, dim=0) # (batch, seq_len, hidden_dim)

        # BiLSTM enhancement [cite: 368]
        # Make sure input is (batch, seq_len, features)
        if token_embeddings.dim() == 2: # Add batch dim if missing (shouldn't be needed with dataloader)
             token_embeddings = token_embeddings.unsqueeze(0)

        # Packed sequence might be better for variable lengths, but requires more complex handling
        # For simplicity with batch_size=1 and padding:
        lstm_out, _ = self.bilstm_token(token_embeddings) # (batch, seq_len, hidden_dim)

        return self.dropout(lstm_out)


# Relation-Aware Graph Attention Layer (Adapted from reference's logic)
class RGATLayer(nn.Module):
     def __init__(self, in_dim, out_dim, num_relations, dropout=DROPOUT, alpha=ALPHA_LEAKY):
         super().__init__()
         self.in_dim = in_dim
         self.out_dim = out_dim
         self.num_relations = num_relations
         self.dropout = nn.Dropout(dropout)
         self.leakyrelu = nn.LeakyReLU(alpha)

         # Relation-specific transformations (Query, Key, Value, Relation)
         # Using ModuleList to hold layers for each relation type
         self.W_Q = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_relations)])
         self.W_K = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_relations)])
         self.W_V = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_relations)])
         self.W_R = nn.ModuleList([nn.Linear(in_dim * 2, out_dim) for _ in range(num_relations)]) # Takes target node + source node

         # Attention mechanism (shared across relations but applied per relation)
         self.a = nn.Linear(out_dim * 2, 1) # Attention scoring function

         self._init_weights()

     def _init_weights(self):
         for i in range(self.num_relations):
             nn.init.xavier_uniform_(self.W_Q[i].weight, gain=1.414)
             nn.init.xavier_uniform_(self.W_K[i].weight, gain=1.414)
             nn.init.xavier_uniform_(self.W_V[i].weight, gain=1.414)
             nn.init.xavier_uniform_(self.W_R[i].weight, gain=1.414)
         nn.init.xavier_uniform_(self.a.weight, gain=1.414)


     def forward(self, node_features, edge_index, edge_type):
         # node_features: (num_nodes, in_dim)
         # edge_index: (2, num_edges)
         # edge_type: (num_edges,)
         num_nodes = node_features.size(0)
         h_prime = torch.zeros_like(node_features) # (num_nodes, out_dim)

         for r in range(self.num_relations):
             # Filter edges for the current relation type
             mask = (edge_type == r)
             if mask.sum() == 0: continue # Skip if no edges of this type
             rel_edge_index = edge_index[:, mask]
             source_nodes, target_nodes = rel_edge_index[0], rel_edge_index[1]

             # Get features for source and target nodes involved in this relation
             source_features = node_features[source_nodes] # (num_rel_edges, in_dim)
             target_features = node_features[target_nodes] # (num_rel_edges, in_dim)

             # Apply relation-specific transformations
             # Relation feature: combines source and target features
             rel_feat_input = torch.cat([source_features, target_features], dim=1) # (num_rel_edges, in_dim * 2)
             rel_transformed = self.leakyrelu(self.W_R[r](rel_feat_input)) # (num_rel_edges, out_dim)

             # Key, Query, Value transformations based *only* on the relation-transformed features
             K = self.W_K[r](rel_transformed) # (num_rel_edges, out_dim)
             Q = self.W_Q[r](rel_transformed) # (num_rel_edges, out_dim) # Query based on relation context
             V = self.W_V[r](rel_transformed) # (num_rel_edges, out_dim)

             # Calculate attention scores
             # Here, Q represents the *target* node's perspective influenced by the relation,
             # and K represents the *source* node's perspective influenced by the relation.
             a_input = torch.cat([Q, K], dim=1) # (num_rel_edges, out_dim * 2)
             e = self.leakyrelu(self.a(a_input)).squeeze(1) # (num_rel_edges,)

             # Normalize attention scores using softmax over incoming edges for each target node
             # Need scatter_softmax or similar logic
             attention = torch.zeros_like(e)
             unique_targets, target_counts = torch.unique(target_nodes, return_counts=True)
             for target_idx in unique_targets:
                 target_mask = (target_nodes == target_idx)
                 e_target = e[target_mask]
                 attention[target_mask] = F.softmax(e_target, dim=0)

             attention = self.dropout(attention) # Apply dropout to attention weights

             # Aggregate neighbor features (Value vectors weighted by attention)
             # Use scatter_add to sum weighted values for each target node
             weighted_values = V * attention.unsqueeze(1) # (num_rel_edges, out_dim)
             # Accumulate message for each node 'i' based on incoming messages from 'j'
             h_prime.scatter_add_(0, target_nodes.unsqueeze(1).expand_as(weighted_values), weighted_values)

         return F.elu(h_prime) # Apply non-linearity


# Simplified Cross-View Attention
class CrossViewAttention(nn.Module):
     def __init__(self, embed_dim):
         super().__init__()
         self.q_proj = nn.Linear(embed_dim, embed_dim)
         self.k_proj = nn.Linear(embed_dim, embed_dim)
         self.v_proj = nn.Linear(embed_dim, embed_dim)
         self.out_proj = nn.Linear(embed_dim, embed_dim)
         self.scale = embed_dim ** -0.5

     def forward(self, query_nodes, key_value_nodes, edge_index):
         # query_nodes: (num_query, dim) - nodes to be updated
         # key_value_nodes: (num_kv, dim) - nodes providing context
         # edge_index: (2, num_cross_edges) - mapping query_idx -> key_value_idx
         if edge_index.numel() == 0:
             return query_nodes # No cross-view connections

         query_idx, kv_idx = edge_index[0], edge_index[1]

         Q = self.q_proj(query_nodes[query_idx]) # (num_edges, dim)
         K = self.k_proj(key_value_nodes[kv_idx]) # (num_edges, dim)
         V = self.v_proj(key_value_nodes[kv_idx]) # (num_edges, dim)

         attn_scores = (Q * K).sum(dim=-1) * self.scale # (num_edges,) Simplified dot-product

         # Softmax over incoming edges for each query node
         attention = torch.zeros_like(attn_scores)
         unique_queries, _ = torch.unique(query_idx, return_counts=True)
         for q_idx in unique_queries:
              mask = (query_idx == q_idx)
              scores_q = attn_scores[mask]
              attention[mask] = F.softmax(scores_q, dim=0)

         weighted_values = V * attention.unsqueeze(1) # (num_edges, dim)

         # Aggregate messages
         aggregated_context = torch.zeros_like(query_nodes) # (num_query, dim)
         aggregated_context.scatter_add_(0, query_idx.unsqueeze(1).expand_as(weighted_values), weighted_values)

         # Combine original query node feature with aggregated context
         # Simple addition or concatenation + projection
         updated_query_nodes = query_nodes + self.out_proj(aggregated_context) # Residual connection
         return updated_query_nodes


# Standard GAT Layer (for Document Level)
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=DROPOUT, alpha=ALPHA_LEAKY):
        super().__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False) # Attention scoring

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)


    def forward(self, node_features, edge_index):
        h = self.W(node_features) # (num_nodes, out_dim)
        num_nodes = h.size(0)

        source_nodes, target_nodes = edge_index[0], edge_index[1]

        # Calculate attention scores e_ij
        h_target = h[target_nodes] # Features of target nodes for each edge
        h_source = h[source_nodes] # Features of source nodes for each edge
        a_input = torch.cat([h_source, h_target], dim=1) # (num_edges, 2 * out_dim)
        e = self.leakyrelu(self.a(a_input)).squeeze(1) # (num_edges,)

        # Normalize using softmax (scatter_softmax equivalent)
        attention = torch.zeros_like(e)
        unique_targets, _ = torch.unique(target_nodes, return_counts=True)
        for target_idx in unique_targets:
              mask = (target_nodes == target_idx)
              e_target = e[mask]
              attention[mask] = F.softmax(e_target, dim=0)

        attention = F.dropout(attention, self.dropout, training=self.training)

        # Aggregate neighbor features
        h_prime = torch.zeros_like(h) # (num_nodes, out_dim)
        weighted_source_features = h_source * attention.unsqueeze(1) # (num_edges, out_dim)
        h_prime.scatter_add_(0, target_nodes.unsqueeze(1).expand_as(weighted_source_features), weighted_source_features)

        return F.elu(h_prime)


# --- Main Dual-View GNN Model ---
class DualViewGNN(nn.Module):
    def __init__(self, num_relations=8, dropout=DROPOUT): # 8 within-view relation types
        super().__init__()
        self.encoder = BaseEncoder(dropout=dropout)

        # Paragraph Level R-GATs (Example: 1 layer each)
        self.rgat_factual = RGATLayer(HIDDEN_DIM, HIDDEN_DIM, num_relations, dropout=dropout)
        self.rgat_interpretive = RGATLayer(HIDDEN_DIM, HIDDEN_DIM, num_relations, dropout=dropout)

        # Cross-View Attention
        self.cross_attn_f = CrossViewAttention(HIDDEN_DIM) # Update Factual based on Interpretive
        self.cross_attn_i = CrossViewAttention(HIDDEN_DIM) # Update Interpretive based on Factual

        # LayerNorms for stability
        self.norm_f1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_i1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_f2 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_i2 = nn.LayerNorm(HIDDEN_DIM)

        # Paragraph Aggregation -> Document Level Input
        self.para_aggregate_proj = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM) # Combine F and I summaries

        # Document Level GAT (Example: 1 layer)
        self.doc_gat = GATLayer(HIDDEN_DIM, HIDDEN_DIM, dropout=dropout)
        self.norm_doc = nn.LayerNorm(HIDDEN_DIM)

        # Final Classifier MLP
        # Input: Longformer Sent Emb + Updated Para Emb + Aggregated Event Emb
        classifier_input_dim = HIDDEN_DIM * 3
        self.bias_classifier_1 = nn.Linear(classifier_input_dim, HIDDEN_DIM)
        self.bias_classifier_2 = nn.Linear(HIDDEN_DIM, 2) # 2 classes: non-bias, bias
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def _get_event_embeddings(self, token_embeddings, events):
        # Average embeddings of tokens within each event mention
        event_embeddings = []
        if not events:
            return torch.empty((0, HIDDEN_DIM), device=token_embeddings.device)

        for event in events:
            # event['token_indices'] contains list of token indices in the flattened input_ids
            indices = torch.tensor(event['token_indices'], device=token_embeddings.device).long()
            if indices.numel() > 0:
                 event_embeddings.append(token_embeddings[indices].mean(dim=0))
            else:
                 # Handle empty token indices if necessary (e.g., placeholder)
                 event_embeddings.append(torch.zeros(HIDDEN_DIM, device=token_embeddings.device))
        return torch.stack(event_embeddings)


    def forward(self, batch):
        # Since batch_size=1, we access the first element
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        num_factual = batch['num_factual'].item()
        num_interpretive = batch['num_interpretive'].item()
        factual_events = batch['factual_events'] # List of dicts
        interpretive_events = batch['interpretive_events']
        factual_edge_index = batch['factual_edge_index'].squeeze(0).to(device)
        factual_edge_type = batch['factual_edge_type'].squeeze(0).to(device)
        interpretive_edge_index = batch['interpretive_edge_index'].squeeze(0).to(device)
        interpretive_edge_type = batch['interpretive_edge_type'].squeeze(0).to(device)
        cross_f_to_i_index = batch['cross_f_to_i_index'].squeeze(0).to(device)
        cross_i_to_f_index = batch['cross_i_to_f_index'].squeeze(0).to(device)
        paragraph_mapping = batch['paragraph_mapping'].squeeze(0).to(device)
        num_paragraphs = batch['num_paragraphs'].item()
        doc_edge_index = batch['doc_edge_index'].squeeze(0).to(device)
        sentence_start_end_tokens = batch['sentence_start_end_tokens'] # This is tricky with batching=1
        sent_idx_to_label_idx = batch['sent_idx_to_label_idx']

        # 1. Base Encoding
        # Squeeze batch dim since batch_size=1
        token_embeddings = self.encoder(input_ids, attention_mask).squeeze(0) # (seq_len, hidden_dim)

        # 2. Extract Initial Event Embeddings
        factual_event_embeds = self._get_event_embeddings(token_embeddings, factual_events) # (num_factual, dim)
        interpretive_event_embeds = self._get_event_embeddings(token_embeddings, interpretive_events) # (num_interp, dim)

        # --- Paragraph Level Processing (Treat whole doc as one paragraph for now) ---
        # 3. Within-View R-GAT
        updated_factual_embeds = factual_event_embeds
        if num_factual > 0 and factual_edge_index.numel() > 0:
            updated_factual_embeds = self.norm_f1(factual_event_embeds + self.rgat_factual(factual_event_embeds, factual_edge_index, factual_edge_type))

        updated_interpretive_embeds = interpretive_event_embeds
        if num_interpretive > 0 and interpretive_edge_index.numel() > 0:
            updated_interpretive_embeds = self.norm_i1(interpretive_event_embeds + self.rgat_interpretive(interpretive_event_embeds, interpretive_edge_index, interpretive_edge_type))

        # 4. Cross-View Attention
        if num_factual > 0 and num_interpretive > 0:
            factual_after_cross = self.cross_attn_f(updated_factual_embeds, updated_interpretive_embeds, cross_f_to_i_index)
            interpretive_after_cross = self.cross_attn_i(updated_interpretive_embeds, updated_factual_embeds, cross_i_to_f_index)
            # Apply LayerNorm after cross-attention + residual
            updated_factual_embeds = self.norm_f2(updated_factual_embeds + factual_after_cross)
            updated_interpretive_embeds = self.norm_i2(updated_interpretive_embeds + interpretive_after_cross)

        # 5. Aggregate Paragraph Representations
        # Since we treat the whole doc as one paragraph:
        para_factual_summary = updated_factual_embeds.mean(dim=0) if num_factual > 0 else torch.zeros(HIDDEN_DIM, device=device)
        para_interpretive_summary = updated_interpretive_embeds.mean(dim=0) if num_interpretive > 0 else torch.zeros(HIDDEN_DIM, device=device)
        para_combined = torch.cat([para_factual_summary, para_interpretive_summary], dim=0)
        paragraph_vectors = self.relu(self.para_aggregate_proj(para_combined)).unsqueeze(0) # (1, dim)

        # --- Document Level Processing ---
        # 6. Document GAT
        # With only 1 paragraph, doc GAT doesn't do much, just passes through LayerNorm
        updated_paragraph_vectors = self.norm_doc(paragraph_vectors) # (num_paras, dim)

        # --- Final Sentence Classification ---
        sentence_final_embeddings = []
        valid_sentence_indices = sorted(sent_idx_to_label_idx.keys())

        for sent_idx in valid_sentence_indices:
            # a) Longformer Sentence Embedding (e.g., average pooling tokens)
            if sent_idx in sentence_start_end_tokens:
                start, end = sentence_start_end_tokens[sent_idx]
                longformer_sent_embed = token_embeddings[start:end].mean(dim=0)
            else: # Should not happen if filtering worked
                longformer_sent_embed = torch.zeros(HIDDEN_DIM, device=device)


            # b) Updated Paragraph Context
            para_idx = paragraph_mapping[sent_idx].item() # Get the paragraph index for this sentence
            paragraph_context = updated_paragraph_vectors[para_idx]

            # c) Aggregated Event Embeddings within the sentence
            sent_factual_indices = [fe['local_idx'] for fe in factual_events if fe['sent_idx'] == sent_idx]
            sent_interpretive_indices = [ie['local_idx'] for ie in interpretive_events if ie['sent_idx'] == sent_idx]

            aggregated_event_embed = torch.zeros(HIDDEN_DIM, device=device)
            count = 0
            if sent_factual_indices:
                aggregated_event_embed += updated_factual_embeds[sent_factual_indices].sum(dim=0)
                count += len(sent_factual_indices)
            if sent_interpretive_indices:
                 aggregated_event_embed += updated_interpretive_embeds[sent_interpretive_indices].sum(dim=0)
                 count += len(sent_interpretive_indices)
            if count > 0:
                aggregated_event_embed /= count

            # Concatenate features [cite: 553]
            combined_features = torch.cat([longformer_sent_embed, paragraph_context, aggregated_event_embed], dim=0)
            sentence_final_embeddings.append(combined_features)

        if not sentence_final_embeddings:
            # Handle cases with no valid sentences (e.g., all filtered out)
            return torch.empty((0, 2), device=device)


        final_input = torch.stack(sentence_final_embeddings) # (num_valid_sentences, dim*3)

        # Classifier MLP
        logits = self.bias_classifier_2(self.dropout(self.relu(self.bias_classifier_1(final_input))))
        return logits


# --- Evaluation Function ---
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None: continue # Skip errors
            # Ensure labels are on the correct device
            labels = batch['sentence_labels'].squeeze(0).to(device)
            if labels.numel() == 0: continue # Skip articles with no valid labels


            logits = model(batch)
            if logits.shape[0] != labels.shape[0]:
                 logging.warning(f"Logits shape {logits.shape} mismatch with labels shape {labels.shape}. Skipping batch.")
                 continue # Skip if shapes don't match (e.g., empty output)


            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    if not all_labels: # Handle case where no valid batches were processed
        logging.warning("No labels collected during evaluation.")
        return avg_loss, 0.0, 0.0, {} # Return zero metrics


    # Ensure there are positive samples for binary metrics
    if sum(all_labels) == 0 or sum(all_labels) == len(all_labels):
        logging.warning("Evaluation set contains only one class. Binary metrics might be ill-defined.")
        # Calculate macro anyway, binary F1 will be 0 or nan.
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        # Handle binary F1 score gracefully
        _, _, f1_biased, _ = precision_recall_fscore_support(
             all_labels, all_preds, average='binary', pos_label=1, zero_division=0
         )
        report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    else:
         precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
             all_labels, all_preds, average='macro', zero_division=0
         )
         precision_biased, recall_biased, f1_biased, _ = precision_recall_fscore_support(
             all_labels, all_preds, average='binary', pos_label=1, zero_division=0
         )
         report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    return avg_loss, f1_macro, f1_biased, report_dict


# --- Training Loop ---
def train_model(fold_index, train_paths, dev_paths, test_paths):
    logging.info(f"\n{'='*20} Fold {fold_index} {'='*20}")
    logging.info(f"Train files: {len(train_paths)}, Dev files: {len(dev_paths)}, Test files: {len(test_paths)}")

    # Create fold-specific output directory
    fold_output_dir = os.path.join(BASE_OUTPUT_DIR, f"fold_{fold_index}")
    os.makedirs(fold_output_dir, exist_ok=True)
    log_file = os.path.join(fold_output_dir, f"train_fold_{fold_index}.log")

    # Set up file logging for this fold specifically
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler) # Add handler to root logger

    # Datasets and Dataloaders
    train_dataset = DualViewDataset(train_paths)
    dev_dataset = DualViewDataset(dev_paths)
    test_dataset = DualViewDataset(test_paths)

    # Note: Shuffle=True might break if dataset relies on order; keep False with BATCH_SIZE=1
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Optimizer, Scheduler
    model = DualViewGNN()
    model = nn.DataParallel(model) # Use DataParallel for multi-GPU
    model.to(device)

    param_all = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in NO_DECAY)) and ('longformer' in n))], 'lr': LONGFORMER_LR, 'weight_decay': LONGFORMER_WEIGHT_DECAY},
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in NO_DECAY)) and (not 'longformer' in n))], 'lr': NON_LONGFORMER_LR, 'weight_decay': NON_LONGFORMER_WEIGHT_DECAY},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in NO_DECAY)) and ('longformer' in n))], 'lr': LONGFORMER_LR, 'weight_decay': 0.0},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in NO_DECAY)) and (not 'longformer' in n))], 'lr': NON_LONGFORMER_LR, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)

    num_train_steps = NUM_EPOCHS * len(train_dataloader)
    warmup_steps = int(WARMUP_PROPORTION * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    criterion = nn.CrossEntropyLoss()
    best_dev_biased_f1 = 0.0
    best_epoch = -1

    global_step = 0
    steps_per_eval = len(train_dataloader) // (check_times // NUM_EPOCHS) if check_times > 0 else len(train_dataloader)

    # --- Epoch Loop ---
    for epoch_i in range(NUM_EPOCHS):
        logging.info(f"\n--- Epoch {epoch_i+1}/{NUM_EPOCHS} ---")
        epoch_t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch_i+1}")):
            if batch is None:
                 logging.warning(f"Skipping problematic batch at step {step}")
                 continue # Skip batch if dataset returned None
            global_step += 1

            # Move labels to device
            labels = batch['sentence_labels'].squeeze(0).to(device)
            if labels.numel() == 0:
                 logging.warning(f"Skipping batch {step} due to no valid labels.")
                 continue

            model.zero_grad()
            logits = model(batch)

            # Ensure logits and labels shapes match before loss calculation
            if logits.shape[0] != labels.shape[0]:
                 logging.error(f"Shape mismatch! Logits: {logits.shape}, Labels: {labels.shape}. Skipping batch {step}.")
                 continue


            loss = criterion(logits, labels)
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # --- Intermediate Evaluation ---
            if global_step % steps_per_eval == 0 and step > 0:
                avg_train_loss = total_train_loss / steps_per_eval
                logging.info(f"  Step {global_step}/{num_train_steps} | Avg Train Loss: {avg_train_loss:.4f} | Elapsed: {format_time(time.time() - epoch_t0)}")
                total_train_loss = 0 # Reset for next interval

                logging.info("  Running intermediate validation...")
                dev_loss, dev_macro_f1, dev_biased_f1, _ = evaluate_model(model, dev_dataloader, criterion)
                logging.info(f"  Intermediate Validation Loss: {dev_loss:.4f}")
                logging.info(f"  Intermediate Validation Macro F1: {dev_macro_f1:.4f}")
                logging.info(f"  Intermediate Validation Biased F1: {dev_biased_f1:.4f}")

                # Save best model based on Biased F1
                if dev_biased_f1 > best_dev_biased_f1:
                    logging.info(f"  🎉 New best Biased F1 found: {dev_biased_f1:.4f} (improved from {best_dev_biased_f1:.4f})")
                    best_dev_biased_f1 = dev_biased_f1
                    best_epoch = epoch_i + 1
                    # Save model state dict (handle DataParallel)
                    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save(model_state, os.path.join(fold_output_dir, f"best_model_fold_{fold_index}.ckpt"))
                    logging.info(f"  💾 Best model saved to {fold_output_dir}")
                model.train() # Set back to train mode

        # --- End of Epoch Evaluation ---
        logging.info(f"\n--- End of Epoch {epoch_i+1} ---")
        logging.info(f"Epoch Training Time: {format_time(time.time() - epoch_t0)}")
        logging.info("Running end-of-epoch validation...")
        dev_loss, dev_macro_f1, dev_biased_f1, dev_report = evaluate_model(model, dev_dataloader, criterion)
        logging.info(f"  Validation Loss: {dev_loss:.4f}")
        logging.info(f"  Validation Macro F1: {dev_macro_f1:.4f}")
        logging.info(f"  Validation Biased F1: {dev_biased_f1:.4f}")
        logging.info(f"  Validation Report:\n{json.dumps(dev_report, indent=2)}")


        if dev_biased_f1 > best_dev_biased_f1:
            logging.info(f"  🎉 New best Biased F1 found: {dev_biased_f1:.4f} (improved from {best_dev_biased_f1:.4f})")
            best_dev_biased_f1 = dev_biased_f1
            best_epoch = epoch_i + 1
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(model_state, os.path.join(fold_output_dir, f"best_model_fold_{fold_index}.ckpt"))
            logging.info(f"  💾 Best model saved to {fold_output_dir}")


    logging.info(f"\n--- Training Complete for Fold {fold_index} ---")
    logging.info(f"Best Biased F1 on Dev set: {best_dev_biased_f1:.4f} (found at epoch {best_epoch})")

    # --- Final Test Evaluation ---
    logging.info("\n--- Running Final Test Evaluation ---")
    best_model_path = os.path.join(fold_output_dir, f"best_model_fold_{fold_index}.ckpt")
    if os.path.exists(best_model_path):
        # Load the best state dict
        test_model = DualViewGNN() # Create a new instance
        test_model.load_state_dict(torch.load(best_model_path))
        test_model = nn.DataParallel(test_model) # Wrap with DataParallel if training used it
        test_model.to(device)
        logging.info(f"Loaded best model from {best_model_path}")

        test_loss, test_macro_f1, test_biased_f1, test_report = evaluate_model(test_model, test_dataloader, criterion)
        logging.info(f"  Test Loss: {test_loss:.4f}")
        logging.info(f"  Test Macro F1: {test_macro_f1:.4f}")
        logging.info(f"  Test Biased F1: {test_biased_f1:.4f}")
        logging.info(f"  Test Report:\n{json.dumps(test_report, indent=2)}")

        results = {
            'fold': fold_index,
            'best_dev_biased_f1': best_dev_biased_f1,
            'test_loss': test_loss,
            'test_macro_f1': test_macro_f1,
            'test_biased_f1': test_biased_f1,
            'test_report': test_report
        }
    else:
        logging.error("Could not find best model checkpoint to run test evaluation.")
        results = {'fold': fold_index, 'error': 'Best model checkpoint not found.'}

    # Remove the file handler for this fold to avoid logging conflicts
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

    return results

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-View GNN for Media Bias Detection")
    parser.add_argument('--start_fold', type=int, default=0, help='Fold index to start training from (0-9)')
    parser.add_argument('--end_fold', type=int, default=9, help='Fold index to end training at (0-9)')
    args = parser.parse_args()

    # --- Setup Root Logger ---
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[logging.StreamHandler()]) # Log to console

    # --- Define Folds ---
    # Using the exact folding logic from your reference script
    triples_0 = list(range(5, 105, 10))
    triples_1 = list(range(7, 107, 10))
    triples_2 = list(range(6, 106, 10))
    triples_3 = list(range(4, 104, 10))
    triples_4 = list(range(2, 102, 10))
    triples_5 = list(range(9, 109, 10))
    triples_6 = list(range(8, 108, 10))
    triples_7 = list(range(0, 100, 10))
    triples_8 = list(range(3, 103, 10))
    triples_9 = list(range(1, 101, 10))
    triples = [triples_0, triples_1, triples_2, triples_3, triples_4, triples_5, triples_6, triples_7, triples_8, triples_9]

    all_folds_paths = []
    for folder_i in range(10):
        this_folder = []
        nyt_index = folder_i
        hpo_index = folder_i + 1
        fox_index = folder_i + 2
        if nyt_index >= 10: nyt_index -= 10
        if hpo_index >= 10: hpo_index -= 10
        if fox_index >= 10: fox_index -= 10

        for triple_index in triples[nyt_index]:
            fname = os.path.join(DATA_DIR, f"basil_{triple_index}_nyt_event_graph_classified.json")
            if os.path.exists(fname): this_folder.append(fname)
        for triple_index in triples[hpo_index]:
             fname = os.path.join(DATA_DIR, f"basil_{triple_index}_hpo_event_graph_classified.json")
             if os.path.exists(fname): this_folder.append(fname)
        for triple_index in triples[fox_index]:
             fname = os.path.join(DATA_DIR, f"basil_{triple_index}_fox_event_graph_classified.json")
             if os.path.exists(fname): this_folder.append(fname)
        all_folds_paths.append(this_folder)
        logging.info(f"Fold {folder_i} includes {len(this_folder)} files.")

    # --- Run Cross-Validation ---
    all_results = []
    start_fold = max(0, args.start_fold)
    end_fold = min(9, args.end_fold)
    logging.info(f"Starting 10-fold cross-validation from fold {start_fold} to {end_fold}...")

    for i in range(start_fold, end_fold + 1):
        test_fold_index = i
        dev_fold_index = (i - 1 + 10) % 10 # Previous fold is dev

        test_paths = all_folds_paths[test_fold_index]
        dev_paths = all_folds_paths[dev_fold_index]
        train_paths = []
        for j in range(10):
            if j != test_fold_index and j != dev_fold_index:
                train_paths.extend(all_folds_paths[j])

        # Ensure no overlap
        assert not set(test_paths) & set(dev_paths)
        assert not set(test_paths) & set(train_paths)
        assert not set(dev_paths) & set(train_paths)

        fold_results = train_model(test_fold_index, train_paths, dev_paths, test_paths)
        all_results.append(fold_results)

    # --- Aggregate and Save Overall Results ---
    logging.info("\n--- Cross-Validation Summary ---")
    all_test_biased_f1 = [r.get('test_biased_f1', 0.0) for r in all_results if 'error' not in r]
    all_test_macro_f1 = [r.get('test_macro_f1', 0.0) for r in all_results if 'error' not in r]

    if all_test_biased_f1:
         avg_biased_f1 = np.mean(all_test_biased_f1)
         std_biased_f1 = np.std(all_test_biased_f1)
         avg_macro_f1 = np.mean(all_test_macro_f1)
         std_macro_f1 = np.std(all_test_macro_f1)
         logging.info(f"Average Test Biased F1: {avg_biased_f1:.4f} +/- {std_biased_f1:.4f}")
         logging.info(f"Average Test Macro F1:  {avg_macro_f1:.4f} +/- {std_macro_f1:.4f}")

         summary_results = {
            'avg_biased_f1': avg_biased_f1,
            'std_biased_f1': std_biased_f1,
            'avg_macro_f1': avg_macro_f1,
            'std_macro_f1': std_macro_f1,
            'fold_results': all_results
         }
         summary_file = os.path.join(BASE_OUTPUT_DIR, "cross_validation_summary.json")
         with open(summary_file, 'w', encoding='utf-8') as f:
             json.dump(summary_results, f, indent=2)
         logging.info(f"Overall results saved to {summary_file}")
    else:
         logging.error("No valid results collected across folds.")

    logging.info("🏁 Training Script Finished.")