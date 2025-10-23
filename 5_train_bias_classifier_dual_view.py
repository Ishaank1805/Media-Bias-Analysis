# 5_train_bias_classifier_dual_view.py

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
import subprocess # Keep subprocess if needed for other tasks, though not used in core logic here

# --- Environment & Setup ---

# Check for GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'✅ There are {torch.cuda.device_count()} GPU(s) available.')
    # Find the primary GPU name
    primary_gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
    print(f'✅ We will use the GPU: {primary_gpu_name}')
else:
    print('⚠️ No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Set seed for reproducibility
def set_seed(seed_val=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    print(f"🌱 Seed set to {seed_val}")

set_seed()

# --- Configuration & Hyperparameters ---
# Paths (using scratch as requested)
BASE_OUTPUT_DIR = "/scratch/Media-Bias-Analysis/results/"
DATA_DIR = "./BASIL_event_graph_classified/" # Make sure this path points to your classified JSONs

# Model Hyperparameters
MAX_LEN = 2048 # Max sequence length for Longformer
HIDDEN_DIM = 768 # Standard for base models like Longformer-base
GAT_HEADS = 4 # Example number of attention heads for GAT
DROPOUT = 0.1 # Common dropout rate
ALPHA_LEAKY = 0.2 # Standard alpha for LeakyReLU

# Training Hyperparameters
NUM_EPOCHS = 5 # As per report
BATCH_SIZE = 1 # Keep batch size 1 due to variable graph sizes and reference code
CHECK_TIMES = 4 * NUM_EPOCHS # Evaluate 4 times per epoch
NO_DECAY = ['bias', 'LayerNorm.weight'] # Standard practice]
LONGFORMER_WEIGHT_DECAY = 1e-2 # From reference
NON_LONGFORMER_WEIGHT_DECAY = 1e-2 # From reference
WARMUP_PROPORTION = 0.1 # Common practice, reference had 0.0
LONGFORMER_LR = 1e-5 # From reference
NON_LONGFORMER_LR = 2e-5 # From reference

# --- Helper Functions ---
def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# --- Dataset Definition ---
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

class DualViewDataset(Dataset):
    """
    Custom PyTorch Dataset for loading classified event graph data for the
    Two-Level Dual-View GNN model.
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths
        print(f"🔍 Initialized dataset with {len(file_paths)} files.")

    def __len__(self):
        return len(self.file_paths)

    def _get_paragraph_grouping(self, sentences):
        """
        Groups sentences into paragraphs.
        Currently uses a simple heuristic: treat the whole document as one paragraph
        as BASIL lacks explicit paragraph markers.
        """
        para_groups = {}
        # Simple approach for BASIL: group all sentences into paragraph 0
        para_groups[0] = list(range(len(sentences)))
        return para_groups

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                article_json = json.load(f)
        except Exception as e:
            # Log error and return None to skip this file in the DataLoader collate function
            logging.error(f"❌ Error reading or parsing {file_path}: {e}")
            return None

        # --- 1. Tokenization and Basic Info ---
        input_ids = []
        attention_mask = []
        token_to_sentence_map = [] # Map token index in input_ids -> sentence index
        sentence_labels = {} # Map sentence index -> bias label (0 or 1)
        sentence_start_end_tokens = {} # Map sentence index -> (start_token_idx, end_token_idx) in input_ids

        # Add Longformer start token '<s>'
        input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['input_ids'])
        attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['attention_mask'])
        token_to_sentence_map.append(-1) # Special index for start token

        current_token_idx = len(input_ids) # Start counting after '<s>'
        processed_sentences_count = 0
        for sent_i, sentence in enumerate(article_json['sentences']):
            sentence_text = sentence.get('sentence_text', '')
            # Use label_info_bias for BASIL, default to 0 if missing or -1 (like title)
            label = sentence.get('label_info_bias', 0)
            if label == -1: label = 0 # Treat unlabelled as non-biased

            sentence_labels[sent_i] = label

            # Add space prefix for robustness, like in reference code's tokenizer calls
            encoded = tokenizer.encode_plus(' ' + sentence_text, add_special_tokens=False)
            sent_ids = encoded['input_ids']
            sent_mask = encoded['attention_mask']

            # Check if adding this sentence exceeds MAX_LEN (leave space for '</s>')
            if current_token_idx + len(sent_ids) < MAX_LEN - 1:
                sent_start_token_idx = current_token_idx
                input_ids.extend(sent_ids)
                attention_mask.extend(sent_mask)
                token_to_sentence_map.extend([sent_i] * len(sent_ids))
                current_token_idx += len(sent_ids)
                sentence_start_end_tokens[sent_i] = (sent_start_token_idx, current_token_idx)
                processed_sentences_count += 1
            else:
                # Truncate if possible, otherwise break
                remaining_len = MAX_LEN - 1 - current_token_idx
                if remaining_len > 0:
                     sent_start_token_idx = current_token_idx
                     input_ids.extend(sent_ids[:remaining_len])
                     attention_mask.extend(sent_mask[:remaining_len])
                     token_to_sentence_map.extend([sent_i] * remaining_len)
                     current_token_idx += remaining_len
                     sentence_start_end_tokens[sent_i] = (sent_start_token_idx, current_token_idx)
                     processed_sentences_count += 1
                     logging.warning(f"⚠️ Sentence {sent_i} in {file_path} truncated.")
                else:
                     logging.warning(f"⚠️ Sentence {sent_i} and onwards in {file_path} skipped due to MAX_LEN.")
                break # Stop adding sentences

        # Add Longformer end token '</s>'
        input_ids.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['input_ids'])
        attention_mask.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['attention_mask'])
        token_to_sentence_map.append(-1) # Special index for end token

        # Padding
        num_pad = MAX_LEN - len(input_ids)
        if num_pad > 0:
            # Use tokenizer's pad token id and 0 for attention mask
            input_ids.extend([tokenizer.pad_token_id] * num_pad)
            attention_mask.extend([0] * num_pad)
            token_to_sentence_map.extend([-1] * num_pad) # Special index for padding

        # --- 2. Event and Relation Processing ---
        factual_events = [] # List of dicts {'token_indices': [start, end), 'sent_idx': int, 'original_idx': int, 'type': 'F', 'local_idx': int}
        interpretive_events = []
        event_map = {} # Map original_event_token_index -> event_info_dict

        # Map original token index from JSON to its start/end position in input_ids
        original_idx_to_input_idx = {}
        current_input_idx = 1 # Start after '<s>'
        for sent_i, sentence in enumerate(article_json['sentences']):
             if sent_i not in sentence_start_end_tokens: continue # Skip sentences not included due to truncation
             for token_info in sentence['tokens']:
                 original_token_idx = token_info['index_of_token']
                 token_text = token_info['token_text']
                 # Re-tokenize to find length in current input_ids
                 encoded = tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)
                 token_len = len(encoded['input_ids'])

                 # Check bounds based on actual tokenization length
                 potential_end_idx = current_input_idx + token_len
                 if potential_end_idx <= sentence_start_end_tokens[sent_i][1]: # Check if token fits within sentence bounds in input_ids
                     original_idx_to_input_idx[original_token_idx] = (current_input_idx, potential_end_idx)
                     current_input_idx = potential_end_idx
                 else:
                     # Token might be truncated or partially included, handle based on needs.
                     # For simplicity, we only map if fully included.
                     pass
             if current_input_idx >= MAX_LEN -1 : break # Stop if we reach max length


        # Populate factual_events, interpretive_events, and event_map
        for event_token in article_json.get('event_tokens', []):
            original_token_idx = event_token['index_of_token']
            if original_token_idx not in original_idx_to_input_idx:
                # logging.warning(f"Event token {original_token_idx} ('{event_token['token_text']}') not found in input_ids mapping for {file_path}, likely truncated.")
                continue # Skip events whose tokens were truncated/not mapped

            start_idx, end_idx = original_idx_to_input_idx[original_token_idx]
            sent_idx = token_to_sentence_map[start_idx] # Find sentence index from first token
            if sent_idx == -1: continue # Skip if somehow mapped outside sentence bounds

            event_info = {
                'token_indices': list(range(start_idx, end_idx)),
                'sent_idx': sent_idx,
                'original_idx': original_token_idx
            }

            classification = event_token.get('fi_classification')
            if classification == 'FACTUAL':
                event_info['type'] = 'F'
                event_info['local_idx'] = len(factual_events)
                factual_events.append(event_info)
                event_map[original_token_idx] = event_info
            elif classification == 'INTERPRETIVE':
                event_info['type'] = 'I'
                event_info['local_idx'] = len(interpretive_events)
                interpretive_events.append(event_info)
                event_map[original_token_idx] = event_info
            else:
                 logging.warning(f"Event token {original_token_idx} in {file_path} has unexpected classification: {classification}")


        # --- 3. Graph Construction (Adjacency Lists/Edge Indices) ---
        num_factual = len(factual_events)
        num_interpretive = len(interpretive_events)

        # Edges: (source_local_idx, target_local_idx, relation_type_idx)
        # Relation type indices: 0:coref, 1:temp_before, 2:temp_after, 3:temp_overlap,
        #                        4:causal_cause, 5:causal_caused, 6:sub_contains, 7:sub_contained
        #                       8:cross_interprets (F->I), 9:cross_supported_by (I->F)
        factual_edges = []
        interpretive_edges = []
        cross_view_edges = []

        relation_map = {
            'coreference': 0, 'temporal': {1: 1, 2: 2, 3: 3}, # Map original labels to new indices
            'causal': {1: 4, 2: 5}, 'subevent': {1: 6, 2: 7}
        }

        for rel in article_json.get('relation_label', []):
            event1_orig_idx = rel['event_1']['index_of_token']
            event2_orig_idx = rel['event_2']['index_of_token']

            # Ensure both events involved in the relation were successfully mapped
            if event1_orig_idx not in event_map or event2_orig_idx not in event_map:
                continue

            ev1_info = event_map[event1_orig_idx]
            ev2_info = event_map[event2_orig_idx]
            ev1_type, ev1_local_idx = ev1_info['type'], ev1_info['local_idx']
            ev2_type, ev2_local_idx = ev2_info['type'], ev2_info['local_idx']

            # Determine existing relations based on labels
            rels_to_add = []
            if rel['label_coreference'] == 1: rels_to_add.append(relation_map['coreference'])
            if rel['label_temporal'] in relation_map['temporal']: rels_to_add.append(relation_map['temporal'][rel['label_temporal']])
            if rel['label_causal'] in relation_map['causal']: rels_to_add.append(relation_map['causal'][rel['label_causal']])
            if rel['label_subevent'] in relation_map['subevent']: rels_to_add.append(relation_map['subevent'][rel['label_subevent']])

            # Add edges based on event types
            if ev1_type == 'F' and ev2_type == 'F':
                for rel_type_idx in rels_to_add:
                    factual_edges.append((ev1_local_idx, ev2_local_idx, rel_type_idx))
                    # Add reverse edge for symmetric relations (coref, overlap)
                    if rel_type_idx in [0, 3]:
                        factual_edges.append((ev2_local_idx, ev1_local_idx, rel_type_idx))
            elif ev1_type == 'I' and ev2_type == 'I':
                 for rel_type_idx in rels_to_add:
                    interpretive_edges.append((ev1_local_idx, ev2_local_idx, rel_type_idx))
                    if rel_type_idx in [0, 3]:
                        interpretive_edges.append((ev2_local_idx, ev1_local_idx, rel_type_idx))
            else: # Cross-view edge exists if any relation exists between them
                 if rels_to_add: # Or if they co-occur in the same sentence (Add this check if needed)
                    # Add directed cross-view edges
                    if ev1_type == 'F': # F -> I ("interprets")
                        cross_view_edges.append((ev1_local_idx, ev2_local_idx, 8))
                    else: # I -> F ("supported-by")
                        cross_view_edges.append((ev1_local_idx, ev2_local_idx, 9))

        # Add cross-view edges based on sentence co-occurrence (optional, as per report)
        sent_events_map = {} # sent_idx -> {'F': [local_f_idx,...], 'I': [local_i_idx,...]}
        for orig_idx, event_info in event_map.items():
            s_idx = event_info['sent_idx']
            e_type = event_info['type']
            loc_idx = event_info['local_idx']
            if s_idx not in sent_events_map: sent_events_map[s_idx] = {'F': [], 'I': []}
            sent_events_map[s_idx][e_type].append(loc_idx)

        for s_idx, events in sent_events_map.items():
            for f_idx in events['F']:
                for i_idx in events['I']:
                    # Add bidirectional cross-links based on co-occurrence
                    cross_view_edges.append((f_idx, i_idx, 8)) # F -> I
                    cross_view_edges.append((i_idx, f_idx, 9)) # I -> F


        # Remove duplicate edges before converting to tensor
        factual_edges = list(set(factual_edges))
        interpretive_edges = list(set(interpretive_edges))
        cross_view_edges = list(set(cross_view_edges))

        # Convert edge lists to PyTorch Geometric format [2, num_edges] and edge types [num_edges]
        def edges_to_tensor(edges):
             if not edges:
                 # Return empty tensors with correct shape if no edges exist
                 return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
             edge_array = np.array(edges)
             edge_index = torch.tensor(edge_array[:, :2].T, dtype=torch.long) # Shape [2, num_edges]
             edge_type = torch.tensor(edge_array[:, 2], dtype=torch.long) # Shape [num_edges]
             return edge_index, edge_type

        factual_edge_index, factual_edge_type = edges_to_tensor(factual_edges)
        interpretive_edge_index, interpretive_edge_type = edges_to_tensor(interpretive_edges)

        # Separate cross-view edges for easier processing in the model
        cross_f_to_i_edges = [(s, t, type) for s, t, type in cross_view_edges if type == 8]
        cross_i_to_f_edges = [(s, t, type) for s, t, type in cross_view_edges if type == 9]
        cross_f_to_i_index, _ = edges_to_tensor(cross_f_to_i_edges) # Type is implicitly 8
        cross_i_to_f_index, _ = edges_to_tensor(cross_i_to_f_edges) # Type is implicitly 9

        # --- 4. Paragraph and Document Structure ---
        # Using the simple heuristic for BASIL: one paragraph per document
        paragraph_groups = self._get_paragraph_grouping(article_json['sentences'])
        num_paragraphs = len(paragraph_groups)
        # Map sentence index to paragraph index (all map to 0 in this case)
        para_map_list = [-1] * len(article_json['sentences'])
        for para_idx, sent_indices in paragraph_groups.items():
             for sent_idx in sent_indices:
                 para_map_list[sent_idx] = para_idx
        paragraph_mapping = torch.tensor(para_map_list, dtype=torch.long)


        # Document graph edges: Simplified - no edges for a single paragraph
        doc_edge_index = torch.empty((2,0), dtype=torch.long)


        # --- 5. Final Output Dict ---
        # Filter labels for sentences that were actually processed (not truncated)
        valid_sent_indices = sorted([s_idx for s_idx in sentence_labels if s_idx in sentence_start_end_tokens and sentence_labels[s_idx] != -1])
        final_labels = torch.tensor([sentence_labels[i] for i in valid_sent_indices], dtype=torch.long)
        # Map original sent index to its index in the final_labels tensor
        sent_idx_to_label_idx = {sent_idx: label_idx for label_idx, sent_idx in enumerate(valid_sent_indices)}

        # Need to pass original sentence indices corresponding to final_labels for aggregation later
        final_label_sent_indices = torch.tensor(valid_sent_indices, dtype=torch.long)


        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sentence_labels": final_labels, # Labels only for valid, non-truncated sentences
            "final_label_sent_indices": final_label_sent_indices, # Original indices corresponding to labels
            "num_factual": num_factual,
            "num_interpretive": num_interpretive,
            "factual_events": factual_events, # List of dicts (for getting embeddings)
            "interpretive_events": interpretive_events, # List of dicts
            "factual_edge_index": factual_edge_index,
            "factual_edge_type": factual_edge_type,
            "interpretive_edge_index": interpretive_edge_index,
            "interpretive_edge_type": interpretive_edge_type,
            "cross_f_to_i_index": cross_f_to_i_index, # Shape [2, num_edges]
            "cross_i_to_f_index": cross_i_to_f_index, # Shape [2, num_edges]
            "paragraph_mapping": paragraph_mapping, # Maps original sentence index -> paragraph index
            "num_paragraphs": num_paragraphs,
            "doc_edge_index": doc_edge_index,
             "sent_idx_to_label_idx": sent_idx_to_label_idx # Dict map
        }

# Collate function to handle None values from dataset errors
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch failed
    # Since batch size is 1, just return the first valid item
    return batch[0]


# --- Model Architecture ---

# Base Encoder (Longformer + BiLSTM) - Adapted from reference
class BaseEncoder(nn.Module):
    """Encodes input text using Longformer followed by a BiLSTM."""
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)
        # Ensure hidden_size matches Longformer's output dimension if BiLSTM is used
        self.bilstm_token = nn.LSTM(input_size=self.longformer.config.hidden_size,
                                    hidden_size=self.longformer.config.hidden_size // 2,
                                    batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        print("✅ BaseEncoder initialized.")

    def forward(self, input_ids, attention_mask):
        # Longformer encoding
        # Ensure global attention is applied correctly if needed, e.g., on <s> token
        # For simplicity, using default local attention here.
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Sum last 4 layers
        hidden_states = outputs.hidden_states # Tuple of tensors (batch, seq_len, hidden_dim)
        if len(hidden_states) < 4:
             logging.warning("Longformer output less than 4 hidden states, using last state only.")
             token_embeddings = hidden_states[-1]
        else:
            token_embeddings_layers = torch.stack(hidden_states[-4:], dim=0)
            token_embeddings = torch.sum(token_embeddings_layers, dim=0) # (batch, seq_len, hidden_dim)


        # BiLSTM enhancement
        lstm_out, _ = self.bilstm_token(token_embeddings) # (batch, seq_len, hidden_dim)

        return self.dropout(lstm_out)


# Relation-Aware Graph Attention Layer
class RGATLayer(nn.Module):
    """ Relation-Aware Graph Attention Layer """
    def __init__(self, in_dim, out_dim, num_relations, dropout=DROPOUT, alpha=ALPHA_LEAKY):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.dropout_p = dropout
        self.alpha = alpha

        # Relation-specific transformations
        self.W_Q = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.W_K = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.W_V = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        # Simplified: Attention score based only on Q and K derived from nodes, relation type selects transform
        self.a = nn.Linear(2 * out_dim, 1, bias=False) # Attention scoring

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout_p)
        self._init_weights()
        print(f"✅ RGATLayer initialized (in={in_dim}, out={out_dim}, relations={num_relations}).")

    def _init_weights(self):
        for i in range(self.num_relations):
            nn.init.xavier_uniform_(self.W_Q[i].weight, gain=1.414)
            nn.init.xavier_uniform_(self.W_K[i].weight, gain=1.414)
            nn.init.xavier_uniform_(self.W_V[i].weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)

    def forward(self, node_features, edge_index, edge_type):
        num_nodes = node_features.size(0)
        h_prime = torch.zeros((num_nodes, self.out_dim), device=node_features.device)

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0: continue
            rel_edge_index = edge_index[:, mask]
            source_nodes, target_nodes = rel_edge_index[0], rel_edge_index[1]

            # Apply relation-specific transformations to source and target features
            source_feats_r = node_features[source_nodes]
            target_feats_r = node_features[target_nodes]

            Q_r = self.W_Q[r](target_feats_r) # Query based on target node
            K_r = self.W_K[r](source_feats_r) # Key based on source node
            V_r = self.W_V[r](source_feats_r) # Value based on source node

            # Calculate attention scores
            a_input = torch.cat([Q_r, K_r], dim=1) # (num_rel_edges, 2 * out_dim)
            e = self.leakyrelu(self.a(a_input)).squeeze(1) # (num_rel_edges,)

            # Normalize using scatter_softmax equivalent (manual softmax per target node)
            attention = torch.full_like(e, fill_value=-9e15) # Initialize with large negative value
            unique_targets, target_inverse_indices, target_counts = torch.unique(target_nodes, return_inverse=True, return_counts=True)

            # Efficient softmax computation using scatter_max and subtraction
            e_max = torch.scatter_reduce(input=e, dim=0, index=target_inverse_indices, reduce="amax", include_self=False)
            e_exp = torch.exp(e - e_max[target_inverse_indices]) # Subtract max for stability
            e_sum = torch.scatter_add(src=e_exp, index=target_inverse_indices, dim=0, dim_size=num_nodes)
            attention = e_exp / (e_sum[target_inverse_indices] + 1e-10) # Add epsilon for stability

            attention = self.dropout(attention) # Apply dropout

            # Aggregate neighbor features (Value vectors weighted by attention)
            weighted_values = V_r * attention.unsqueeze(1) # (num_rel_edges, out_dim)
            # Use scatter_add to sum weighted values for each target node
            h_prime.scatter_add_(0, target_nodes.unsqueeze(1).expand_as(weighted_values), weighted_values)

        return F.elu(h_prime) # Apply non-linearity


# Simplified Cross-View Attention
class CrossViewAttention(nn.Module):
    """ Simplified attention mechanism for cross-view interaction. """
    def __init__(self, embed_dim, dropout=DROPOUT):
        super().__init__()
        # Single projection for simplicity, could use separate Q/K/V
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attend = nn.Linear(embed_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(ALPHA_LEAKY)
        print(f"✅ CrossViewAttention initialized (dim={embed_dim}).")


    def forward(self, query_nodes, key_value_nodes, edge_index):
        # query_nodes: (num_query, dim) - nodes to be updated (e.g., Factual)
        # key_value_nodes: (num_kv, dim) - nodes providing context (e.g., Interpretive)
        # edge_index: (2, num_cross_edges) - mapping query_idx -> key_value_idx
        num_query_nodes = query_nodes.size(0)
        if edge_index.numel() == 0 or key_value_nodes.numel() == 0:
            return torch.zeros_like(query_nodes) # Return zero context if no edges or context nodes

        query_idx, kv_idx = edge_index[0], edge_index[1]

        # Project features
        query_proj = self.proj(query_nodes[query_idx]) # (num_edges, dim)
        kv_proj = self.proj(key_value_nodes[kv_idx])    # (num_edges, dim)

        # Calculate attention scores
        a_input = torch.cat([query_proj, kv_proj], dim=1) # (num_edges, dim * 2)
        e = self.leakyrelu(self.attend(a_input)).squeeze(1) # (num_edges,)

        # Normalize attention scores (scatter_softmax equivalent)
        attention = torch.full_like(e, fill_value=-9e15)
        unique_queries, query_inverse_indices, _ = torch.unique(query_idx, return_inverse=True, return_counts=True)
        e_max = torch.scatter_reduce(input=e, dim=0, index=query_inverse_indices, reduce="amax", include_self=False)
        e_exp = torch.exp(e - e_max[query_inverse_indices])
        e_sum = torch.scatter_add(src=e_exp, index=query_inverse_indices, dim=0, dim_size=num_query_nodes)
        attention = e_exp / (e_sum[query_inverse_indices] + 1e-10)
        attention = self.dropout(attention)

        # Aggregate weighted context (use projected key_value nodes as values)
        weighted_values = kv_proj * attention.unsqueeze(1) # (num_edges, dim)

        aggregated_context = torch.zeros((num_query_nodes, query_nodes.size(1)), device=query_nodes.device)
        aggregated_context.scatter_add_(0, query_idx.unsqueeze(1).expand_as(weighted_values), weighted_values)

        return aggregated_context # Return only the aggregated context

# Standard GAT Layer (for Document Level)
class GATLayer(nn.Module):
    """ Standard Graph Attention Layer. """
    def __init__(self, in_dim, out_dim, dropout=DROPOUT, alpha=ALPHA_LEAKY):
        super().__init__()
        self.dropout_p = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False) # Attention scoring

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout_p)
        self._init_weights()
        print(f"✅ GATLayer initialized (in={in_dim}, out={out_dim}).")


    def _init_weights(self):
         nn.init.xavier_uniform_(self.W.weight, gain=1.414)
         nn.init.xavier_uniform_(self.a.weight, gain=1.414)

    def forward(self, node_features, edge_index):
        if node_features.numel() == 0: # Handle empty graph
             return node_features

        h = self.W(node_features) # (num_nodes, out_dim)
        num_nodes = h.size(0)
        if edge_index.numel() == 0: # Handle graph with no edges
             return F.elu(h) # Apply activation and return

        source_nodes, target_nodes = edge_index[0], edge_index[1]

        # Calculate attention scores e_ij
        h_target = h[target_nodes] # Features of target nodes for each edge
        h_source = h[source_nodes] # Features of source nodes for each edge
        a_input = torch.cat([h_source, h_target], dim=1) # (num_edges, 2 * out_dim)
        e = self.leakyrelu(self.a(a_input)).squeeze(1) # (num_edges,)

        # Normalize using scatter_softmax equivalent
        attention = torch.full_like(e, fill_value=-9e15)
        unique_targets, target_inverse_indices, _ = torch.unique(target_nodes, return_inverse=True, return_counts=True)
        e_max = torch.scatter_reduce(input=e, dim=0, index=target_inverse_indices, reduce="amax", include_self=False)
        e_exp = torch.exp(e - e_max[target_inverse_indices])
        e_sum = torch.scatter_add(src=e_exp, index=target_inverse_indices, dim=0, dim_size=num_nodes)
        attention = e_exp / (e_sum[target_inverse_indices] + 1e-10) # Add epsilon for stability
        attention = self.dropout(attention)


        # Aggregate neighbor features
        h_prime = torch.zeros_like(h) # (num_nodes, out_dim)
        weighted_source_features = h_source * attention.unsqueeze(1) # (num_edges, out_dim)
        h_prime.scatter_add_(0, target_nodes.unsqueeze(1).expand_as(weighted_source_features), weighted_source_features)

        return F.elu(h_prime)


# --- Main Dual-View GNN Model ---
class DualViewGNN(nn.Module):
    """
    Implements the Two-Level Dual-View Graph Neural Network for sentence-level bias detection.
    Level 1: Paragraph-level R-GATs for Factual/Interpretive events + Cross-View Attention.
    Level 2: Document-level GAT over aggregated paragraph representations.
    Final Classifier: Combines Longformer, Paragraph, and Event features.
    """
    def __init__(self, num_relations=8, dropout=DROPOUT): # 8 within-view relation types
        super().__init__()
        self.encoder = BaseEncoder(dropout=dropout)

        # Paragraph Level R-GATs
        self.rgat_factual = RGATLayer(HIDDEN_DIM, HIDDEN_DIM, num_relations, dropout=dropout)
        self.rgat_interpretive = RGATLayer(HIDDEN_DIM, HIDDEN_DIM, num_relations, dropout=dropout)

        # Cross-View Attention
        self.cross_attn_f = CrossViewAttention(HIDDEN_DIM, dropout=dropout) # Update Factual based on Interpretive
        self.cross_attn_i = CrossViewAttention(HIDDEN_DIM, dropout=dropout) # Update Interpretive based on Factual

        # LayerNorms
        self.norm_f1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_i1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_f2 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_i2 = nn.LayerNorm(HIDDEN_DIM)

        # Paragraph Aggregation -> Document Level Input
        # Combines Factual and Interpretive summaries before projection
        self.para_aggregate_proj = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)

        # Document Level GAT
        self.doc_gat = GATLayer(HIDDEN_DIM, HIDDEN_DIM, dropout=dropout)
        self.norm_doc = nn.LayerNorm(HIDDEN_DIM)

        # Final Classifier MLP
        classifier_input_dim = HIDDEN_DIM * 3 # Longformer Sent + Updated Para + Aggregated Event
        self.bias_classifier_1 = nn.Linear(classifier_input_dim, HIDDEN_DIM)
        self.bias_classifier_2 = nn.Linear(HIDDEN_DIM, 2) # 2 classes: non-bias, bias
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        print("✅ DualViewGNN model initialized.")


    def _get_event_embeddings(self, token_embeddings, events):
        """ Averages token embeddings for each event mention. """
        event_embeddings = []
        if not events:
            return torch.empty((0, HIDDEN_DIM), device=token_embeddings.device)

        for event in events:
            indices = torch.tensor(event['token_indices'], device=token_embeddings.device, dtype=torch.long)
            # Ensure indices are within the bounds of token_embeddings
            if indices.numel() > 0 and indices.max() < token_embeddings.size(0):
                 event_embeddings.append(token_embeddings[indices].mean(dim=0))
            else:
                 # Handle empty or out-of-bounds indices
                 # logging.warning(f"Invalid indices {indices} for event {event.get('original_idx', '')}. Using zeros.")
                 event_embeddings.append(torch.zeros(HIDDEN_DIM, device=token_embeddings.device))

        # Handle case where no valid events were found
        if not event_embeddings:
            return torch.empty((0, HIDDEN_DIM), device=token_embeddings.device)

        return torch.stack(event_embeddings)


    def forward(self, batch):
        # Unpack batch - assumes batch size is 1 from collate_fn handling
        input_ids = batch['input_ids'].unsqueeze(0).to(device) # Add batch dim
        attention_mask = batch['attention_mask'].unsqueeze(0).to(device) # Add batch dim
        num_factual = batch['num_factual']
        num_interpretive = batch['num_interpretive']
        factual_events = batch['factual_events'] # List of dicts
        interpretive_events = batch['interpretive_events']
        factual_edge_index = batch['factual_edge_index'].to(device)
        factual_edge_type = batch['factual_edge_type'].to(device)
        interpretive_edge_index = batch['interpretive_edge_index'].to(device)
        interpretive_edge_type = batch['interpretive_edge_type'].to(device)
        cross_f_to_i_index = batch['cross_f_to_i_index'].to(device)
        cross_i_to_f_index = batch['cross_i_to_f_index'].to(device)
        paragraph_mapping = batch['paragraph_mapping'].to(device)
        num_paragraphs = batch['num_paragraphs']
        doc_edge_index = batch['doc_edge_index'].to(device)
        final_label_sent_indices = batch['final_label_sent_indices'].to(device) # Original indices of labelled sentences

        # 1. Base Encoding
        token_embeddings = self.encoder(input_ids, attention_mask).squeeze(0) # (seq_len, hidden_dim)

        # 2. Extract Initial Event Embeddings
        factual_event_embeds = self._get_event_embeddings(token_embeddings, factual_events) # (num_factual, dim)
        interpretive_event_embeds = self._get_event_embeddings(token_embeddings, interpretive_events) # (num_interp, dim)

        # --- Paragraph Level Processing ---
        # NOTE: For BASIL, this operates on the whole document as one paragraph.
        # This structure allows extension to multi-paragraph documents later.

        # 3. Within-View R-GAT + Residual + Norm
        updated_factual_embeds = factual_event_embeds
        if num_factual > 0 and factual_edge_index.numel() > 0:
            factual_updates = self.rgat_factual(factual_event_embeds, factual_edge_index, factual_edge_type)
            updated_factual_embeds = self.norm_f1(factual_event_embeds + factual_updates)

        updated_interpretive_embeds = interpretive_event_embeds
        if num_interpretive > 0 and interpretive_edge_index.numel() > 0:
            interpretive_updates = self.rgat_interpretive(interpretive_event_embeds, interpretive_edge_index, interpretive_edge_type)
            updated_interpretive_embeds = self.norm_i1(interpretive_event_embeds + interpretive_updates)

        # 4. Cross-View Attention + Residual + Norm
        factual_context = torch.zeros_like(updated_factual_embeds)
        interpretive_context = torch.zeros_like(updated_interpretive_embeds)
        if num_factual > 0 and num_interpretive > 0:
             factual_context = self.cross_attn_f(updated_factual_embeds, updated_interpretive_embeds, cross_i_to_f_index) # F gets context from I via I->F edges
             interpretive_context = self.cross_attn_i(updated_interpretive_embeds, updated_factual_embeds, cross_f_to_i_index) # I gets context from F via F->I edges

        if num_factual > 0:
             updated_factual_embeds = self.norm_f2(updated_factual_embeds + factual_context)
        if num_interpretive > 0:
             updated_interpretive_embeds = self.norm_i2(updated_interpretive_embeds + interpretive_context)


        # 5. Aggregate Paragraph Representations (Mean pooling events within the paragraph)
        # For single paragraph case:
        para_factual_summary = updated_factual_embeds.mean(dim=0, keepdim=True) if num_factual > 0 else torch.zeros((1, HIDDEN_DIM), device=device)
        para_interpretive_summary = updated_interpretive_embeds.mean(dim=0, keepdim=True) if num_interpretive > 0 else torch.zeros((1, HIDDEN_DIM), device=device)
        para_combined = torch.cat([para_factual_summary, para_interpretive_summary], dim=1) # (1, dim*2)
        paragraph_vectors = self.relu(self.para_aggregate_proj(para_combined)) # (1, dim)


        # --- Document Level Processing ---
        # 6. Document GAT + Residual + Norm
        # If num_paragraphs > 1 and doc_edge_index exists, GAT updates paragraph_vectors
        if num_paragraphs > 1 and doc_edge_index.numel() > 0:
             doc_gat_updates = self.doc_gat(paragraph_vectors, doc_edge_index)
             updated_paragraph_vectors = self.norm_doc(paragraph_vectors + doc_gat_updates) # (num_paras, dim)
        else:
             # If only one paragraph or no doc edges, just normalize
             updated_paragraph_vectors = self.norm_doc(paragraph_vectors) # (num_paras, dim)


        # --- Final Sentence Classification ---
        sentence_final_embeddings = []

        # Iterate through the *original indices* of sentences that have labels
        for label_idx, original_sent_idx in enumerate(final_label_sent_indices):
            original_sent_idx = original_sent_idx.item() # Convert tensor to int

            # a) Longformer Sentence Embedding (using sentence start/end tokens)
            sent_token_indices = []
            # Find all token indices belonging to this original sentence index
            for token_idx, s_idx in enumerate(token_to_sentence_map):
                if s_idx == original_sent_idx:
                    sent_token_indices.append(token_idx)

            if sent_token_indices:
                 longformer_sent_embed = token_embeddings[torch.tensor(sent_token_indices, device=device)].mean(dim=0)
            else:
                 longformer_sent_embed = torch.zeros(HIDDEN_DIM, device=device)

            # b) Updated Paragraph Context
            # Find the paragraph this sentence belongs to
            para_idx = paragraph_mapping[original_sent_idx].item()
            # Ensure para_idx is valid before indexing
            if 0 <= para_idx < updated_paragraph_vectors.size(0):
                 paragraph_context = updated_paragraph_vectors[para_idx]
            else:
                 # Handle potential index out of bounds if paragraph mapping is inconsistent
                 logging.warning(f"Invalid paragraph index {para_idx} for sentence {original_sent_idx}. Using zero vector.")
                 paragraph_context = torch.zeros(HIDDEN_DIM, device=device)


            # c) Aggregated Event Embeddings within the sentence
            sent_factual_indices = [fe['local_idx'] for fe in factual_events if fe['sent_idx'] == original_sent_idx]
            sent_interpretive_indices = [ie['local_idx'] for ie in interpretive_events if ie['sent_idx'] == original_sent_idx]

            aggregated_event_embed = torch.zeros(HIDDEN_DIM, device=device)
            count = 0
            if sent_factual_indices and num_factual > 0:
                 # Ensure indices are valid
                 valid_f_indices = [idx for idx in sent_factual_indices if idx < updated_factual_embeds.size(0)]
                 if valid_f_indices:
                      aggregated_event_embed += updated_factual_embeds[torch.tensor(valid_f_indices, device=device, dtype=torch.long)].sum(dim=0)
                      count += len(valid_f_indices)
            if sent_interpretive_indices and num_interpretive > 0:
                 valid_i_indices = [idx for idx in sent_interpretive_indices if idx < updated_interpretive_embeds.size(0)]
                 if valid_i_indices:
                      aggregated_event_embed += updated_interpretive_embeds[torch.tensor(valid_i_indices, device=device, dtype=torch.long)].sum(dim=0)
                      count += len(valid_i_indices)
            if count > 0:
                aggregated_event_embed /= count

            # Concatenate features
            combined_features = torch.cat([longformer_sent_embed, paragraph_context, aggregated_event_embed], dim=0)
            sentence_final_embeddings.append(combined_features)

        if not sentence_final_embeddings:
            logging.warning(f"No valid sentence embeddings generated for file {idx}.")
            # Return empty tensor with correct shape for CrossEntropyLoss compatibility
            return torch.empty((0, 2), device=device)

        final_input = torch.stack(sentence_final_embeddings) # (num_valid_sentences, dim*3)

        # Classifier MLP
        hidden = self.dropout(self.relu(self.bias_classifier_1(final_input)))
        logits = self.bias_classifier_2(hidden) # (num_valid_sentences, 2)
        return logits


# --- Evaluation Function ---
def evaluate_model(model, dataloader, criterion):
    """ Evaluates the model on a given dataset loader. """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None: continue # Skip errors from dataset loading

            labels = batch['sentence_labels'].to(device) # Already filtered in dataset
            if labels.numel() == 0: continue # Skip articles with no valid labels

            logits = model(batch)
            if logits.numel() == 0 or logits.shape[0] != labels.shape[0]:
                 logging.warning(f"Logits shape mismatch ({logits.shape}) or empty logits for labels ({labels.shape}). Skipping batch.")
                 continue

            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0) # Weight loss by number of sentences

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    if not all_labels: # Handle case where no valid batches were processed
        logging.warning("⚠️ No valid labels collected during evaluation. Returning zero metrics.")
        return 0.0, 0.0, 0.0, classification_report(all_labels, all_preds, output_dict=True, zero_division=0) # Return empty report

    avg_loss = total_loss / len(all_labels)

    # Calculate metrics with zero_division=0 to handle cases with no predicted/true samples for a class
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    # Calculate binary metrics specifically for the 'biased' class (label 1)
    precision_biased, recall_biased, f1_biased, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1, zero_division=0
    )
    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    return avg_loss, f1_macro, f1_biased, report_dict


# --- Training Loop ---
def train_model(fold_index, train_paths, dev_paths, test_paths):
    """ Trains and evaluates the model for a single cross-validation fold. """
    fold_start_time = time.time()
    # Ensure fold index is passed correctly
    fold_output_dir = os.path.join(BASE_OUTPUT_DIR, f"fold_{fold_index}")
    os.makedirs(fold_output_dir, exist_ok=True)
    log_file = os.path.join(fold_output_dir, f"train_fold_{fold_index}.log")

    # === Setup Fold-Specific Logging ===
    fold_logger = logging.getLogger(f"Fold_{fold_index}")
    fold_logger.setLevel(logging.INFO)
    # Remove existing handlers to prevent duplicate logging if re-running
    for handler in fold_logger.handlers[:]:
        fold_logger.removeHandler(handler)

    # Console handler (uses root logger's formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.getLogger().handlers[0].formatter)
    fold_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w') # Overwrite log file each time
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Fold %(fold)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    fold_logger.addHandler(file_handler)
    # Add fold index to log records using LoggerAdapter
    adapter = logging.LoggerAdapter(fold_logger, {'fold': fold_index})
    adapter.info(f"\n{'='*25} Starting Fold {fold_index} {'='*25}")
    adapter.info(f"📁 Output Dir: {fold_output_dir}")
    adapter.info(f"🚂 Train files: {len(train_paths)}, 🧑‍💻 Dev files: {len(dev_paths)}, 🧪 Test files: {len(test_paths)}")


    # Datasets and Dataloaders
    train_dataset = DualViewDataset(train_paths)
    dev_dataset = DualViewDataset(dev_paths)
    test_dataset = DualViewDataset(test_paths)

    # Use collate_fn to handle potential None values from dataset errors
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn) # Shuffle train data
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model, Optimizer, Scheduler
    model = DualViewGNN()
    # Apply DataParallel for multi-GPU usage
    if torch.cuda.device_count() > 1:
        adapter.info(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    model.to(device)

    # Optimizer setup (from reference)
    param_all = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in NO_DECAY)) and ('encoder.longformer' in n))], 'lr': LONGFORMER_LR, 'weight_decay': LONGFORMER_WEIGHT_DECAY},
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in NO_DECAY)) and (not 'encoder.longformer' in n))], 'lr': NON_LONGFORMER_LR, 'weight_decay': NON_LONGFORMER_WEIGHT_DECAY},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in NO_DECAY)) and ('encoder.longformer' in n))], 'lr': LONGFORMER_LR, 'weight_decay': 0.0},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in NO_DECAY)) and (not 'encoder.longformer' in n))], 'lr': NON_LONGFORMER_LR, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)

    # Calculate total steps, ensuring it's at least 1
    num_train_batches = len(train_dataloader)
    if num_train_batches == 0:
         adapter.error("❌ Train dataloader is empty. Check data paths and filtering.")
         # Clean up logger handlers
         for handler in fold_logger.handlers[:]: fold_logger.removeHandler(handler)
         file_handler.close()
         return {'fold': fold_index, 'error': 'Empty train dataloader.'}

    num_train_steps = NUM_EPOCHS * num_train_batches
    warmup_steps = int(WARMUP_PROPORTION * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    criterion = nn.CrossEntropyLoss()
    best_dev_biased_f1 = -1.0 # Initialize to handle cases where F1 is 0
    best_epoch = -1

    global_step = 0
    # Ensure steps_per_eval is at least 1
    steps_per_eval = max(1, num_train_batches // (CHECK_TIMES // NUM_EPOCHS)) if CHECK_TIMES > 0 else num_train_batches

    # --- Epoch Loop ---
    for epoch_i in range(NUM_EPOCHS):
        adapter.info(f"\n--- Epoch {epoch_i+1}/{NUM_EPOCHS} ---")
        epoch_t0 = time.time()
        total_train_loss = 0
        num_train_samples = 0 # Count actual processed samples
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch_i+1}")
        for step, batch in enumerate(batch_iterator):
            if batch is None:
                 adapter.warning(f"Skipping problematic batch at step {step}")
                 continue

            labels = batch['sentence_labels'].to(device)
            if labels.numel() == 0:
                 adapter.warning(f"Skipping batch {step} due to no valid labels.")
                 continue

            global_step += 1
            model.zero_grad()
            logits = model(batch)

            if logits.numel() == 0 or logits.shape[0] != labels.shape[0]:
                 adapter.error(f"Shape mismatch or empty logits! Logits: {logits.shape}, Labels: {labels.shape}. Skipping batch {step}.")
                 continue

            loss = criterion(logits, labels)
            # Accumulate loss weighted by the number of sentences in the batch
            batch_loss = loss.item() * labels.size(0)
            total_train_loss += batch_loss
            num_train_samples += labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            batch_iterator.set_postfix({'loss': f'{loss.item():.4f}'}) # Show loss for current batch

            # --- Intermediate Evaluation ---
            if global_step % steps_per_eval == 0 and step > 0:
                avg_interval_train_loss = total_train_loss / num_train_samples if num_train_samples > 0 else 0
                adapter.info(f"  Step {global_step}/{num_train_steps} | Avg Train Loss (last {steps_per_eval} steps): {avg_interval_train_loss:.4f} | Elapsed: {format_time(time.time() - epoch_t0)}")
                total_train_loss = 0 # Reset interval loss
                num_train_samples = 0

                adapter.info("  Running intermediate validation...")
                dev_loss, dev_macro_f1, dev_biased_f1, _ = evaluate_model(model, dev_dataloader, criterion)
                adapter.info(f"  Intermediate Validation Results -> Loss: {dev_loss:.4f}, Macro F1: {dev_macro_f1:.4f}, Biased F1: {dev_biased_f1:.4f}")

                if dev_biased_f1 > best_dev_biased_f1:
                    adapter.info(f"  🎉 New best Biased F1: {dev_biased_f1:.4f} (Prev best: {best_dev_biased_f1:.4f})")
                    best_dev_biased_f1 = dev_biased_f1
                    best_epoch = epoch_i + 1
                    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                    torch.save(model_to_save.state_dict(), os.path.join(fold_output_dir, f"best_model_fold_{fold_index}.ckpt"))
                    adapter.info(f"  💾 Best model checkpoint saved.")
                model.train() # Ensure model is back in training mode

        # --- End of Epoch Evaluation ---
        adapter.info(f"\n--- End of Epoch {epoch_i+1} ---")
        epoch_duration = format_time(time.time() - epoch_t0)
        adapter.info(f"Epoch Training Time: {epoch_duration}")
        adapter.info("Running end-of-epoch validation...")
        dev_loss, dev_macro_f1, dev_biased_f1, dev_report = evaluate_model(model, dev_dataloader, criterion)
        adapter.info(f"  Validation Loss: {dev_loss:.4f}")
        adapter.info(f"  Validation Macro F1: {dev_macro_f1:.4f}")
        adapter.info(f"  Validation Biased F1: {dev_biased_f1:.4f}")
        # Log the classification report dictionary neatly
        adapter.info(f"  Validation Report:\n{json.dumps(dev_report, indent=4)}")


        if dev_biased_f1 > best_dev_biased_f1:
            adapter.info(f"  🎉 New best Biased F1 found: {dev_biased_f1:.4f} (Prev best: {best_dev_biased_f1:.4f})")
            best_dev_biased_f1 = dev_biased_f1
            best_epoch = epoch_i + 1
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), os.path.join(fold_output_dir, f"best_model_fold_{fold_index}.ckpt"))
            adapter.info(f"  💾 Best model checkpoint saved.")

    adapter.info(f"\n🏁 Training Complete for Fold {fold_index}")
    adapter.info(f"Best Biased F1 on Dev set: {best_dev_biased_f1:.4f} (found at epoch {best_epoch})")

    # --- Final Test Evaluation ---
    adapter.info("\n🧪 Running Final Test Evaluation on Best Model 🧪")
    best_model_path = os.path.join(fold_output_dir, f"best_model_fold_{fold_index}.ckpt")
    results = {'fold': fold_index, 'best_dev_biased_f1': best_dev_biased_f1, 'best_epoch': best_epoch}
    if os.path.exists(best_model_path):
        try:
            # Load the best state dict into a fresh model instance on CPU first
            test_model_state_dict = torch.load(best_model_path, map_location='cpu')
            test_model = DualViewGNN() # Create base model instance
            # Load state dict (handle potential mismatches, e.g. if saved from DataParallel)
            # Adjust keys if saved model was wrapped in DataParallel
            if next(iter(test_model_state_dict)).startswith('module.'):
                 test_model_state_dict = {k.partition('module.')[2]: v for k, v in test_model_state_dict.items()}
            test_model.load_state_dict(test_model_state_dict)

            # Wrap in DataParallel *after* loading state_dict if multiple GPUs are available
            if torch.cuda.device_count() > 1:
                test_model = nn.DataParallel(test_model)
            test_model.to(device) # Move model (or wrapped model) to device
            adapter.info(f"✅ Successfully loaded best model from {best_model_path}")

            test_loss, test_macro_f1, test_biased_f1, test_report = evaluate_model(test_model, test_dataloader, criterion)
            adapter.info(f"  📊 Test Set Results:")
            adapter.info(f"    Test Loss: {test_loss:.4f}")
            adapter.info(f"    Test Macro F1: {test_macro_f1:.4f}")
            adapter.info(f"    Test Biased F1: {test_biased_f1:.4f}")
            adapter.info(f"    Test Classification Report:\n{json.dumps(test_report, indent=4)}")

            results.update({
                'test_loss': test_loss,
                'test_macro_f1': test_macro_f1,
                'test_biased_f1': test_biased_f1,
                'test_report': test_report
            })
        except Exception as e:
             adapter.error(f"❌ Error during test evaluation: {e}")
             results['error'] = f"Test evaluation failed: {e}"
    else:
        adapter.error("❌ Could not find best model checkpoint to run test evaluation.")
        results['error'] = 'Best model checkpoint not found.'

    adapter.info(f"--- Fold {fold_index} Finished | Total Time: {format_time(time.time() - fold_start_time)} ---")
    # Clean up fold-specific handlers
    for handler in fold_logger.handlers[:]:
        fold_logger.removeHandler(handler)
        handler.close()

    return results


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-View GNN for Media Bias Detection")
    parser.add_argument('--start_fold', type=int, default=0, help='Fold index to start training from (0-9)')
    parser.add_argument('--end_fold', type=int, default=9, help='Fold index to end training at (0-9)')
    args = parser.parse_args()

    # --- Setup Root Logger (Console Only) ---
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    # Clear existing handlers if any, then add stream handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[logging.StreamHandler()])

    logging.info("🚀 Starting Dual-View GNN Training Script 🚀")
    logging.info(f"💾 Base Output Directory: {BASE_OUTPUT_DIR}")
    logging.info(f"💾 Data Directory: {DATA_DIR}")
    logging.info(f"Hyperparameters: Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, MaxLen={MAX_LEN}, LR_LM={LONGFORMER_LR}, LR_Other={NON_LONGFORMER_LR}")


    # --- Define Folds (Using BASIL structure from reference) ---
    logging.info("📁 Defining file splits for 10-fold cross-validation...")
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
        # Calculate indices for NYT, HPO, FOX based on fold_i, wrapping around
        nyt_index = folder_i % 10
        hpo_index = (folder_i + 1) % 10
        fox_index = (folder_i + 2) % 10

        # Collect file paths for each source within the current fold split
        for triple_idx in triples[nyt_index]:
            fname = os.path.join(DATA_DIR, f"basil_{triple_idx}_nyt_event_graph_classified.json")
            if os.path.exists(fname): this_folder.append(fname)
            else: logging.warning(f"File not found: {fname}")
        for triple_idx in triples[hpo_index]:
            fname = os.path.join(DATA_DIR, f"basil_{triple_idx}_hpo_event_graph_classified.json")
            if os.path.exists(fname): this_folder.append(fname)
            else: logging.warning(f"File not found: {fname}")
        for triple_idx in triples[fox_index]:
            fname = os.path.join(DATA_DIR, f"basil_{triple_idx}_fox_event_graph_classified.json")
            if os.path.exists(fname): this_folder.append(fname)
            else: logging.warning(f"File not found: {fname}")

        all_folds_paths.append(this_folder)
        logging.info(f"Fold {folder_i} definition includes {len(this_folder)} files.")

    # --- Run Cross-Validation ---
    all_results = []
    start_fold = max(0, args.start_fold)
    end_fold = min(9, args.end_fold)
    logging.info(f"🚦 Starting 10-fold cross-validation from fold {start_fold} to {end_fold}...")
    overall_start_time = time.time()

    for i in range(start_fold, end_fold + 1):
        test_fold_index = i
        dev_fold_index = (i - 1 + 10) % 10 # Dev fold is the one before test fold (wrapping around)

        test_paths = all_folds_paths[test_fold_index]
        dev_paths = all_folds_paths[dev_fold_index]
        train_paths = []
        for j in range(10):
            if j != test_fold_index and j != dev_fold_index:
                train_paths.extend(all_folds_paths[j])

        # Basic check for overlap - should not happen with this logic
        if set(test_paths) & set(dev_paths) or set(test_paths) & set(train_paths) or set(dev_paths) & set(train_paths):
             logging.error(f"❌ FATAL ERROR: Overlap detected in data splits for fold {i}. Aborting.")
             exit()
        if not train_paths or not dev_paths or not test_paths:
             logging.error(f"❌ FATAL ERROR: Empty data split for fold {i}. Train:{len(train_paths)}, Dev:{len(dev_paths)}, Test:{len(test_paths)}. Check DATA_DIR and file naming.")
             # Optionally continue to next fold or exit
             all_results.append({'fold': i, 'error': 'Empty data split detected.'})
             continue # Try next fold

        fold_results = train_model(test_fold_index, train_paths, dev_paths, test_paths)
        all_results.append(fold_results)

    # --- Aggregate and Save Overall Results ---
    logging.info(f"\n{'='*20} Cross-Validation Summary {'='*20}")
    valid_results = [r for r in all_results if 'error' not in r]
    all_test_biased_f1 = [r.get('test_biased_f1', 0.0) for r in valid_results]
    all_test_macro_f1 = [r.get('test_macro_f1', 0.0) for r in valid_results]

    if valid_results:
         avg_biased_f1 = np.mean(all_test_biased_f1)
         std_biased_f1 = np.std(all_test_biased_f1)
         avg_macro_f1 = np.mean(all_test_macro_f1)
         std_macro_f1 = np.std(all_test_macro_f1)
         logging.info(f"📈 Average Test Biased F1 across {len(valid_results)} folds: {avg_biased_f1:.4f} +/- {std_biased_f1:.4f}")
         logging.info(f"📈 Average Test Macro F1 across {len(valid_results)} folds:  {avg_macro_f1:.4f} +/- {std_macro_f1:.4f}")

         # Add averages to the results summary
         summary_results = {
            'avg_biased_f1': avg_biased_f1,
            'std_biased_f1': std_biased_f1,
            'avg_macro_f1': avg_macro_f1,
            'std_macro_f1': std_macro_f1,
            'fold_results': all_results # Include results with errors for completeness
         }
         summary_file = os.path.join(BASE_OUTPUT_DIR, "cross_validation_summary.json")
         try:
             with open(summary_file, 'w', encoding='utf-8') as f:
                 json.dump(summary_results, f, indent=4)
             logging.info(f"✅ Overall results summary saved to {summary_file}")
         except Exception as e:
             logging.error(f"❌ Failed to save summary results: {e}")
    else:
         logging.error("❌ No valid results collected across any folds.")

    total_time = format_time(time.time() - overall_start_time)
    logging.info(f"🏁 Total Training Script Finished. Total Time: {total_time} 🏁")