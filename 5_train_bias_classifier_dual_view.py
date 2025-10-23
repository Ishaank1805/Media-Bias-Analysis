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
import subprocess # From reference

# --- Environment & Setup ---

# Check for GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'✅ There are {torch.cuda.device_count()} GPU(s) available.')
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
# Paths
BASE_OUTPUT_DIR = "/scratch/Media-Bias-Analysis/results/"
DATA_DIR = "./BASIL_event_graph_classified/" # Data folder in current directory
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True) # Ensure base results dir exists

# Model Hyperparameters
MAX_LEN = 2048 # Adjusted from reference's 2550 to a more standard Longformer length
HIDDEN_DIM = 768
GAT_HEADS = 4
DROPOUT = 0.1
ALPHA_LEAKY = 0.2

# Training Hyperparameters from reference code
NUM_EPOCHS = 5
BATCH_SIZE = 1
CHECK_TIMES_PER_EPOCH = 4 # 4 times per epoch
CHECK_TIMES = CHECK_TIMES_PER_EPOCH * NUM_EPOCHS
NO_DECAY = ['bias', 'LayerNorm.weight']
LONGFORMER_WEIGHT_DECAY = 1e-2
NON_LONGFORMER_WEIGHT_DECAY = 1e-2
WARMUP_PROPORTION = 0.0 # From reference
LONGFORMER_LR = 1e-5
NON_LONGFORMER_LR = 2e-5

# --- Helper Functions ---
def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# --- Dataset Definition ---
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

class DualViewDataset(Dataset):
    """
    Custom PyTorch Dataset for loading classified event graph data.
    Implements logic from report:
    - Loads classified JSONs.
    - Tokenizes text with Longformer.
    - Separates events into Factual/Interpretive based on 'fi_classification'.
    - Constructs graph edge indices for Factual, Interpretive, and Cross-View.
    - Handles simplified paragraph (doc=para) and document-level graph structure.
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths
        logging.info(f"🔍 Initialized dataset with {len(file_paths)} files.")

    def __len__(self):
        return len(self.file_paths)

    def _get_paragraph_grouping(self, sentences):
        """
        Groups sentences into paragraphs.
        Per report (Sec 4.4), sentence-level graphs are redundant.
        Per discussion, BASIL lacks paragraph markers.
        Strategy: Treat the whole document as one paragraph (Paragraph 0).
        """
        para_groups = {}
        para_groups[0] = list(range(len(sentences)))
        return para_groups

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                article_json = json.load(f)
        except Exception as e:
            logging.error(f"❌ Error reading or parsing {file_path}: {e}")
            return None # Skip this file in collate_fn

        # --- 1. Tokenization and Basic Info ---
        input_ids = []
        attention_mask = []
        token_to_sentence_map = [] # Map token index in input_ids -> sentence index
        sentence_labels = {} # Map sentence index -> bias label (0 or 1)
        sentence_start_end_tokens = {} # Map sentence index -> (start_token_idx, end_token_idx) in input_ids

        # Add Longformer start token '<s>'
        input_ids.append(tokenizer.bos_token_id) # BOS token id
        attention_mask.append(1)
        token_to_sentence_map.append(-1) # Special index for start token

        current_token_idx = len(input_ids) # Start counting after '<s>'
        processed_sentences_indices = [] # Keep track of indices of sentences included

        for sent_i, sentence in enumerate(article_json['sentences']):
            sentence_text = sentence.get('sentence_text', '')
            # Use label_info_bias for BASIL, default to 0 if missing or -1 (like title)
            label = sentence.get('label_info_bias', 0)
            if label == -1: label = 0 # Treat unlabelled as non-biased

            # Add space prefix
            encoded = tokenizer.encode_plus(' ' + sentence_text, add_special_tokens=False, max_length=MAX_LEN - current_token_idx - 1, truncation=True)
            sent_ids = encoded['input_ids']
            sent_mask = encoded['attention_mask']

            if not sent_ids: continue # Skip empty sentences

            sent_start_token_idx = current_token_idx
            input_ids.extend(sent_ids)
            attention_mask.extend(sent_mask)
            token_to_sentence_map.extend([sent_i] * len(sent_ids))
            current_token_idx += len(sent_ids)
            sentence_start_end_tokens[sent_i] = (sent_start_token_idx, current_token_idx)
            sentence_labels[sent_i] = label
            processed_sentences_indices.append(sent_i)

            if current_token_idx >= MAX_LEN - 1: # Check if we filled up
                break # Stop adding sentences

        # Add Longformer end token '</s>'
        input_ids.append(tokenizer.eos_token_id)
        attention_mask.append(1)
        token_to_sentence_map.append(-1)

        # Padding
        num_pad = MAX_LEN - len(input_ids)
        if num_pad > 0:
            input_ids.extend([tokenizer.pad_token_id] * num_pad)
            attention_mask.extend([0] * num_pad)
            token_to_sentence_map.extend([-1] * num_pad)
        elif num_pad < 0: # Truncate if overflown
             input_ids = input_ids[:MAX_LEN]
             attention_mask = attention_mask[:MAX_LEN]
             token_to_sentence_map = token_to_sentence_map[:MAX_LEN]
             input_ids[-1] = tokenizer.eos_token_id # Ensure EOS is last token
             attention_mask[-1] = 1


        # --- 2. Event and Relation Processing ---
        factual_events = []
        interpretive_events = []
        event_map = {} # original_idx -> info

        # Map original token index from JSON to its start/end position in input_ids
        original_idx_to_input_idx = {}
        current_input_idx = 1 # Start after '<s>'
        for sent_i, sentence in enumerate(article_json['sentences']):
             if sent_i not in sentence_start_end_tokens: continue
             sent_start, sent_end = sentence_start_end_tokens[sent_i]
             token_start_in_sent = sent_start
             for token_info in sentence['tokens']:
                 original_token_idx = token_info['index_of_token']
                 token_text = token_info['token_text']
                 encoded = tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)
                 token_len = len(encoded['input_ids'])

                 if token_start_in_sent + token_len <= sent_end:
                     original_idx_to_input_idx[original_token_idx] = (token_start_in_sent, token_start_in_sent + token_len)
                     token_start_in_sent += token_len
                 else:
                     if token_start_in_sent < sent_end:
                         original_idx_to_input_idx[original_token_idx] = (token_start_in_sent, sent_end)
                         token_start_in_sent = sent_end
             if token_start_in_sent > current_input_idx:
                  current_input_idx = token_start_in_sent
             if current_input_idx >= MAX_LEN -1 : break


        for event_token in article_json.get('event_tokens', []):
            original_token_idx = event_token['index_of_token']
            if original_token_idx not in original_idx_to_input_idx:
                continue

            start_idx, end_idx = original_idx_to_input_idx[original_token_idx]
            if start_idx >= len(token_to_sentence_map) or start_idx < 0: continue
            sent_idx = token_to_sentence_map[start_idx]
            if sent_idx == -1: continue

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


        # --- 3. Graph Construction ---
        num_factual = len(factual_events)
        num_interpretive = len(interpretive_events)
        factual_edges = []
        interpretive_edges = []
        cross_view_edges = []
        # Relation types: coref, temporal, causal, subevent
        # Cross-view edges: "interprets" (F->I), "supported-by" (I->F)
        relation_map = {
            'coreference': 0, 'temporal': {1: 1, 2: 2, 3: 3}, # 0, 1, 2, 3
            'causal': {1: 4, 2: 5}, # 4, 5
            'subevent': {1: 6, 2: 7} # 6, 7
        }
        # Cross-view edge types
        REL_INTERPRETS = 8
        REL_SUPPORTED_BY = 9


        for rel in article_json.get('relation_label', []):
            ev1_orig_idx = rel['event_1']['index_of_token']
            ev2_orig_idx = rel['event_2']['index_of_token']
            if ev1_orig_idx not in event_map or ev2_orig_idx not in event_map: continue

            ev1_info = event_map[ev1_orig_idx]
            ev2_info = event_map[ev2_orig_idx]
            ev1_type, ev1_local_idx = ev1_info['type'], ev1_info['local_idx']
            ev2_type, ev2_local_idx = ev2_info['type'], ev2_info['local_idx']

            rels_to_add = []
            if rel.get('label_coreference') == 1: rels_to_add.append(relation_map['coreference'])
            if rel.get('label_temporal', 0) in relation_map['temporal']: rels_to_add.append(relation_map['temporal'][rel['label_temporal']])
            if rel.get('label_causal', 0) in relation_map['causal']: rels_to_add.append(relation_map['causal'][rel['label_causaausalausall']])
            if rel.get('label_subevent', 0) in relation_map['subevent']: rels_to_add.append(relation_map['subevent'][rel['label_subevent']])

            # Factual Subgraph, Interpretive Subgraph
            if ev1_type == 'F' and ev2_type == 'F':
                for rel_idx in rels_to_add:
                    factual_edges.append((ev1_local_idx, ev2_local_idx, rel_idx))
                    if rel_idx in [0, 3]: factual_edges.append((ev2_local_idx, ev1_local_idx, rel_idx)) # Symmetric
            elif ev1_type == 'I' and ev2_type == 'I':
                for rel_idx in rels_to_add:
                    interpretive_edges.append((ev1_local_idx, ev2_local_idx, rel_idx))
                    if rel_idx in [0, 3]: interpretive_edges.append((ev2_local_idx, ev1_local_idx, rel_idx))
            else: # Cross-View Edges
                if rels_to_add: # Only add cross-edge if a relation exists
                     if ev1_type == 'F': cross_view_edges.append((ev1_local_idx, ev2_local_idx, REL_INTERPRETS)) # F -> I
                     else: cross_view_edges.append((ev1_local_idx, ev2_local_idx, REL_SUPPORTED_BY)) # I -> F

        # Add cross-view edges based on sentence co-occurrence
        sent_events_map = {}
        for orig_idx, event_info in event_map.items():
            s_idx = event_info['sent_idx']
            e_type = event_info['type']
            loc_idx = event_info['local_idx']
            if s_idx not in sent_events_map: sent_events_map[s_idx] = {'F': [], 'I': []}
            sent_events_map[s_idx][e_type].append(loc_idx)
        for s_idx, events in sent_events_map.items():
            for f_idx in events['F']:
                for i_idx in events['I']:
                    cross_view_edges.append((f_idx, i_idx, REL_INTERPRETS)) # F -> I ("interprets")
                    cross_view_edges.append((i_idx, f_idx, REL_SUPPORTED_BY)) # I -> F ("supported-by")

        factual_edges = list(set(factual_edges))
        interpretive_edges = list(set(interpretive_edges))
        cross_view_edges = list(set(cross_view_edges))

        def edges_to_tensor(edges):
            if not edges: return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
            edge_array = np.array(edges)
            edge_index = torch.tensor(edge_array[:, :2].T, dtype=torch.long)
            edge_type = torch.tensor(edge_array[:, 2], dtype=torch.long)
            return edge_index, edge_type

        factual_edge_index, factual_edge_type = edges_to_tensor(factual_edges)
        interpretive_edge_index, interpretive_edge_type = edges_to_tensor(interpretive_edges)
        cross_f_to_i_edges = [(s, t, type) for s, t, type in cross_view_edges if type == REL_INTERPRETS]
        cross_i_to_f_edges = [(s, t, type) for s, t, type in cross_view_edges if type == REL_SUPPORTED_BY]
        cross_f_to_i_index, _ = edges_to_tensor(cross_f_to_i_edges)
        cross_i_to_f_index, _ = edges_to_tensor(cross_i_to_f_edges)

        # --- 4. Paragraph and Document Structure ---
        paragraph_groups = self._get_paragraph_grouping(article_json['sentences']) #
        num_paragraphs = len(paragraph_groups)
        para_map_list = [-1] * len(article_json['sentences'])
        for para_idx, sent_indices in paragraph_groups.items():
             for sent_idx in sent_indices:
                 para_map_list[sent_idx] = para_idx
        paragraph_mapping = torch.tensor(para_map_list, dtype=torch.long)

        # Document graph edges (Simplified: no edges for single paragraph)
        doc_edge_index = torch.empty((2,0), dtype=torch.long)

        # --- 5. Final Output Dict ---
        valid_sent_indices = sorted([idx for idx in processed_sentences_indices if sentence_labels[idx] != -1])
        final_labels = torch.tensor([sentence_labels[i] for i in valid_sent_indices], dtype=torch.long)
        final_label_sent_indices = torch.tensor(valid_sent_indices, dtype=torch.long)

        if final_labels.numel() == 0:
             logging.warning(f"No valid sentence labels found for {file_path}. Skipping.")
             return None

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sentence_labels": final_labels,
            "final_label_sent_indices": final_label_sent_indices,
            "num_factual": num_factual,
            "num_interpretive": num_interpretive,
            "factual_events": factual_events,
            "interpretive_events": interpretive_events,
            "factual_edge_index": factual_edge_index,
            "factual_edge_type": factual_edge_type,
            "interpretive_edge_index": interpretive_edge_index,
            "interpretive_edge_type": interpretive_edge_type,
            "cross_f_to_i_index": cross_f_to_i_index,
            "cross_i_to_f_index": cross_i_to_f_index,
            "paragraph_mapping": paragraph_mapping,
            "num_paragraphs": num_paragraphs,
            "doc_edge_index": doc_edge_index,
            "token_to_sentence_map": token_to_sentence_map
        }

# Collate function to handle None values
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    # Assuming BATCH_SIZE=1
    return batch[0]


# --- Model Architecture ---

class BaseEncoder(nn.Module):
    """Encodes input text using Longformer followed by a BiLSTM."""
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)
        self.bilstm_token = nn.LSTM(input_size=self.longformer.config.hidden_size,
                                    hidden_size=self.longformer.config.hidden_size // 2,
                                    batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        logging.info("✅ BaseEncoder initialized.")

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if not hidden_states or len(hidden_states) < 4:
            token_embeddings = outputs.last_hidden_state
        else:
             token_embeddings_layers = torch.stack(hidden_states[-4:], dim=0) #
             token_embeddings = torch.sum(token_embeddings_layers, dim=0)

        lstm_out, _ = self.bilstm_token(token_embeddings) #
        return self.dropout(lstm_out)


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
        self.a = nn.Linear(2 * out_dim, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout_p)
        self._init_weights()
        logging.info(f"✅ RGATLayer initialized (in={in_dim}, out={out_dim}, relations={num_relations}).")

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

            source_feats_r = node_features[source_nodes]
            target_feats_r = node_features[target_nodes]

            Q_r = self.W_Q[r](target_feats_r)
            K_r = self.W_K[r](source_feats_r)
            V_r = self.W_V[r](source_feats_r)

            a_input = torch.cat([Q_r, K_r], dim=1)
            e = self.leakyrelu(self.a(a_input)).squeeze(1)

            # --- Fixed scatter_reduce call ---
            # Use torch_scatter ops. The signature is (src, index, dim, dim_size, reduce)
            # 1. Get max for stability
            e_max = torch.ops.torch_scatter.scatter_reduce(e, target_nodes, "amax", num_nodes, False)[0]
            # 2. Subtract max
            e_stable = e - e_max.gather(0, target_nodes)
            e_exp = torch.exp(e_stable)
            # 3. Sum exps
            e_sum = torch.ops.torch_scatter.scatter_reduce(e_exp, target_nodes, "add", num_nodes, False)[0]
            # 4. Normalize
            attention = e_exp / (e_sum.gather(0, target_nodes) + 1e-10)
            # --- End Fix ---

            attention = self.dropout(attention)
            weighted_values = V_r * attention.unsqueeze(1)
            
            # Use scatter_add
            h_prime = torch.ops.torch_scatter.scatter_reduce(weighted_values, target_nodes.unsqueeze(1).expand_as(weighted_values), "add", num_nodes, False, out=h_prime)[0]

        return F.elu(h_prime)


class CrossViewAttention(nn.Module):
    """ Simplified attention for cross-view interaction """
    def __init__(self, embed_dim, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attend = nn.Linear(embed_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(ALPHA_LEAKY)
        logging.info(f"✅ CrossViewAttention initialized (dim={embed_dim}).")

    def forward(self, query_nodes, key_value_nodes, edge_index):
        num_query_nodes = query_nodes.size(0)
        if edge_index.numel() == 0 or key_value_nodes.numel() == 0:
            return torch.zeros_like(query_nodes)

        query_idx, kv_idx = edge_index[0], edge_index[1]

        if query_idx.max() >= num_query_nodes or kv_idx.max() >= key_value_nodes.size(0):
             logging.error(f"❌ CrossViewAttention index out of bounds! query_max={query_idx.max()}/{num_query_nodes}, kv_max={kv_idx.max()}/{key_value_nodes.size(0)}")
             return torch.zeros_like(query_nodes)

        query_proj = self.proj(query_nodes[query_idx])
        kv_proj = self.proj(key_value_nodes[kv_idx])

        a_input = torch.cat([query_proj, kv_proj], dim=1)
        e = self.leakyrelu(self.attend(a_input)).squeeze(1)

        # --- Fixed scatter_reduce call ---
        e_max = torch.ops.torch_scatter.scatter_reduce(e, query_idx, "amax", num_query_nodes, False)[0]
        e_stable = e - e_max.gather(0, query_idx)
        e_exp = torch.exp(e_stable)
        e_sum = torch.ops.torch_scatter.scatter_reduce(e_exp, query_idx, "add", num_query_nodes, False)[0]
        attention = e_exp / (e_sum.gather(0, query_idx) + 1e-10)
        # --- End Fix ---

        attention = self.dropout(attention)
        weighted_values = kv_proj * attention.unsqueeze(1)

        aggregated_context = torch.zeros((num_query_nodes, query_nodes.size(1)), device=query_nodes.device)
        aggregated_context = torch.ops.torch_scatter.scatter_reduce(weighted_values, query_idx.unsqueeze(1).expand_as(weighted_values), "add", num_query_nodes, False, out=aggregated_context)[0]

        return aggregated_context


class GATLayer(nn.Module):
    """ Standard GAT Layer for Document-Level Graph """
    def __init__(self, in_dim, out_dim, dropout=DROPOUT, alpha=ALPHA_LEAKY):
        super().__init__()
        self.dropout_p = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout_p)
        self._init_weights()
        logging.info(f"✅ GATLayer initialized (in={in_dim}, out={out_dim}).")

    def _init_weights(self):
         nn.init.xavier_uniform_(self.W.weight, gain=1.414)
         nn.init.xavier_uniform_(self.a.weight, gain=1.414)

    def forward(self, node_features, edge_index):
        if node_features.numel() == 0: return node_features
        h = self.W(node_features)
        num_nodes = h.size(0)
        if edge_index.numel() == 0: return F.elu(h)

        source_nodes, target_nodes = edge_index[0], edge_index[1]
        if source_nodes.max() >= num_nodes or target_nodes.max() >= num_nodes:
             logging.error(f"❌ GATLayer index out of bounds! Max source={source_nodes.max()}, Max target={target_nodes.max()}, Num nodes={num_nodes}")
             return F.elu(h)

        h_target = h[target_nodes]
        h_source = h[source_nodes]
        a_input = torch.cat([h_source, h_target], dim=1)
        e = self.leakyrelu(self.a(a_input)).squeeze(1)

        # --- Fixed scatter_reduce call ---
        e_max = torch.ops.torch_scatter.scatter_reduce(e, target_nodes, "amax", num_nodes, False)[0]
        e_stable = e - e_max.gather(0, target_nodes)
        e_exp = torch.exp(e_stable)
        e_sum = torch.ops.torch_scatter.scatter_reduce(e_exp, target_nodes, "add", num_nodes, False)[0]
        attention = e_exp / (e_sum.gather(0, target_nodes) + 1e-10)
        # --- End Fix ---
        
        attention = self.dropout(attention)
        weighted_source_features = h_source * attention.unsqueeze(1)

        h_prime = torch.zeros_like(h)
        h_prime = torch.ops.torch_scatter.scatter_reduce(weighted_source_features, target_nodes.unsqueeze(1).expand_as(weighted_source_features), "add", num_nodes, False, out=h_prime)[0]

        return F.elu(h_prime)


class DualViewGNN(nn.Module):
    """ Main model implementing the Two-Level Dual-View architecture """
    def __init__(self, num_relations=8, dropout=DROPOUT): # 8 within-view relation types
        super().__init__()
        self.encoder = BaseEncoder(dropout=dropout) #

        # Paragraph Level R-GATs for Factual/Interpretive
        self.rgat_factual = RGATLayer(HIDDEN_DIM, HIDDEN_DIM, num_relations, dropout=dropout)
        self.rgat_interpretive = RGATLayer(HIDDEN_DIM, HIDDEN_DIM, num_relations, dropout=dropout)

        # Cross-View Attention
        self.cross_attn_f = CrossViewAttention(HIDDEN_DIM, dropout=dropout)
        self.cross_attn_i = CrossViewAttention(HIDDEN_DIM, dropout=dropout)

        self.norm_f1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_i1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_f2 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_i2 = nn.LayerNorm(HIDDEN_DIM)

        # Paragraph Aggregation
        self.para_aggregate_proj = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)

        # Document Level GAT
        self.doc_gat = GATLayer(HIDDEN_DIM, HIDDEN_DIM, dropout=dropout)
        self.norm_doc = nn.LayerNorm(HIDDEN_DIM)

        # Final Classifier MLP
        classifier_input_dim = HIDDEN_DIM * 3
        self.bias_classifier_1 = nn.Linear(classifier_input_dim, HIDDEN_DIM)
        self.bias_classifier_2 = nn.Linear(HIDDEN_DIM, 2) # 2 classes: non-bias, bias
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        logging.info("✅ DualViewGNN model initialized.")

    def _get_event_embeddings(self, token_embeddings, events):
        """ Averages token embeddings for each event mention. """
        event_embeddings = []
        if not events:
            return torch.empty((0, HIDDEN_DIM), device=token_embeddings.device)

        for event in events:
            indices = torch.tensor(event['token_indices'], device=token_embeddings.device, dtype=torch.long)
            if indices.numel() > 0 and indices.max() < token_embeddings.size(0):
                 event_embeddings.append(token_embeddings[indices].mean(dim=0))
            else:
                 event_embeddings.append(torch.zeros(HIDDEN_DIM, device=token_embeddings.device))

        if not event_embeddings:
            return torch.empty((0, HIDDEN_DIM), device=token_embeddings.device)
        return torch.stack(event_embeddings)


    def forward(self, batch):
        # Handle DataParallel by checking input type
        if isinstance(batch, list): # DataParallel wraps batch in a list
            batch = batch[0]

        input_ids = batch['input_ids'].unsqueeze(0).to(device)
        attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
        num_factual = batch['num_factual']
        num_interpretive = batch['num_interpretive']
        factual_events = batch['factual_events']
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
        final_label_sent_indices = batch['final_label_sent_indices'].to(device)
        token_to_sentence_map = batch['token_to_sentence_map']

        # 1. Base Encoding
        token_embeddings = self.encoder(input_ids, attention_mask).squeeze(0) # (seq_len, hidden_dim)

        # 2. Extract Initial Event Embeddings
        factual_event_embeds = self._get_event_embeddings(token_embeddings, factual_events)
        interpretive_event_embeds = self._get_event_embeddings(token_embeddings, interpretive_events)

        # 3. Within-View R-GAT + Residual + Norm
        updated_factual_embeds = factual_event_embeds
        if num_factual > 0 and factual_edge_index.numel() > 0:
            factual_updates = self.rgat_factual(factual_event_embeds, factual_edge_index, factual_edge_type)
            if factual_updates.shape == factual_event_embeds.shape:
                updated_factual_embeds = self.norm_f1(factual_event_embeds + factual_updates)

        updated_interpretive_embeds = interpretive_event_embeds
        if num_interpretive > 0 and interpretive_edge_index.numel() > 0:
            interpretive_updates = self.rgat_interpretive(interpretive_event_embeds, interpretive_edge_index, interpretive_edge_type)
            if interpretive_updates.shape == interpretive_event_embeds.shape:
                updated_interpretive_embeds = self.norm_i1(interpretive_event_embeds + interpretive_updates)

        # 4. Cross-View Attention + Residual + Norm
        factual_context = torch.zeros_like(updated_factual_embeds)
        interpretive_context = torch.zeros_like(updated_interpretive_embeds)
        if num_factual > 0 and num_interpretive > 0:
             factual_context = self.cross_attn_f(updated_factual_embeds, updated_interpretive_embeds, cross_i_to_f_index)
             interpretive_context = self.cross_attn_i(updated_interpretive_embeds, updated_factual_embeds, cross_f_to_i_index)

        if num_factual > 0:
             if factual_context.shape == updated_factual_embeds.shape:
                 updated_factual_embeds = self.norm_f2(updated_factual_embeds + factual_context)
        if num_interpretive > 0:
             if interpretive_context.shape == updated_interpretive_embeds.shape:
                 updated_interpretive_embeds = self.norm_i2(updated_interpretive_embeds + interpretive_context)

        # 5. Aggregate Paragraph Representations
        # Simplified: mean pooling for the single 'paragraph' (doc)
        para_factual_summary = updated_factual_embeds.mean(dim=0, keepdim=True) if num_factual > 0 else torch.zeros((1, HIDDEN_DIM), device=device)
        para_interpretive_summary = updated_interpretive_embeds.mean(dim=0, keepdim=True) if num_interpretive > 0 else torch.zeros((1, HIDDEN_DIM), device=device)
        para_combined = torch.cat([para_factual_summary, para_interpretive_summary], dim=1)
        paragraph_vectors = self.relu(self.para_aggregate_proj(para_combined)) # (num_paras=1, dim)

        # 6. Document Level GAT
        # With num_paragraphs=1, GAT just applies normalization
        updated_paragraph_vectors = self.norm_doc(paragraph_vectors)

        # 7. Final Sentence Classification
        sentence_final_embeddings = []
        for label_idx, original_sent_idx in enumerate(final_label_sent_indices):
            original_sent_idx = original_sent_idx.item()

            # a) Longformer Sentence Embedding
            sent_token_indices = [idx for idx, s_map_idx in enumerate(token_to_sentence_map) if s_map_idx == original_sent_idx]
            if sent_token_indices:
                longformer_sent_embed = token_embeddings[torch.tensor(sent_token_indices, device=device)].mean(dim=0)
            else:
                longformer_sent_embed = torch.zeros(HIDDEN_DIM, device=device)

            # b) Updated Paragraph Context
            para_idx = 0 # All sentences belong to paragraph 0
            paragraph_context = updated_paragraph_vectors[para_idx]

            # c) Aggregated Event Embeddings
            sent_factual_indices = [fe['local_idx'] for fe in factual_events if fe['sent_idx'] == original_sent_idx]
            sent_interpretive_indices = [ie['local_idx'] for ie in interpretive_events if ie['sent_idx'] == original_sent_idx]
            aggregated_event_embed = torch.zeros(HIDDEN_DIM, device=device)
            count = 0
            if sent_factual_indices and num_factual > 0:
                 valid_f_indices = [idx for idx in sent_factual_indices if idx < updated_factual_embeds.size(0)]
                 if valid_f_indices:
                      aggregated_event_embed += updated_factual_embeds[torch.tensor(valid_f_indices, device=device, dtype=torch.long)].sum(dim=0)
                      count += len(valid_f_indices)
            if sent_interpretive_indices and num_interpretive > 0:
                 valid_i_indices = [idx for idx in sent_interpretive_indices if idx < updated_interpretive_embeds.size(0)]
                 if valid_i_indices:
                      aggregated_event_embed += updated_interpretive_embeds[torch.tensor(valid_i_indices, device=device, dtype=torch.long)].sum(dim=0)
                      count += len(valid_i_indices)
            if count > 0: aggregated_event_embed /= count

            # Concatenate three representations
            combined_features = torch.cat([longformer_sent_embed, paragraph_context, aggregated_event_embed], dim=0)
            sentence_final_embeddings.append(combined_features)

        if not sentence_final_embeddings:
            return torch.empty((0, 2), device=device)

        final_input = torch.stack(sentence_final_embeddings)
        hidden = self.dropout(self.relu(self.bias_classifier_1(final_input)))
        logits = self.bias_classifier_2(hidden) #
        return logits


# --- Evaluation Function ---
def evaluate_model(model, dataloader, criterion, adapter):
    """ Evaluates the model on a given dataset loader. """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if batch is None: continue
            labels = batch['sentence_labels'].to(device)
            if labels.numel() == 0: continue

            # Handle DataParallel wrapper
            if isinstance(model, nn.DataParallel):
                logits = model(batch)
            else:
                logits = model(batch)
                
            if logits.numel() == 0 or logits.shape[0] != labels.shape[0]:
                 adapter.warning(f"Eval shape mismatch ({logits.shape} vs {labels.shape}) or empty logits. Skipping.")
                 continue

            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            num_samples += labels.size(0)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    if num_samples == 0:
        adapter.warning("⚠️ No valid samples found during evaluation.")
        return 0.0, 0.0, 0.0, {}

    avg_loss = total_loss / num_samples
    try:
        report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        f1_macro = report_dict.get('macro avg', {}).get('f1-score', 0.0)
        f1_biased = report_dict.get('1', {}).get('f1-score', 0.0) # '1' is the key for the biased class
    except Exception as e:
        adapter.error(f"Error during classification report: {e}")
        f1_macro = 0.0
        f1_biased = 0.0
        report_dict = {}

    return avg_loss, f1_macro, f1_biased, report_dict


# --- Training Loop Function ---
def train_model(fold_index, train_paths, dev_paths, test_paths):
    """ Trains and evaluates the model for a single cross-validation fold. """
    fold_start_time = time.time()
    fold_output_dir = os.path.join(BASE_OUTPUT_DIR, f"fold_{fold_index}")
    os.makedirs(fold_output_dir, exist_ok=True)
    log_file = os.path.join(fold_output_dir, f"train_fold_{fold_index}.log")

    # === Setup Fold-Specific Logging ===
    fold_logger = logging.getLogger(f"Fold_{fold_index}")
    fold_logger.setLevel(logging.INFO)
    fold_logger.propagate = False # Prevent duplicate logs to root
    for handler in fold_logger.handlers[:]: fold_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    fold_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - Fold %(fold)s - %(message)s'))
    fold_logger.addHandler(file_handler)
    
    adapter = logging.LoggerAdapter(fold_logger, {'fold': fold_index})
    adapter.info(f"\n{'='*25} Starting Fold {fold_index} {'='*25}")
    adapter.info(f"📁 Output Dir: {fold_output_dir}")
    adapter.info(f"🚂 Train files: {len(train_paths)}, 🧑‍💻 Dev files: {len(dev_paths)}, 🧪 Test files: {len(test_paths)}")

    # Datasets and Dataloaders
    train_dataset = DualViewDataset(train_paths)
    dev_dataset = DualViewDataset(dev_paths)
    test_dataset = DualViewDataset(test_paths)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model, Optimizer, Scheduler
    model = DualViewGNN()
    if torch.cuda.device_count() > 1:
        adapter.info(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    model.to(device)

    param_all = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in NO_DECAY)) and ('encoder.longformer' in n))], 'lr': LONGFORMER_LR, 'weight_decay': LONGFORMER_WEIGHT_DECAY},
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in NO_DECAY)) and (not 'encoder.longformer' in n))], 'lr': NON_LONGFORMER_LR, 'weight_decay': NON_LONGFORMER_WEIGHT_DECAY},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in NO_DECAY)) and ('encoder.longformer' in n))], 'lr': LONGFORMER_LR, 'weight_decay': 0.0},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in NO_DECAY)) and (not 'encoder.longformer' in n))], 'lr': NON_LONGFORMER_LR, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)

    num_train_batches = len(train_dataloader)
    if num_train_batches == 0:
         adapter.error("❌ Train dataloader is empty. Check data paths.")
         for handler in fold_logger.handlers[:]: fold_logger.removeHandler(handler); handler.close()
         return {'fold': fold_index, 'error': 'Empty train dataloader.'}

    num_train_steps = NUM_EPOCHS * num_train_batches
    warmup_steps = int(WARMUP_PROPORTION * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    criterion = nn.CrossEntropyLoss()
    best_dev_biased_f1 = -1.0
    best_epoch = -1
    global_step = 0
    steps_per_eval = max(1, num_train_batches // CHECK_TIMES_PER_EPOCH)

    # --- Epoch Loop ---
    for epoch_i in range(NUM_EPOCHS):
        adapter.info(f"\n--- Epoch {epoch_i+1}/{NUM_EPOCHS} ---")
        epoch_t0 = time.time()
        total_train_loss = 0
        num_train_samples = 0
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Training Fold {fold_index} Epoch {epoch_i+1}", leave=False)
        for step, batch in enumerate(batch_iterator):
            if batch is None: continue
            labels = batch['sentence_labels'].to(device)
            if labels.numel() == 0: continue
            global_step += 1
            optimizer.zero_grad()
            
            # Handle DataParallel wrapper
            if isinstance(model, nn.DataParallel):
                logits = model(batch)
            else:
                logits = model(batch)

            if logits.numel() == 0 or logits.shape[0] != labels.shape[0]:
                 adapter.error(f"Shape mismatch! Logits: {logits.shape}, Labels: {labels.shape}. Skipping batch {step}.")
                 continue

            loss = criterion(logits, labels)
            batch_loss = loss.item() * labels.size(0)
            total_train_loss += batch_loss
            num_train_samples += labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch_iterator.set_postfix({'loss': f'{loss.item():.4f}'})

            # --- Intermediate Evaluation ---
            if global_step % steps_per_eval == 0 and step > 0:
                avg_interval_train_loss = total_train_loss / num_train_samples if num_train_samples > 0 else 0
                adapter.info(f"  Step {global_step}/{num_train_steps} | Avg Train Loss: {avg_interval_train_loss:.4f} | Elapsed: {format_time(time.time() - epoch_t0)}")
                total_train_loss = 0; num_train_samples = 0

                adapter.info("  Running intermediate validation...")
                dev_loss, dev_macro_f1, dev_biased_f1, _ = evaluate_model(model, dev_dataloader, criterion, adapter)
                adapter.info(f"  Validation -> Loss: {dev_loss:.4f}, Macro F1: {dev_macro_f1:.4f}, Biased F1: {dev_biased_f1:.4f}")

                if dev_biased_f1 > best_dev_biased_f1:
                    adapter.info(f"  🎉 New best Biased F1: {dev_biased_f1:.4f} (Prev best: {best_dev_biased_f1:.4f})")
                    best_dev_biased_f1 = dev_biased_f1
                    best_epoch = epoch_i + 1
                    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                    torch.save(model_to_save.state_dict(), os.path.join(fold_output_dir, f"best_model_fold_{fold_index}.ckpt"))
                    adapter.info(f"  💾 Best model checkpoint saved.")
                model.train() # Set back to train mode

        # --- End of Epoch Evaluation ---
        adapter.info(f"\n--- End of Epoch {epoch_i+1} ---")
        adapter.info(f"Epoch Training Time: {format_time(time.time() - epoch_t0)}")
        adapter.info("Running end-of-epoch validation...")
        dev_loss, dev_macro_f1, dev_biased_f1, dev_report = evaluate_model(model, dev_dataloader, criterion, adapter)
        adapter.info(f"  Validation Loss: {dev_loss:.4f}")
        adapter.info(f"  Validation Macro F1: {dev_macro_f1:.4f}")
        adapter.info(f"  Validation Biased F1: {dev_biased_f1:.4f}")
        adapter.info(f"  Validation Report:\n{json.dumps(dev_report, indent=4)}")

        if dev_biased_f1 > best_dev_biased_f1:
            adapter.info(f"  🎉 New best Biased F1: {dev_biased_f1:.4f} (Prev best: {best_dev_biased_f1:.4f})")
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
            test_model_state_dict = torch.load(best_model_path, map_location='cpu')
            test_model = DualViewGNN()
            if next(iter(test_model_state_dict)).startswith('module.'):
                 test_model_state_dict = {k.partition('module.')[2]: v for k, v in test_model_state_dict.items()}
            test_model.load_state_dict(test_model_state_dict)

            if torch.cuda.device_count() > 1:
                test_model = nn.DataParallel(test_model)
            test_model.to(device)
            adapter.info(f"✅ Successfully loaded best model from {best_model_path}")

            test_loss, test_macro_f1, test_biased_f1, test_report = evaluate_model(test_model, test_dataloader, criterion, adapter)
            adapter.info(f"  📊 Test Set Results:")
            adapter.info(f"    Test Loss: {test_loss:.4f}")
            adapter.info(f"    Test Macro F1: {test_macro_f1:.4f}")
            adapter.info(f"    Test Biased F1: {test_biased_f1:.4f}")
            adapter.info(f"    Test Classification Report:\n{json.dumps(test_report, indent=4)}")

            results.update({ 'test_loss': test_loss, 'test_macro_f1': test_macro_f1, 'test_biased_f1': test_biased_f1, 'test_report': test_report })
        except Exception as e:
             adapter.error(f"❌ Error during test evaluation: {e}")
             results['error'] = f"Test evaluation failed: {e}"
    else:
        adapter.error("❌ Could not find best model checkpoint to run test evaluation.")
        results['error'] = 'Best model checkpoint not found.'

    adapter.info(f"--- Fold {fold_index} Finished | Total Time: {format_time(time.time() - fold_start_time)} ---")
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
    # This ensures the main script info (like file finding) is printed to terminal
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[logging.StreamHandler()])

    logging.info("🚀 Starting Dual-View GNN Training Script 🚀")
    logging.info(f"💾 Base Output Directory: {BASE_OUTPUT_DIR}")
    logging.info(f"💾 Data Directory: {DATA_DIR}")
    logging.info(f"Hyperparameters: Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, MaxLen={MAX_LEN}, LR_LM={LONGFORMER_LR}, LR_Other={NON_LONGFORMER_LR}")

    # --- Define Folds (Using BASIL structure from reference) ---
    logging.info("📁 Defining file splits for 10-fold cross-validation...")
    # - Re-implementing the exact 10-fold split from reference
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
    total_files_found = 0
    # - Re-implementing the fold construction logic
    for folder_i in range(10):
        this_folder = []
        nyt_index = folder_i % 10
        hpo_index = (folder_i + 1) % 10
        fox_index = (folder_i + 2) % 10

        # Corrected file name logic to find the *classified* files
        for triple_idx in triples[nyt_index]:
            fname = os.path.join(DATA_DIR, f"basil_{triple_idx}_nyt_event_graph_classified.json")
            if os.path.exists(fname): this_folder.append(fname); total_files_found += 1
            else: logging.warning(f"File not found: {fname}")
        for triple_idx in triples[hpo_index]:
            fname = os.path.join(DATA_DIR, f"basil_{triple_idx}_hpo_event_graph_classified.json")
            if os.path.exists(fname): this_folder.append(fname); total_files_found += 1
            else: logging.warning(f"File not found: {fname}")
        for triple_idx in triples[fox_index]:
            fname = os.path.join(DATA_DIR, f"basil_{triple_idx}_fox_event_graph_classified.json")
            if os.path.exists(fname): this_folder.append(fname); total_files_found += 1
            else: logging.warning(f"File not found: {fname}")

        all_folds_paths.append(this_folder)
        logging.info(f"Fold {folder_i} definition includes {len(this_folder)} files.")
    
    logging.info(f"✅ Total files found and assigned to folds: {total_files_found}")
    if total_files_found == 0:
        logging.error(f"❌ FATAL: No files were found using the naming pattern in {DATA_DIR}. Please check file names (e.g., 'basil_0_nyt_event_graph_classified.json') and the DATA_DIR path.")
        exit()

    # --- Run Cross-Validation ---
    all_results = []
    start_fold = max(0, args.start_fold)
    end_fold = min(9, args.end_fold)
    logging.info(f"🚦 Starting 10-fold cross-validation from fold {start_fold} to {end_fold}...")
    overall_start_time = time.time()

    # - Re-implementing the main fold loop
    for i in range(start_fold, end_fold + 1):
        test_fold_index = i
        dev_fold_index = (i - 1 + 10) % 10 # Dev fold is the one before test fold
        # - Re-implementing train/dev/test path assignment
        test_paths = all_folds_paths[test_fold_index]
        dev_paths = all_folds_paths[dev_fold_index]
        train_paths = []
        for j in range(10):
            if j != test_fold_index and j != dev_fold_index:
                train_paths.extend(all_folds_paths[j])

        # Sanity check
        if not train_paths or not dev_paths or not test_paths:
             logging.error(f"❌ FATAL ERROR: Empty data split for fold {i}. Train:{len(train_paths)}, Dev:{len(dev_paths)}, Test:{len(test_paths)}. This likely means files are missing.")
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

         summary_results = {
            'avg_biased_f1': avg_biased_f1,
            'std_biased_f1': std_biased_f1,
            'avg_macro_f1': avg_macro_f1,
            'std_macro_f1': std_macro_f1,
            'fold_results': all_results
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