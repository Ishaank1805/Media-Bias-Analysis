#!/usr/bin/env python3
"""
Ablation 1: No Dual-View (Unified Graph)
This model removes the Factual/Interpretive split.
All events are fed into a single GNN (R_GAT_Unified).
Cross-View Attention and Contrastive Loss are disabled.
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
import torch.multiprocessing as mp
from tqdm import tqdm
import logging


from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer, LongformerModel
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Try to import sentence-transformers for semantic paragraph detection
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠ sentence-transformers not available. Install with: pip install sentence-transformers")
    print("  Will use heuristic paragraph detection instead.\n")

# ==================== COMMAND LINE ARGUMENTS ====================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Ablation 1: Unified Graph GNN for Media Bias Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Testing/debugging arguments
    parser.add_argument('--debug', action='store_true',
                       help='Quick debug mode: 1 fold, 3 epochs, first 10 files only')
    parser.add_argument('--n_folds', type=int, default=10,
                       help='Number of folds to run (1-10)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to use (for quick testing)')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    
    # Model hyperparameters
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--max_len', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--edge_dropout', type=float, default=0.1,
                       help='Adaptive edge dropout rate')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='./BASIL_event_graph_classified',
                       help='Directory containing classified event graphs')
    parser.add_argument('--results_dir', type=str, default='./results_ablation_1',
                       help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='/scratch/atharv.johar/logs',
                       help='Directory to save log files')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_semantic', action='store_true',
                       help='Disable semantic paragraph detection (use heuristic)')
    
    # NEW ARGUMENT: GPU parallelization
    parser.add_argument('--gpu_num', type=int, default=4,
                       help='Number of GPUs to use concurrently (default: 4)')
    
    args = parser.parse_args()
    
    # Debug mode overrides
    if args.debug:
        print("\n🐛 DEBUG MODE ENABLED")
        print("   - Running 1 fold only")
        print("   - Using 3 epochs")
        print("   - Limiting to first 30 files\n")
        args.n_folds = 1
        args.epochs = 3
        args.max_files = 30
    
    return args


# ==================== CONFIGURATION ====================

args = parse_arguments()

def get_device(gpu_id):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id % torch.cuda.device_count()}")
    else:
        return torch.device("cpu")

device = get_device(0)

RESULTS_DIR = args.results_dir
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

print("="*70)
print("ABLATION 1: NO DUAL-VIEW (UNIFIED GRAPH)")
print("="*70)
if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")
else:
    print("Using CPU only")
print(f"Mode: {'DEBUG' if args.debug else 'FULL'}")
print(f"Folds: {args.n_folds}")
print(f"Epochs: {args.epochs}")
if args.max_files:
    print(f"Max files: {args.max_files}")
print(f"GPUs in parallel: {args.gpu_num}")
print("="*70 + "\n")


# ==================== HYPERPARAMETERS ====================

MAX_LEN = args.max_len
num_epochs = args.epochs
batch_size = args.batch_size

lambda_event = 1.0
lambda_coreference = 1.0
lambda_temporal = 1.0
lambda_causal = 1.0
lambda_subevent = 1.0
lambda_contrastive = 0.0 # Disabled for this ablation

no_decay = ['bias', 'LayerNorm.weight']
longformer_weight_decay = 1e-2
non_longformer_weight_decay = 1e-2
warmup_proportion = 0.1
longformer_lr = 1e-5
non_longformer_lr = 2e-5
edge_dropout_rate = args.edge_dropout

# ==================== UTILITY FUNCTIONS ====================

def get_available_classified_files(max_files=None):
    classified_path = args.data_dir
    
    if not os.path.exists(classified_path):
        print(f"❌ ERROR: Data directory not found: {classified_path}")
        print(f"   Please check the path or use --data_dir argument")
        sys.exit(1)
    
    all_files = sorted([f for f in os.listdir(classified_path) 
                       if f.endswith('_classified.json')])
    
    # Limit files if specified
    if max_files is not None:
        all_files = all_files[:max_files]
        print(f"⚠ Limited to first {max_files} files for testing")
    
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
    
    print(f"Found {len(file_info)} classified files")
    print(f"Unique triplets: {len(set([f['triplet'] for f in file_info]))}\n")
    
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
        fold_idx = i % n_folds
        folds[fold_idx].extend([info['path'] for info in triplets[triplet_num]])
    
    return folds

def detect_paragraphs_semantic(sentences):
    """
    INNOVATION 1: Semantic paragraph detection using sentence embeddings
    Detects topic shifts instead of using arbitrary windows
    """
    if not SEMANTIC_AVAILABLE or args.no_semantic:
        return detect_paragraphs_heuristic(sentences)
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    current_para = 0
    sent_data = []
    
    for i, sent in enumerate(sentences):
        if sent.get('sentence_id') == 'title':
            sent['paragraph_id'] = -1
            continue
        
        if len(sent.get('sentence_text', '')) > 1:
            sent_data.append({
                'index': i,
                'text': sent['sentence_text'],
                'sent': sent
            })
    
    if len(sent_data) == 0:
        return sentences
    
    # Batch encode sentences
    texts = [s['text'] for s in sent_data]
    embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=32)
    
    sent_data[0]['sent']['paragraph_id'] = 0
    
    for i in range(1, len(sent_data)):
        # Cosine similarity with previous sentence
        similarity = np.dot(embeddings[i-1], embeddings[i]) / \
                    (np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-8)
        
        # Low similarity = topic shift = new paragraph
        if similarity < 0.70:
            current_para += 1
        
        sent_data[i]['sent']['paragraph_id'] = current_para
    
    return sentences

def detect_paragraphs_heuristic(sentences):
    """Fallback heuristic if semantic detection unavailable"""
    current_para = 0
    prev_id = None
    
    for i, sent in enumerate(sentences):
        if sent.get('sentence_id') == 'title':
            sent['paragraph_id'] = -1
            continue
        
        if 'sentence_id' in sent and isinstance(sent['sentence_id'], int):
            curr_id = sent['sentence_id']
            if prev_id is not None and isinstance(prev_id, int):
                if curr_id - prev_id > 1:
                    current_para += 1
            prev_id = curr_id
        elif i > 0 and i % 5 == 0:
            current_para += 1
        
        sent['paragraph_id'] = current_para
    
    return sentences

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def setup_logger(fold_idx, log_dir):
    """
    Setup logger for a specific fold
    Creates both file and console handlers
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f'fold_{fold_idx}')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create file handler
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{log_dir}/ablation_1_fold_{fold_idx}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler (optional, for critical messages)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - Fold %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# ==================== DATASET ====================

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

        # INNOVATION: Semantic paragraph detection
        article_json['sentences'] = detect_paragraphs_semantic(article_json['sentences'])
        
        token_to_sent = {}
        token_to_para = {}
        for sent_i, sent in enumerate(article_json['sentences']):
            para_id = sent.get('paragraph_id', 0)
            if para_id == -1:
                para_id = 0
            for tok in sent['tokens']:
                token_to_sent[tok['index_of_token']] = sent_i
                token_to_para[tok['index_of_token']] = para_id

        input_ids = []
        attention_mask = []
        label_sentence = []
        
        for sent_i, sent_data in enumerate(article_json['sentences']):
            if len(sent_data.get('sentence_text', '')) > 1:
                start = len(input_ids)
                input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['input_ids'])
                attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['attention_mask'])
                end = len(input_ids)
                
                if end < MAX_LEN:
                    para_id = sent_data.get('paragraph_id', 0)
                    if para_id == -1:
                        para_id = 0
                    
                    label_sentence.append([
                        start, end, sent_i, 
                        sent_data.get('label_info_lex_bias', -1), 
                        para_id
                    ])
                
                for token_data in sent_data['tokens']:
                    token_text = token_data['token_text']
                    word_encoding = tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)
                    
                    if len(input_ids) + len(word_encoding['input_ids']) >= MAX_LEN:
                        break
                        
                    input_ids.extend(word_encoding['input_ids'])
                    attention_mask.extend(word_encoding['attention_mask'])

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
        
        for sent_i, sent_data in enumerate(article_json['sentences']):
            if len(sent_data.get('sentence_text', '')) <= 1:
                continue
                
            current_pos += 1
            
            for token_data in sent_data['tokens']:
                token_text = token_data['token_text']
                start_pos = current_pos
                word_encoding = tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)
                current_pos += len(word_encoding['input_ids'])
                end_pos = current_pos
                
                if end_pos >= MAX_LEN:
                    break
                
                position_map[token_data['index_of_token']] = (start_pos, end_pos)
            
            current_pos += 1
            
            if current_pos >= MAX_LEN:
                break
        
        event_words = []
        if 'event_tokens' in article_json:
            for event_token in article_json['event_tokens']:
                token_idx = event_token['index_of_token']
                
                if token_idx not in position_map:
                    continue
                
                start_pos, end_pos = position_map[token_idx]
                sent_idx = token_to_sent.get(token_idx, 0)
                para_id = token_to_para.get(token_idx, 0)
                
                fi_class = event_token.get('fi_classification', 'FACTUAL')
                fi_label = 1 if fi_class == 'INTERPRETIVE' else 0
                
                event_words.append([
                    start_pos, end_pos, sent_idx, token_idx,
                    event_token['prob_event'][0],
                    event_token['prob_event'][1],
                    event_token['label_event'],
                    fi_label,
                    para_id
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
        label_coreference_raw = []
        label_temporal_raw = []
        label_causal_raw = []
        label_subevent_raw = []

        if event_words.size(0) > 0 and 'relation_label' in article_json:
            token_to_row = {}
            for i in range(event_words.size(0)):
                token_idx = int(event_words[i, 3])
                token_to_row[token_idx] = i
            
            for rel in article_json['relation_label']:
                event_1_idx = rel['event_1']['index_of_token']
                event_2_idx = rel['event_2']['index_of_token']
                
                if event_1_idx in token_to_row and event_2_idx in token_to_row:
                    event_1_row = token_to_row[event_1_idx]
                    event_2_row = token_to_row[event_2_idx]
                    
                    event_pairs_raw.append([event_1_row, event_2_row])
                    
                    label_coreference_raw.append([
                        rel['prob_coreference'][0],
                        rel['prob_coreference'][1],
                        rel['label_coreference']
                    ])
                    label_temporal_raw.append([
                        rel['prob_temporal'][0],
                        rel['prob_temporal'][1],
                        rel['prob_temporal'][2],
                        rel['prob_temporal'][3],
                        rel['label_temporal']
                    ])
                    label_causal_raw.append([
                        rel['prob_causal'][0],
                        rel['prob_causal'][1],
                        rel['prob_causal'][2],
                        rel['label_causal']
                    ])
                    label_subevent_raw.append([
                        rel['prob_subevent'][0],
                        rel['prob_subevent'][1],
                        rel['prob_subevent'][2],
                        rel['label_subevent']
                    ])

        if len(event_pairs_raw) == 0:
            event_pairs = torch.zeros((0, 2), dtype=torch.long)
            label_coreference = torch.zeros((0, 3), dtype=torch.float)
            label_temporal = torch.zeros((0, 5), dtype=torch.float)
            label_causal = torch.zeros((0, 4), dtype=torch.float)
            label_subevent = torch.zeros((0, 4), dtype=torch.float)
        else:
            event_pairs = torch.tensor(event_pairs_raw, dtype=torch.long)
            label_coreference = torch.tensor(label_coreference_raw, dtype=torch.float)
            label_temporal = torch.tensor(label_temporal_raw, dtype=torch.float)
            label_causal = torch.tensor(label_causal_raw, dtype=torch.float)
            label_subevent = torch.tensor(label_subevent_raw, dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_sentence": label_sentence,
            "event_words": event_words,
            "event_pairs": event_pairs,
            "label_coreference": label_coreference,
            "label_temporal": label_temporal,
            "label_causal": label_causal,
            "label_subevent": label_subevent
        }

# ==================== INNOVATIONS (Modified for Ablation) ====================

class AdaptiveEdgeDropout(nn.Module):
    """
    INNOVATION 1: Adaptive edge dropout based on relation confidence
    Drops weak/noisy edges during training to improve robustness
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, event_pairs, relation_probs_dict, training=True):
        if not training or event_pairs.size(0) == 0:
            return event_pairs, relation_probs_dict
        
        # Compute edge importance (max prob across relations, excluding hard label)
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
        
        # Keep top (1 - dropout_rate) edges
        threshold = avg_importance.quantile(self.dropout_rate)
        keep_mask = avg_importance > threshold
        
        filtered_pairs = event_pairs[keep_mask]
        filtered_probs = {}
        for key, val in relation_probs_dict.items():
            filtered_probs[key] = val[keep_mask]
        
        return filtered_pairs, filtered_probs


class MultiScaleAttentionPooling(nn.Module):
    """
    INNOVATION 2: Multi-head attention pooling for event aggregation
    Learns importance weights instead of simple averaging
    """
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
        
        # Use mean as query
        query = embeddings.mean(dim=0, keepdim=True)  # (1, feature_dim)
        
        seq_len = embeddings.size(0)
        
        # Project and reshape for multi-head attention
        Q = self.W_Q(query).view(1, self.num_heads, self.head_dim)  # (1, num_heads, head_dim)
        K = self.W_K(embeddings).view(seq_len, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)
        V = self.W_V(embeddings).view(seq_len, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)
        
        # Transpose for batched matrix multiplication
        Q = Q.transpose(0, 1)  # (num_heads, 1, head_dim)
        K = K.transpose(0, 1)  # (num_heads, N, head_dim)
        V = V.transpose(0, 1)  # (num_heads, N, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)  # (num_heads, 1, N)
        attention = F.softmax(scores, dim=2)  # (num_heads, 1, N)
        
        # Apply attention to values
        out = torch.matmul(attention, V)  # (num_heads, 1, head_dim)
        
        # Concatenate heads
        out = out.transpose(0, 1).contiguous().view(1, -1)  # (1, num_heads * head_dim) = (1, feature_dim)
        
        return self.W_out(out).squeeze(0)  # (feature_dim,)

# ==================== MODEL COMPONENTS ====================

class Token_Embedding(nn.Module):
    def __init__(self):
        super(Token_Embedding, self).__init__()
        self.longformermodel = LongformerModel.from_pretrained(
            'allenai/longformer-base-4096', output_hidden_states=True)

    def forward(self, input_ids, attention_mask):
        # Move inputs to the Longformer device to avoid CPU/GPU mismatch
        device = next(self.longformermodel.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = self.longformermodel(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[2]
        token_embeddings_layers = torch.stack(hidden_states, dim=0)
        token_embeddings_layers = token_embeddings_layers[:, 0, :, :]
        token_embeddings = torch.sum(token_embeddings_layers[-4:, :, :], dim=0)
        return token_embeddings


class R_GAT_Layer(nn.Module):
    """Relation-Aware Graph Attention"""
    def __init__(self, feature_dim):
        super(R_GAT_Layer, self).__init__()
        self.feature_dim = feature_dim
        
        self.relation_types = ['coref', 'before', 'after', 'overlap', 
                               'cause', 'caused', 'contain', 'contained']
        
        self.W_Q = nn.Linear(feature_dim, feature_dim)
        self.W_K = nn.Linear(feature_dim, feature_dim)
        self.W_V = nn.Linear(feature_dim, feature_dim)

        self.W_R = nn.ModuleDict({r: nn.Linear(feature_dim, feature_dim, bias=False) 
                                  for r in self.relation_types})
        self.a_R = nn.ModuleDict({r: nn.Linear(feature_dim * 2, 1) 
                                  for r in self.relation_types})
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, event_embeddings, event_pairs, relation_probs_dict):
        N = event_embeddings.size(0)
        N_pairs = event_pairs.size(0)
        
        if N == 0 or N_pairs == 0:
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
                all_neighbor_V = V[neighbor_indices]
                aggregated_msg = torch.sum(attention_weights * all_neighbor_V, dim=0)
                aggregated_messages.append(aggregated_msg)
            
            if len(aggregated_messages) > 0:
                combined_message = torch.stack(aggregated_messages).mean(dim=0)
                new_embeddings[i] = combined_message + event_embeddings[i]
            else:
                new_embeddings[i] = event_embeddings[i]
                
        return new_embeddings

class GAT_Layer(nn.Module):
    """Document-level paragraph GAT"""
    def __init__(self, feature_dim):
        super(GAT_Layer, self).__init__()
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
        attention_input = torch.cat([H_i, H_j], dim=1)
        e = self.a(attention_input).view(N, N)
        e = e.masked_fill(adj_matrix == 0, float('-inf'))
        attention_weights = F.softmax(self.leakyrelu(e), dim=1)
        new_embeddings = attention_weights @ H
        return new_embeddings


# ==================== MAIN MODEL (Ablation 1) ====================

class Dual_View_Model(nn.Module):
    """
    ABLATION 1: No Dual-View (Unified Graph)
    All events are processed by a single GNN.
    """
    def __init__(self, class_weights=None):
        super(Dual_View_Model, self).__init__()
        feature_dim = 768
        self.feature_dim = feature_dim
        
        # Store class weights as a buffer (not a parameter)
        if class_weights is None:
            class_weights = torch.tensor([1.0, 3.0])
        self.register_buffer('class_weights', class_weights)
        
        self.token_embedding = Token_Embedding()
        self.bilstm_token = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim//2, 
                                     batch_first=True, bidirectional=True)
        
        # Soft distillation heads (unchanged)
        self.event_head_1 = nn.Linear(feature_dim, feature_dim, bias=True)
        nn.init.xavier_uniform_(self.event_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.event_head_1.bias)
        self.event_head_2 = nn.Linear(feature_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.event_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.event_head_2.bias)
        
        self.coreference_head_1 = nn.Linear(feature_dim * 4, feature_dim, bias=True)
        nn.init.xavier_uniform_(self.coreference_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.coreference_head_1.bias)
        self.coreference_head_2 = nn.Linear(feature_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.coreference_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.coreference_head_2.bias)
        
        self.temporal_head_1 = nn.Linear(feature_dim * 4, feature_dim, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_1.bias)
        self.temporal_head_2 = nn.Linear(feature_dim, 4, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_2.bias)
        
        self.causal_head_1 = nn.Linear(feature_dim * 4, feature_dim, bias=True)
        nn.init.xavier_uniform_(self.causal_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_1.bias)
        self.causal_head_2 = nn.Linear(feature_dim, 3, bias=True)
        nn.init.xavier_uniform_(self.causal_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_2.bias)
        
        self.subevent_head_1 = nn.Linear(feature_dim * 4, feature_dim, bias=True)
        nn.init.xavier_uniform_(self.subevent_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.subevent_head_1.bias)
        self.subevent_head_2 = nn.Linear(feature_dim, 3, bias=True)
        nn.init.xavier_uniform_(self.subevent_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.subevent_head_2.bias)
        
        # Innovations (partially kept)
        self.adaptive_dropout = AdaptiveEdgeDropout(edge_dropout_rate)
        
        # --- ABLATION CHANGES ---
        # 1. Removed R_GAT_Interpretive, Cross_View_Attn, I_pooling, contrastive_loss
        # 2. Renamed R_GAT_Factual -> R_GAT_Unified
        # 3. Renamed F_pooling -> Unified_pooling
        # 4. Changed paragraph_agg input from feature_dim*2 to feature_dim
        
        self.R_GAT_Unified = R_GAT_Layer(feature_dim)
        self.Unified_pooling = MultiScaleAttentionPooling(feature_dim, num_heads=4)
        self.paragraph_agg = nn.Linear(feature_dim, feature_dim) # Input is feature_dim, not 2*
        # --- END ABLATION CHANGES ---
        
        self.GAT_Document = GAT_Layer(feature_dim)
        
        # Classifier (unchanged)
        self.bias_sentence_1 = nn.Linear(feature_dim * 3, feature_dim, bias=True)
        nn.init.xavier_uniform_(self.bias_sentence_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.bias_sentence_1.bias)
        self.bias_sentence_2 = nn.Linear(feature_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.bias_sentence_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.bias_sentence_2.bias)
        
        self.relu = nn.ReLU()
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='mean')
        # Note: class_weights will be set via register_buffer in __init__
        self.crossentropyloss_sum = nn.CrossEntropyLoss(weight=None, reduction='sum')
        
    def build_paragraph_adjacency_with_coref(self, N_para, event_words, event_pairs, label_coreference, current_device):
        """
        INNOVATION 4: Cross-paragraph coreference links
        Connect paragraphs that discuss the same events
        """
        adj = torch.eye(N_para).to(current_device)
        
        # Sequential connections
        if N_para > 1:
            for i in range(N_para - 1):
                adj[i, i+1] = 1
                adj[i+1, i] = 1
        
        # Coreference connections (cross-paragraph)
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
        # 1. Determine device from the model parameters to avoid mismatches
        # when submodules (e.g. Longformer) move inputs to CUDA internally.
        current_device = next(self.parameters()).device

        # 2. Move data to the correct device
        input_ids = batch['input_ids'].to(current_device)
        attention_mask = batch['attention_mask'].to(current_device)
        label_sentence = batch['label_sentence'][0].to(current_device)
        event_words = batch['event_words'][0].to(current_device)
        event_pairs = batch['event_pairs'][0].to(current_device)
        label_coreference = batch['label_coreference'][0].to(current_device)
        label_temporal = batch['label_temporal'][0].to(current_device)
        label_causal = batch['label_causal'][0].to(current_device)
        label_subevent = batch['label_subevent'][0].to(current_device)
        
        # 3. Create dummy loss tensors on the correct device
        dummy_loss = torch.tensor(0.0).to(current_device)
        
        # Token encoding
        token_embeddings = self.token_embedding(input_ids, attention_mask)
        token_embeddings = token_embeddings.view(1, token_embeddings.shape[0], token_embeddings.shape[1])

        device_tok = token_embeddings.device
        h0 = torch.zeros(2, 1, self.feature_dim//2, device=device_tok).requires_grad_()
        c0 = torch.zeros(2, 1, self.feature_dim//2, device=device_tok).requires_grad_()
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
            
            # Soft distillation
            event_scores = self.event_head_2(self.relu(self.event_head_1(event_embeddings)))
            event_loss = self.crossentropyloss(event_scores, event_words[:, 4:6])
            
            if event_pairs.size(0) > 0:
                event_1_emb = event_embeddings[event_pairs[:, 0].long()]
                event_2_emb = event_embeddings[event_pairs[:, 1].long()]
                pair_sub = torch.sub(event_1_emb, event_2_emb)
                pair_mul = torch.mul(event_1_emb, event_2_emb)
                event_pair_emb = torch.cat([event_1_emb, event_2_emb, pair_sub, pair_mul], dim=1)
                
                coref_scores = self.coreference_head_2(self.relu(self.coreference_head_1(event_pair_emb)))
                coreference_loss = self.crossentropyloss(coref_scores, label_coreference[:, :2])
                
                temp_scores = self.temporal_head_2(self.relu(self.temporal_head_1(event_pair_emb)))
                temporal_loss = self.crossentropyloss(temp_scores, label_temporal[:, :4])
                
                causal_scores = self.causal_head_2(self.relu(self.causal_head_1(event_pair_emb)))
                causal_loss = self.crossentropyloss(causal_scores, label_causal[:, :3])
                
                sub_scores = self.subevent_head_2(self.relu(self.subevent_head_1(event_pair_emb)))
                subevent_loss = self.crossentropyloss(sub_scores, label_subevent[:, :3])
            else:
                coreference_loss = dummy_loss
                temporal_loss = dummy_loss
                causal_loss = dummy_loss
                subevent_loss = dummy_loss
        else:
            event_embeddings = torch.zeros((0, self.feature_dim)).to(current_device)
            event_loss = dummy_loss
            coreference_loss = dummy_loss
            temporal_loss = dummy_loss
            causal_loss = dummy_loss
            subevent_loss = dummy_loss
        
        # --- ABLATION 1: UNIFIED GRAPH ---
        
        # LEVEL 1: Paragraph-level unified-view
        unique_paras = torch.unique(label_sentence[:, 4])
        para_reps = torch.zeros((len(unique_paras), self.feature_dim)).to(current_device)
        
        for p_idx, p_id in enumerate(unique_paras):
            p_mask = (event_words[:, 8] == p_id).nonzero(as_tuple=True)[0]
            
            if p_mask.size(0) == 0:
                sent_mask = (label_sentence[:, 4] == p_id).nonzero(as_tuple=True)[0]
                if sent_mask.size(0) > 0:
                    para_reps[p_idx] = torch.mean(sentence_embeddings[sent_mask], dim=0)
                continue
            
            # 1. Do NOT split events
            para_events = event_embeddings[p_mask]
            
            # 2. Feed ALL events into the single unified GAT
            Unified_updated = self.R_GAT_Unified(para_events, event_pairs_filtered, relation_probs_filtered)
            
            # 3. Bypass Cross-View Attention
            
            # 4. Pool the unified result
            Unified_summary = self.Unified_pooling(Unified_updated)
            
            # 5. Aggregate the single summary vector
            para_reps[p_idx] = F.relu(self.paragraph_agg(Unified_summary))
        
        # 6. Contrastive loss is not applicable
        contrastive_loss = dummy_loss
        
        # --- END ABLATION ---
        
        # LEVEL 2: Document-level (unchanged)
        N_para = para_reps.size(0)
        
        adj_doc = self.build_paragraph_adjacency_with_coref(
            N_para, event_words, event_pairs, label_coreference, current_device
        )
        
        updated_para_reps = self.GAT_Document(para_reps, adj_doc)
        
        # Map to sentences
        sent_to_para = label_sentence[:, 4].long()
        para_context = updated_para_reps[sent_to_para]
        
        # Event aggregation (unchanged)
        event_agg = torch.zeros_like(sentence_embeddings).to(current_device)
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
            return (torch.zeros((1, 2)).to(current_device), label_bias, 
                   dummy_loss, event_loss,
                   coreference_loss, temporal_loss, causal_loss, subevent_loss,
                   contrastive_loss)
        
        bias_scores = self.bias_sentence_2(self.relu(self.bias_sentence_1(final_sent_emb)))
        # Apply class weights manually since we can't set them in __init__ for multiprocessing
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='sum')
        bias_loss = loss_fn(bias_scores, label_bias)

        return (bias_scores, label_bias, bias_loss, 
                event_loss, coreference_loss, temporal_loss, causal_loss, subevent_loss,
                contrastive_loss)


# ==================== EVALUATION ====================

def evaluate(model, eval_dataloader):
    model.eval()
    all_decisions = []
    all_labels = []

    for step, batch in enumerate(eval_dataloader):
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


# ==================== TRAINING ONE FOLD ====================

def train_one_fold(fold_idx, folders):
    """Train and test a single fold"""
    
    current_device = get_device(fold_idx % args.gpu_num)
    
    # Setup logger for this fold
    logger, log_file = setup_logger(fold_idx, args.log_dir)
    
    logger.info("="*70)
    logger.info(f"STARTING FOLD {fold_idx}/{args.n_folds-1} on {current_device}")
    logger.info("="*70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Device: {current_device}")
    logger.info(f"Random seed: {args.seed}")
    
    print(f"\n{'='*70}")
    print(f"STARTING FOLD {fold_idx}/{args.n_folds-1} on {current_device}")
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

    logger.info(f"Training files: {len(train_files)}")
    logger.info(f"Dev files: {len(dev_files)}")
    logger.info(f"Test files: {len(test_files)}")

    # Create class weights on the correct device
    class_weights = torch.tensor([1.0, 3.0]).to(current_device)
    model = Dual_View_Model(class_weights=class_weights)
    model.to(current_device)
    
    logger.info(f"Model created and moved to {current_device}")
    logger.info(f"Class weights: {class_weights.tolist()}")


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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, 
                                                num_training_steps=num_train_steps)

    logger.info(f"Total training steps: {num_train_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning rates - Longformer: {longformer_lr}, Non-Longformer: {non_longformer_lr}")

    best_dev_f1 = 0
    
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)

    for epoch_i in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch_i+1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        np.random.shuffle(train_files)
        train_dataset = custom_dataset(train_files)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            
        model.train()
        total_bias = 0
        total_distill = 0
        num_batch = 0
        
        progress_bar = tqdm(
            total=len(train_dataloader), 
            desc=f"[Fold {fold_idx} on {current_device}] Epoch {epoch_i+1}/{num_epochs}", 
            ncols=100,
            position=fold_idx,
            leave=False
        )
        
        with progress_bar as pbar:
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

                (bias_scores, labels, bias_loss, event_loss, 
                 coref_loss, temp_loss, causal_loss, sub_loss, contrastive_loss) = model(batch)

                if torch.isnan(bias_loss) or torch.isinf(bias_loss):
                    pbar.update(1)
                    continue
                
                total_bias += bias_loss.item()
                distill_loss = event_loss + coref_loss + temp_loss + causal_loss + sub_loss
                total_distill += distill_loss.item()
                num_batch += 1

                try:
                    # Note: contrastive_loss is 0, so it has no effect
                    (lambda_contrastive * contrastive_loss).backward(retain_graph=True) 
                    
                    event_loss.backward(retain_graph=True)
                    coref_loss.backward(retain_graph=True)
                    temp_loss.backward(retain_graph=True)
                    causal_loss.backward(retain_graph=True)
                    sub_loss.backward(retain_graph=True)
                    bias_loss.backward()
                except RuntimeError:
                    pbar.update(1)
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                pbar.set_postfix({
                    "BiasLoss": f"{bias_loss.item():.2f}",
                    "Distill": f"{distill_loss.item():.2f}"
                })
                pbar.update(1)

        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_bias_loss = total_bias / num_batch if num_batch > 0 else 0
        avg_distill_loss = total_distill / num_batch if num_batch > 0 else 0
        
        logger.info(f"Epoch {epoch_i+1} completed in {format_time(epoch_time)}")
        logger.info(f"  Avg Bias Loss: {avg_bias_loss:.4f}")
        logger.info(f"  Avg Distill Loss: {avg_distill_loss:.4f}")
        logger.info(f"  Batches processed: {num_batch}/{len(train_dataloader)}")

        # Save checkpoint based on dev F1
        logger.info("Evaluating on dev set...")
        precision, recall, dev_f1, _, _, _ = evaluate(model, dev_dataloader)
        
        logger.info(f"Dev Results - P: {precision:.4f}, R: {recall:.4f}, F1: {dev_f1:.4f}")
        
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            checkpoint_path = f"{RESULTS_DIR}/fold_{fold_idx}_best.ckpt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"✓ New best F1! Saved checkpoint to {checkpoint_path}")
        else:
            logger.info(f"  (Best F1 remains: {best_dev_f1:.4f})")

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED")
    logger.info("="*70)
    logger.info(f"Best Dev F1: {best_dev_f1:.4f}")

    # Test
    logger.info("\nLoading best model for testing...")
    model.load_state_dict(torch.load(f"{RESULTS_DIR}/fold_{fold_idx}_best.ckpt", map_location=current_device))
    logger.info("Evaluating on test set...")
    precision, recall, test_f1, macro_f1, test_dec, test_lab = evaluate(model, test_dataloader)

    logger.info("\n" + "="*70)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*70)
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {test_f1:.4f}")
    logger.info(f"Macro F1:  {macro_f1:.4f}")
    logger.info("="*70)

    print(f"[Fold {fold_idx} on {current_device}] TEST: P={precision:.4f}, R={recall:.4f}, F1={test_f1:.4f}")
    
    # Log predictions distribution
    unique, counts = np.unique(test_dec, return_counts=True)
    logger.info(f"Prediction distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    unique, counts = np.unique(test_lab, return_counts=True)
    logger.info(f"Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    logger.info(f"\nLog file saved to: {log_file}")
    logger.info("Fold complete!")

    return {
        'fold': fold_idx,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(test_f1),
        'macro_f1': float(macro_f1),
        'predictions': test_dec.tolist(),
        'labels': test_lab.tolist(),
        'log_file': log_file
    }

# ==================== MAIN EXECUTION ====================

def run_fold_on_gpu(fold_i, folders):
    """Helper for multiprocessing — runs one fold on a specified GPU"""
    return train_one_fold(fold_i, folders)


if __name__ == "__main__":
    # Create log directory at startup
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ABLATION 1: NO DUAL-VIEW (UNIFIED GRAPH)")
    print("="*70 + "\n")
    
    if not SEMANTIC_AVAILABLE or args.no_semantic:
        print("⚠ Running without semantic paragraph detection")
        if not args.no_semantic:
            print("  Install for better results: pip install sentence-transformers\n")
    
    available_files = get_available_classified_files(max_files=args.max_files)
    
    n_folds_for_split = 3 if args.debug else 10
    folders = create_cv_folds_triplet_aware(available_files, n_folds=n_folds_for_split, seed=args.seed)
    
    print(f"Logs will be saved to: {args.log_dir}")
    print(f"Results will be saved to: {RESULTS_DIR}\n")
    
    # ---- PARALLEL FOLD TRAINING ACROSS MULTIPLE GPUS ----
    # Use spawn method for CUDA compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set
    
    pool = mp.Pool(processes=args.gpu_num)

    pool_args = [(fold_i, folders) for fold_i in range(args.n_folds)]

    all_fold_results = []
    
    with tqdm(total=args.n_folds, desc="Overall Folds Progress", ncols=100) as pbar:
        for result in pool.starmap(run_fold_on_gpu, pool_args):
            all_fold_results.append(result)
            pbar.update(1)

    pool.close()
    pool.join()
    
    all_fold_results.sort(key=lambda x: x['fold'])

    # ---- AGGREGATE RESULTS ----
    precisions = [f['precision'] for f in all_fold_results]
    recalls = [f['recall'] for f in all_fold_results]
    f1s = [f['f1'] for f in all_fold_results]

    avg_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    avg_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    avg_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)

    # --- Save final results ---
    summary = {
        'dataset': 'BASIL' if 'BASIL' in args.data_dir else args.data_dir,
        'n_folds': args.n_folds,
        'n_files': len(available_files),
        'method': 'Ablation 1: No Dual-View (Unified Graph)',
        'log_directory': args.log_dir,
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'max_len': args.max_len,
            'seed': args.seed,
            'edge_dropout': args.edge_dropout,
            'contrastive_weight': lambda_contrastive # will be 0.0
        },
        'aggregated_metrics': {
            'precision': {'mean': float(avg_precision), 'std': float(std_precision)},
            'recall': {'mean': float(avg_recall), 'std': float(std_recall)},
            'f1': {'mean': float(avg_f1), 'std': float(std_f1)}
        },
        'per_fold_results': all_fold_results
    }
    
    output_filename = f"{RESULTS_DIR}/final_results_ablation_1.json"
    with open(output_filename, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print(f"FINAL RESULT ({args.n_folds} FOLDS) - ABLATION 1")
    print(f"Results saved to {output_filename}")
    print("="*70)
    print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"F1:        {avg_f1:.4f} ± {std_f1:.4f}")
    print("="*70)
    print(f"All logs saved to: {args.log_dir}")
    print("Individual fold logs:")
    for fold_result in all_fold_results:
        if 'log_file' in fold_result:
            print(f"  Fold {fold_result['fold']}: {fold_result['log_file']}")
    print("="*70 + "\n")
