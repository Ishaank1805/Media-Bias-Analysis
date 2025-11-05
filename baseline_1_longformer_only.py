#!/usr/bin/env python3
"""
Baseline 1: Longformer-Only (No Graph)
This model removes all GNN components and uses only the Longformer + BiLSTM
sentence embeddings for classification.
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
import multiprocessing as mp
from tqdm import tqdm

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
        description='Baseline 1: Longformer-Only GNN for Media Bias Detection',
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
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='./BASIL_event_graph_classified',
                       help='Directory containing classified event graphs')
    parser.add_argument('--results_dir', type=str, default='./results_baseline_1',
                       help='Directory to save results for this baseline')
    
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

device = get_device(0) # Main device for reporting

RESULTS_DIR = args.results_dir
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

print("="*70)
print("BASELINE 1: LONGFORMER-ONLY (NO GRAPH)")
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

CLASS_WEIGHTS = torch.tensor([1.0, 3.0]).to(device)

# Optimizer settings
no_decay = ['bias', 'LayerNorm.weight']
longformer_weight_decay = 1e-2
non_longformer_weight_decay = 1e-2
warmup_proportion = 0.1
longformer_lr = 1e-5
non_longformer_lr = 2e-5

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
    Detects topic shifts instead of using arbitrary windows.
    Kept for dataset compatibility, but the model will not use the paragraph info.
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

# ==================== DATASET ====================

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

class custom_dataset(Dataset):
    """
    This dataset loader remains unchanged.
    It will load all data (events, relations, etc.), but the
    Baseline 1 model will simply ignore everything except
    input_ids, attention_mask, and label_sentence.
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        with open(file_path, "r", encoding='utf-8') as in_json:
            article_json = json.load(in_json)

        # Semantic paragraph detection
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

# ==================== MODEL COMPONENTS ====================

class Token_Embedding(nn.Module):
    def __init__(self):
        super(Token_Embedding, self).__init__()
        self.longformermodel = LongformerModel.from_pretrained(
            'allenai/longformer-base-4096', output_hidden_states=True)

    def forward(self, input_ids, attention_mask):
        # Move inputs to the same device as the Longformer model to avoid
        # "Expected all tensors to be on the same device" runtime errors
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

# ==================== MAIN MODEL ====================

class Dual_View_Model(nn.Module):
    """
    BASELINE 1: Longformer-Only Model
    All GNN, distillation, and contrastive components have been removed.
    """
    def __init__(self):
        super(Dual_View_Model, self).__init__()
        feature_dim = 768
        self.feature_dim = feature_dim
        
        # Base Encoder
        self.token_embedding = Token_Embedding()
        self.bilstm_token = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim//2, 
                                     batch_first=True, bidirectional=True)
        
        # Classifier
        # Input dimension is just feature_dim (from BiLSTM)
        self.bias_sentence_1 = nn.Linear(feature_dim, feature_dim, bias=True)
        nn.init.xavier_uniform_(self.bias_sentence_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.bias_sentence_1.bias)
        self.bias_sentence_2 = nn.Linear(feature_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.bias_sentence_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.bias_sentence_2.bias)
        
        self.relu = nn.ReLU()
        # Only the bias loss is needed
        self.crossentropyloss_sum = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, reduction='sum')
        
        
    def forward(self, batch):
        # 1. Determine current device from the model parameters (safer when inputs
        # are moved inside submodules like the Longformer). This ensures any
        # tensors we create (hidden states, etc.) live on the same device.
        current_device = next(self.parameters()).device

        # 2. Move data to the correct device
        input_ids = batch['input_ids'].to(current_device)
        attention_mask = batch['attention_mask'].to(current_device)
        label_sentence = batch['label_sentence'][0].to(current_device)
        
        # 3. Create dummy loss tensors on the correct device
        dummy_loss = torch.tensor(0.0).to(current_device)
        
        # Token encoding
        token_embeddings = self.token_embedding(input_ids, attention_mask)
        token_embeddings = token_embeddings.view(1, token_embeddings.shape[0], token_embeddings.shape[1])

        # Ensure LSTM hidden states are created on the same device as the
        # token embeddings (which may be placed by the Longformer internally).
        device_tok = token_embeddings.device
        h0 = torch.zeros(2, 1, self.feature_dim//2, device=device_tok).requires_grad_()
        c0 = torch.zeros(2, 1, self.feature_dim//2, device=device_tok).requires_grad_()
        token_embeddings, _ = self.bilstm_token(token_embeddings, (h0, c0))
        token_embeddings = token_embeddings[0, :, :]
        
        sent_start_indices = label_sentence[:, 0].long()
        sentence_embeddings = token_embeddings[sent_start_indices]
        
        # --- All GNN Processing and Event Aggregation Removed ---
        
        final_sent_emb = sentence_embeddings
        
        label_bias = label_sentence[:, 3].long()
        
        # Handle title (label == -1)
        if label_bias[0] == -1:
            label_bias = label_bias[1:]
            final_sent_emb = final_sent_emb[1:, :]
        
        # Handle empty or invalid batches
        if label_bias.size(0) == 0 or (label_bias < 0).any() or (label_bias > 1).any():
            return (torch.zeros((1, 2)).to(current_device), label_bias, 
                   dummy_loss, dummy_loss, dummy_loss, dummy_loss, 
                   dummy_loss, dummy_loss, dummy_loss)
        
        # Final Classification
        bias_scores = self.bias_sentence_2(self.relu(self.bias_sentence_1(final_sent_emb)))
        bias_loss = self.crossentropyloss_sum(bias_scores, label_bias)

        # Return 9 items to match the training loop's unpacking
        return (bias_scores, label_bias, bias_loss, 
                dummy_loss, dummy_loss, dummy_loss, 
                dummy_loss, dummy_loss, dummy_loss)


# ==================== EVALUATION ====================

def evaluate(model, eval_dataloader):
    model.eval()
    all_decisions = []
    all_labels = []

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            # Unpacks 9 items, but only uses the first two
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
    
    # 1. Get the assigned device
    current_device = get_device(fold_idx % args.gpu_num)
    
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

    # print(f"[Fold {fold_idx}] Train: {len(train_files)}, Dev: {len(dev_files)}, Test: {len(test_files)}")

    # 2. Move model and class weights to the correct device
    model = Dual_View_Model()
    model.to(current_device)
    global CLASS_WEIGHTS # Need to update the global var for the model
    CLASS_WEIGHTS = torch.tensor([1.0, 3.0]).to(current_device)


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

    best_dev_f1 = 0

    for epoch_i in range(num_epochs):
        np.random.shuffle(train_files)
        train_dataset = custom_dataset(train_files)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            
        model.train()
        total_bias = 0
        num_batch = 0

        # Use tqdm but disable for non-main process to avoid spam
        progress_bar = tqdm(
            total=len(train_dataloader), 
            desc=f"[Fold {fold_idx} on {current_device}] Epoch {epoch_i+1}/{num_epochs}", 
            ncols=100,
            disable=(fold_idx % args.gpu_num != 0) # Only show for first GPU in pool
        )
        
        with progress_bar as pbar:
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

                # Model returns 9 items, but we only care about the bias_loss (item 2)
                (bias_scores, labels, bias_loss, 
                 event_loss, coref_loss, temp_loss, 
                 causal_loss, sub_loss, contrastive_loss) = model(batch)

                if torch.isnan(bias_loss) or torch.isinf(bias_loss) or bias_loss.item() == 0:
                    pbar.update(1)
                    continue
                
                total_bias += bias_loss.item()
                num_batch += 1

                try:
                    # ONLY backpropagate the bias_loss
                    bias_loss.backward()
                except RuntimeError:
                    pbar.update(1)
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                pbar.set_postfix({"BiasLoss": f"{bias_loss.item():.2f}"})
                pbar.update(1)

        # Save checkpoint based on dev F1
        precision, recall, dev_f1, _, _, _ = evaluate(model, dev_dataloader)
        
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(model.state_dict(), f"{RESULTS_DIR}/fold_{fold_idx}_best.ckpt")

    # Test
    model.load_state_dict(torch.load(f"{RESULTS_DIR}/fold_{fold_idx}_best.ckpt", map_location=current_device))
    precision, recall, test_f1, macro_f1, test_dec, test_lab = evaluate(model, test_dataloader)

    print(f"[Fold {fold_idx} on {current_device}] TEST: P={precision:.4f}, R={recall:.4f}, F1={test_f1:.4f}")

    return {
        'fold': fold_idx,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(test_f1),
        'macro_f1': float(macro_f1),
        'predictions': test_dec.tolist(),
        'labels': test_lab.tolist()
    }

# ==================== MAIN EXECUTION ====================

def run_fold_on_gpu(fold_i, folders):
    """Helper for multiprocessing — runs one fold on a specified GPU"""
    gpu_id = fold_i % args.gpu_num
    # This function is now just a wrapper to pass the fold_idx and folders
    # The get_device() call inside train_one_fold will handle assignment
    return train_one_fold(fold_i, folders)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BASELINE 1: LONGFORMER-ONLY (NO GRAPH)")
    print("="*70 + "\n")
    
    if not SEMANTIC_AVAILABLE or args.no_semantic:
        print("⚠ Running without semantic paragraph detection")
        if not args.no_semantic:
            print("  Install for better results: pip install sentence-transformers\n")
    
    available_files = get_available_classified_files(max_files=args.max_files)
    
    n_folds_for_split = 3 if args.debug else 10
    folders = create_cv_folds_triplet_aware(available_files, n_folds=n_folds_for_split, seed=args.seed)
    
    # ---- PARALLEL FOLD TRAINING ACROSS MULTIPLE GPUS ----
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=args.gpu_num)

    # Create a list of arguments for the pool
    pool_args = [(fold_i, folders) for fold_i in range(args.n_folds)]

    all_fold_results = []
    
    # Use tqdm to track the progress of the folds completing
    with tqdm(total=args.n_folds, desc="Overall Folds Progress", ncols=100) as pbar:
        for result in pool.starmap(run_fold_on_gpu, pool_args):
            all_fold_results.append(result)
            pbar.update(1)

    pool.close()
    pool.join()
    
    # Ensure results are sorted by fold index
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
        'method': 'Baseline 1: Longformer-Only (No Graph)',
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'max_len': args.max_len,
            'seed': args.seed
        },
        'aggregated_metrics': {
            'precision': {'mean': float(avg_precision), 'std': float(std_precision)},
            'recall': {'mean': float(avg_recall), 'std': float(std_recall)},
            'f1': {'mean': float(avg_f1), 'std': float(std_f1)}
        },
        'per_fold_results': all_fold_results
    }
    
    output_filename = f"{RESULTS_DIR}/final_results_baseline_1.json"
    with open(output_filename, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print(f"FINAL RESULT ({args.n_folds} FOLDS) - BASELINE 1")
    print(f"Results saved to {output_filename}")
    print("="*70)
    print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"F1:        {avg_f1:.4f} ± {std_f1:.4f}")
