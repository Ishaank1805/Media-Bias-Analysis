# ============================================================================
# EVENT RELATION GRAPH NEURAL NETWORK TRAINING SCRIPT
# ============================================================================
# This script trains a deep learning model to extract events and their relations
# from news articles. It performs 5 main tasks:
# 1. Event Identification: Detect which words/phrases are event triggers
# 2. Event Coreference: Determine which event mentions refer to the same event
# 3. Temporal Relations: Classify temporal relationships (before/after/overlap)
# 4. Causal Relations: Identify cause-effect relationships between events
# 5. Subevent Relations: Find hierarchical relationships (parent/child events)
#
# The model uses Longformer (a transformer designed for long documents) as the
# base encoder, followed by BiLSTM and task-specific classification heads.
# ============================================================================

import os
# Set which GPU to use (GPU 0 in this case)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch

# ============================================================================
# GPU/CPU Device Configuration
# ============================================================================
# Check if CUDA (NVIDIA GPU support) is available and configure accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# ============================================================================
# Import Required Libraries
# ============================================================================

# Data manipulation and processing
import pandas as pd
import numpy as np
import json

# PyTorch components for deep learning
from torch.utils.data import Dataset              # Base class for custom datasets
from tqdm import tqdm                             # Progress bars
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim                           # Optimizers
import torch.nn as nn                             # Neural network modules
import torch.nn.functional as F                   # Functional operations (softmax, etc.)
import torch

# Hugging Face Transformers for pre-trained language models
from transformers import LongformerTokenizer, LongformerModel
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW                     # Adam optimizer with weight decay

# Utilities
import math
import random

# Scikit-learn for evaluation metrics
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# For coreference evaluation
from collections import Counter
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm for CEAFE metric




# ============================================================================
# HYPERPARAMETERS AND TRAINING CONFIGURATION
# ============================================================================

# --- Model Input Configuration ---
MAX_LEN = 2048          # Maximum sequence length (Longformer can handle up to 4096)
batch_size = 1          # Process one article at a time (due to memory constraints)
num_epochs = 5          # Number of complete passes through the training data
check_times = 10 * num_epochs  # How many times to evaluate on dev set during training

# --- Coreference Training Method ---
# Two approaches for training coreference resolution:
# 1. "event_pairs": Binary classification for each event pair (coreferent or not)
# 2. "event_cluster": Cluster-based approach using antecedent probabilities
#coreference_train_method = "event_cluster"
coreference_train_method = "event_pairs"

# --- Loss Weighting for Class Imbalance ---
# These weights address class imbalance in the dataset
# Positive classes are often rare, so we weight them higher during training

# Event identification weights
event_weight_positive = 1           # Weight for positive event labels

# Coreference weights
coreference_weight_positive = 1     # Weight for coreferent pairs

# Temporal relation weights (4 classes: none, before, after, overlap)
temporal_weight_before = 1          # Weight for "before" temporal relation
temporal_weight_after = 1           # Weight for "after" temporal relation
temporal_weight_overlap = 1         # Weight for "overlap" temporal relation

# Causal relation weights (3 classes: none, causes, caused_by)
causal_weight_cause = 1             # Weight for "causes" causal relation
causal_weight_caused = 1            # Weight for "caused_by" causal relation

# Subevent relation weights (3 classes: none, contains, contained_by)
subevent_weight_contain = 1         # Weight for "contains" subevent relation
subevent_weight_contained = 1       # Weight for "contained_by" subevent relation

# --- Optimizer Configuration ---
# Parameters that should not have weight decay (normalization and bias terms)
no_decay = ['bias', 'LayerNorm.weight']

# Weight decay (L2 regularization) to prevent overfitting
longformer_weight_decay = 1e-2      # Weight decay for pre-trained Longformer layers
non_longformer_weight_decay = 1e-2  # Weight decay for task-specific layers

# Learning rate schedule
warmup_proportion = 0.1             # Fraction of training for learning rate warmup

# Learning rates (different for pre-trained vs task-specific layers)
non_longformer_lr = 1e-4            # Learning rate for task-specific layers
longformer_lr = 1e-5                # Learning rate for Longformer (lower to preserve pre-training)




# ============================================================================
# UTILITY FUNCTION: Create File Paths
# ============================================================================

def create_file_path(parent_path, file_names_list):
    """
    Construct full file paths by combining parent directory with file names.
    
    Args:
        parent_path (str): The parent directory path
        file_names_list (list): List of file names
        
    Returns:
        list: List of complete file paths
    """
    file_paths_list = []
    for file_i in range(len(file_names_list)):
        file_name = file_names_list[file_i]
        file_path = parent_path + file_name
        file_paths_list.append(file_path)
    return file_paths_list





# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================
# This class loads and processes individual article JSON files into tensors
# suitable for training the neural network model.
# ============================================================================

# Initialize the Longformer tokenizer (converts text to token IDs)
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')


class custom_dataset(Dataset):
    """
    PyTorch Dataset class for loading preprocessed MAVEN-ERE articles.
    
    This dataset:
    1. Loads article JSON files created by the preprocessing script
    2. Tokenizes the article text using Longformer tokenizer
    3. Creates labels for all tasks (event detection, relations, etc.)
    4. Pads/truncates sequences to MAX_LEN
    
    Args:
        file_paths (list): List of paths to preprocessed JSON files
    """
    
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        """Return the total number of articles in the dataset"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load and process a single article.
        
        Args:
            idx (int): Index of the article to load
            
        Returns:
            dict: Dictionary containing all processed tensors for this article:
                - input_ids: Token IDs for Longformer
                - attention_mask: Mask indicating real vs padding tokens
                - label_event: Event labels for each word/phrase
                - event_pairs: Indices of all event pairs
                - label_coreference: Coreference labels for each pair
                - label_temporal: Temporal relation labels for each pair
                - label_causal: Causal relation labels for each pair
                - label_subevent: Subevent relation labels for each pair
        """
        # Get the path to this article's JSON file
        file_path = self.file_paths[idx]

        # Load the preprocessed article data
        with open(file_path, "r") as in_json:
            article_json = json.load(in_json)

        # ====================================================================
        # Step 1: Initialize lists for tokenization
        # ====================================================================
        input_ids = []        # Will store token IDs
        attention_mask = []   # Will indicate which tokens are real (1) vs padding (0)

        # label_event stores: [start_idx, end_idx, event_label]
        # - start_idx, end_idx: position in input_ids where this word's tokens begin/end
        # - event_label: 0 (non-event) or 1 (event trigger)
        label_event = []

        # ====================================================================
        # Step 2: Add special tokens and tokenize first word
        # ====================================================================
        # Add the <s> start-of-sequence token
        input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens = False)['input_ids'])
        attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens = False)['attention_mask'])

        # Tokenize the first word (no leading space)
        start = len(input_ids)
        word_encoding = tokenizer.encode_plus(article_json['event_label'][0]['token'], add_special_tokens = False)
        input_ids.extend(word_encoding['input_ids'])
        attention_mask.extend(word_encoding['attention_mask'])
        end = len(input_ids)
        # Record where this word's tokens are and whether it's an event
        label_event.append([start, end, article_json['event_label'][0]['event_label']])

        # ====================================================================
        # Step 3: Tokenize remaining words (with leading space)
        # ====================================================================
        # Process all subsequent words
        for word_i in range(1, len(article_json['event_label'])):
            start = len(input_ids)
            # Add space before word (important for proper tokenization)
            word_encoding = tokenizer.encode_plus(' ' + article_json['event_label'][word_i]['token'], add_special_tokens=False)
            input_ids.extend(word_encoding['input_ids'])
            attention_mask.extend(word_encoding['attention_mask'])
            end = len(input_ids)
            # Record this word's position and event label
            label_event.append([start, end, article_json['event_label'][word_i]['event_label']])

        # ====================================================================
        # Step 4: Add end token and padding
        # ====================================================================
        # Add the </s> end-of-sequence token
        input_ids.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['input_ids'])
        attention_mask.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['attention_mask'])
        
        # Calculate how much padding is needed to reach MAX_LEN
        num_pad = MAX_LEN - len(input_ids)
        if num_pad > 0:
            # Add <pad> tokens to reach MAX_LEN
            input_ids.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['input_ids'])
            attention_mask.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['attention_mask'])

        # ====================================================================
        # Step 5: Validation - verify tokenization matches expected output
        # ====================================================================
        # Check that our word-by-word tokenization matches whole-article tokenization
        if len(input_ids) > MAX_LEN:
            # Article is longer than MAX_LEN (will be truncated by model)
            if tokenizer.encode_plus(" ".join(article_json['tokens_list']), add_special_tokens=True)['input_ids'] != input_ids:
                print("word tokenizer unmatched with article tokenizer in " + file_path)
        else:
            # Article fits within MAX_LEN
            if tokenizer.encode_plus(" ".join(article_json['tokens_list']), add_special_tokens = True,
                                     max_length=MAX_LEN, padding='max_length', truncation=True)['input_ids'] != input_ids:
                print("word tokenizer unmatched with article tokenizer in " + file_path)

        # Verify we have labels for all words
        if len(label_event) != len(article_json['event_label']):
            print("number of words unmatched")

        # ====================================================================
        # Step 6: Convert lists to tensors
        # ====================================================================
        input_ids = torch.tensor(input_ids)             # Shape: [sequence_length]
        attention_mask = torch.tensor(attention_mask)   # Shape: [sequence_length]
        label_event = torch.tensor(label_event)         # Shape: [num_words, 3]

        # ====================================================================
        # Step 7: Extract event pair relation labels
        # ====================================================================
        # For each pair of events, we need labels for all relation types
        event_pairs = []           # Stores indices of events in each pair
        label_coreference = []     # Coreference labels for each pair
        label_temporal = []        # Temporal relation labels
        label_causal = []          # Causal relation labels
        label_subevent = []        # Subevent relation labels

        # Iterate through all event pairs defined in the preprocessed data
        for event_pair_i in range(len(article_json['relation_label'])):
            # Store the indices (in label_event) of the two events in this pair
            event_pairs.append([article_json['relation_label'][event_pair_i]['event_1']['index_in_event_label'],
                                article_json['relation_label'][event_pair_i]['event_2']['index_in_event_label']])

            # Store all relation labels for this pair
            label_coreference.append(article_json['relation_label'][event_pair_i]['label_coreference'])
            label_temporal.append(article_json['relation_label'][event_pair_i]['label_temporal'])
            label_causal.append(article_json['relation_label'][event_pair_i]['label_causal'])
            label_subevent.append(article_json['relation_label'][event_pair_i]['label_subevent'])

        # Convert to tensors
        event_pairs = torch.tensor(event_pairs)             # Shape: [num_pairs, 2]
        label_coreference = torch.tensor(label_coreference) # Shape: [num_pairs]
        label_temporal = torch.tensor(label_temporal)       # Shape: [num_pairs]
        label_causal = torch.tensor(label_causal)           # Shape: [num_pairs]
        label_subevent = torch.tensor(label_subevent)       # Shape: [num_pairs]

        # ====================================================================
        # Step 8: Return all data as a dictionary
        # ====================================================================
        dict = {"input_ids": input_ids, "attention_mask": attention_mask,
                "label_event": label_event, "event_pairs": event_pairs, "label_coreference": label_coreference,
                "label_temporal": label_temporal, "label_causal": label_causal, "label_subevent": label_subevent}

        return dict






# ============================================================================
# NEURAL NETWORK MODEL ARCHITECTURE
# ============================================================================


# ============================================================================
# Utility Functions for Tensor Operations
# ============================================================================

def to_var(x):
    """
    Convert a tensor to a backpropagation-enabled tensor and move to GPU.
    
    Args:
        x (Tensor): Input tensor
        
    Returns:
        Tensor: GPU tensor with gradient tracking enabled
    """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """
    Move a tensor to GPU if CUDA is available.
    
    Args:
        x (Tensor): Input tensor
        
    Returns:
        Tensor: GPU tensor if CUDA available, otherwise unchanged
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def pad_and_stack(tensors, pad_size=None, value=0.0):
    """
    Pad a list of tensors to the same size and stack them.
    Used for batching tensors of different lengths.
    
    Args:
        tensors (list): List of 1D tensors
        pad_size (int): Target size (uses max if None)
        value (float): Padding value
        
    Returns:
        Tensor: Stacked and padded tensor
    """
    sizes = [s.shape[0] for s in tensors]
    if not pad_size:
        pad_size = max(sizes)

    padded = []
    for tensor, size in zip(tensors, sizes):
        # Pad tensor to target size
        padded.append(torch.cat((tensor, to_cuda(torch.tensor(value).repeat(pad_size - size)))))
    padded = torch.stack(padded, dim = 0)

    return padded

def flatten(lists):
    """
    Flatten a list of lists into a single list.
    
    Args:
        lists (list): List of lists
        
    Returns:
        list: Flattened list
    """
    return [item for l in lists for item in l]

def fill_expand(labels):
    """
    Convert coreference clusters to a pairwise coreference matrix.
    Used for cluster-based coreference training.
    
    Args:
        labels (list): List of clusters, where each cluster is a list of event indices
        
    Returns:
        Tensor: Matrix where [i][j] = 1 if events i and j are coreferent
        
    Example:
        labels = [[0, 2], [1], [3, 4]]  # Event 0 and 2 are coreferent, etc.
        Returns a 5x5 matrix with 1s for coreferent pairs
    """
    # Find total number of events
    event_num = max(flatten(labels)) + 1
    
    # Initialize matrix (all zeros = not coreferent)
    filled_labels = torch.zeros((event_num, event_num))
    
    # For each cluster
    for gr in labels:
        if len(gr) > 1:
            # Multiple events in cluster - mark all pairs as coreferent
            sorted_gr = sorted(gr)
            for i in range(len(sorted_gr)):
                for j in range(i+1, len(sorted_gr)):
                    filled_labels[sorted_gr[j]][sorted_gr[i]] = 1
        else:
            # Singleton cluster - event is its own antecedent
            try:
                filled_labels[gr[0]][gr[0]] = 1
            except:
                print(gr)
                raise ValueError
    return filled_labels



# ============================================================================
# Token Embedding Module
# ============================================================================

class Token_Embedding(nn.Module):
    """
    Encodes input tokens using Longformer transformer model.
    
    Longformer is specifically designed for long documents (up to 4096 tokens)
    using efficient attention mechanisms.
    
    Input:
        - input_ids: Token IDs from tokenizer, shape [1, sequence_length]
        - attention_mask: Binary mask for real vs padding tokens, shape [1, sequence_length]
        
    Output:
        - token_embeddings: Contextual embeddings for each token, shape [sequence_length, 768]
                           (768 is the hidden size of longformer-base)
    """

    def __init__(self):
        super(Token_Embedding, self).__init__()

        # Load pre-trained Longformer model
        # output_hidden_states=True gives us all layer outputs for feature extraction
        self.longformermodel = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True, )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through Longformer.
        
        Strategy: Sum the last 4 hidden layers for richer representations
        (common practice in BERT-like models)
        """
        
        # Get outputs from Longformer
        outputs = self.longformermodel(input_ids = input_ids, attention_mask = attention_mask)
        
        # outputs[2] contains all hidden states (13 layers total: embedding + 12 transformer layers)
        hidden_states = outputs[2]
        
        # Stack layers: shape [13, batch_size (1), num_tokens, 768]
        token_embeddings_layers = torch.stack(hidden_states, dim=0)
        
        # Remove batch dimension: shape [13, num_tokens, 768]
        token_embeddings_layers = token_embeddings_layers[:, 0, :, :]
        
        # Sum last 4 layers for final representation: shape [num_tokens, 768]
        # Research shows this often works better than using just the last layer
        token_embeddings = torch.sum(token_embeddings_layers[-4:, :, :], dim = 0)

        return token_embeddings



# ============================================================================
# Main Event Relation Graph Model
# ============================================================================

class Event_Relation_Graph(nn.Module):
    """
    Multi-task neural network for event extraction and relation classification.
    
    Architecture:
    1. Token Embedding Layer: Longformer transformer
    2. BiLSTM Layer: Captures sequential dependencies
    3. Event Identification Head: Binary classification (event vs non-event)
    4. Coreference Head: Identifies coreferent event mentions
    5. Temporal Relation Head: 4-way classification (none/before/after/overlap)
    6. Causal Relation Head: 3-way classification (none/causes/caused_by)
    7. Subevent Relation Head: 3-way classification (none/contains/contained_by)
    
    The model uses multi-task learning, training all tasks simultaneously
    with shared lower layers and task-specific classification heads.
    
    Inputs:
        - input_ids: Tokenized article text
        - attention_mask: Valid token mask
        - label_event: Event identification labels
        - event_pairs: Pairs of events for relation classification
        - label_coreference, label_temporal, label_causal, label_subevent: Relation labels
        - coreference_train_method: "event_pairs" or "event_cluster"
    
    Outputs (depends on coreference_train_method):
        - Weighted losses for all tasks
        - Raw scores (logits) for all tasks
        - Predicted/gold coreference clusters
    """

    def __init__(self):
        """
        Initialize all model components:
        - Token embedding layer (Longformer)
        - BiLSTM for sequential modeling
        - Classification heads for each task
        """
        super(Event_Relation_Graph, self).__init__()

        # ====================================================================
        # Base Layers
        # ====================================================================
        
        # Longformer encoder for token embeddings
        self.token_embedding = Token_Embedding()

        # Bidirectional LSTM to capture sequential context
        # Input: 768 (Longformer output), Output: 2 * 384 = 768 (bidirectional)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)

        # ====================================================================
        # Event Identification Head (Binary Classification: event or not)
        # ====================================================================
        # Two-layer network: 768 -> 768 -> 2
        
        self.event_head_1 = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.event_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.event_head_1.bias)

        self.event_head_2 = nn.Linear(768, 2, bias=True)  # Output: [non-event, event]
        nn.init.xavier_uniform_(self.event_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.event_head_2.bias)

        # ====================================================================
        # Coreference Head (depends on training method)
        # ====================================================================
        # Input: Concatenation of [event1, event2, event1-event2, event1*event2] = 768*4
        
        if coreference_train_method == "event_pairs":
            # Binary classification: coreferent or not
            # Three-layer network: 768*4 -> 768 -> 256 -> 2
            
            self.coreference_head_1 = nn.Linear(768 * 4, 768, bias=True)
            nn.init.xavier_uniform_(self.coreference_head_1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_1.bias)

            self.coreference_head_2 = nn.Linear(768, 256, bias=True)
            nn.init.xavier_uniform_(self.coreference_head_2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_2.bias)

            self.coreference_head_3 = nn.Linear(256, 2, bias=True)  # [not coreferent, coreferent]
            nn.init.xavier_uniform_(self.coreference_head_3.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_3.bias)

        else: # coreference_train_method == "event_cluster"
            # Antecedent scoring: single value per pair
            # Three-layer network: 768*4 -> 768 -> 256 -> 1
            
            self.coreference_head_1 = nn.Linear(768 * 4, 768, bias=True)
            nn.init.xavier_uniform_(self.coreference_head_1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_1.bias)

            self.coreference_head_2 = nn.Linear(768, 256, bias=True)
            nn.init.xavier_uniform_(self.coreference_head_2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_2.bias)

            self.coreference_head_3 = nn.Linear(256, 1, bias=True)  # Antecedent score
            nn.init.xavier_uniform_(self.coreference_head_3.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_3.bias)

        # ====================================================================
        # Temporal Relation Head (4-class: none/before/after/overlap)
        # ====================================================================
        # Three-layer network: 768*4 -> 768 -> 256 -> 4
        
        self.temporal_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_1.bias)

        self.temporal_head_2 = nn.Linear(768, 256, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_2.bias)

        self.temporal_head_3 = nn.Linear(256, 4, bias=True)  # [none, before, after, overlap]
        nn.init.xavier_uniform_(self.temporal_head_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_3.bias)

        # ====================================================================
        # Causal Relation Head (3-class: none/causes/caused_by)
        # ====================================================================
        # Three-layer network: 768*4 -> 768 -> 256 -> 3
        
        self.causal_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.causal_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_1.bias)

        self.causal_head_2 = nn.Linear(768, 256, bias=True)
        nn.init.xavier_uniform_(self.causal_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_2.bias)

        self.causal_head_3 = nn.Linear(256, 3, bias=True)  # [none, causes, caused_by]
        nn.init.xavier_uniform_(self.causal_head_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_3.bias)

        # ====================================================================
        # Subevent Relation Head (3-class: none/contains/contained_by)
        # ====================================================================
        # Three-layer network: 768*4 -> 768 -> 256 -> 3
        
        self.subevent_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.subevent_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.subevent_head_1.bias)

        self.subevent_head_2 = nn.Linear(768, 256, bias=True)
        nn.init.xavier_uniform_(self.subevent_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.subevent_head_2.bias)

        self.subevent_head_3 = nn.Linear(256, 3, bias=True)  # [none, contains, contained_by]
        nn.init.xavier_uniform_(self.subevent_head_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.subevent_head_3.bias)

        # ====================================================================
        # Activation and Loss Functions
        # ====================================================================
        
        self.relu = nn.ReLU()  # Activation function for hidden layers
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='none')  # No reduction to apply custom weights



    def forward(self, input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent, coreference_train_method):
        """
        Forward pass through the entire model.
        
        This method performs:
        1. Token encoding (Longformer + BiLSTM)
        2. Event identification
        3. Event pair encoding
        4. All relation classifications
        5. Loss computation with class weighting
        6. Cluster formation for coreference
        
        Args:
            input_ids: Tokenized input, shape [1, seq_len]
            attention_mask: Attention mask, shape [1, seq_len]
            label_event: Event labels, shape [num_words, 3] - [start, end, label]
            event_pairs: Event pair indices, shape [num_pairs, 2]
            label_coreference: Coreference labels, shape [num_pairs]
            label_temporal: Temporal labels, shape [num_pairs]
            label_causal: Causal labels, shape [num_pairs]
            label_subevent: Subevent labels, shape [num_pairs]
            coreference_train_method: "event_pairs" or "event_cluster"
            
        Returns:
            Tuple of losses, scores, and clusters (exact contents depend on coreference_train_method)
        """

        # ====================================================================
        # STEP 1: Encode tokens using Longformer
        # ====================================================================
        # Get contextual embeddings for all tokens
        token_embeddings = self.token_embedding(input_ids, attention_mask) # Shape: [num_tokens, 768]

        # ====================================================================
        # STEP 2: Apply BiLSTM for sequential modeling
        # ====================================================================
        # Reshape for LSTM: [batch_size=1, num_tokens, 768]
        token_embeddings = token_embeddings.view(1, token_embeddings.shape[0], token_embeddings.shape[1])

        # Initialize hidden and cell states for BiLSTM
        h0 = torch.zeros(2, 1, 384).cuda().requires_grad_()  # 2 for bidirectional
        c0 = torch.zeros(2, 1, 384).cuda().requires_grad_()

        # Pass through BiLSTM
        token_embeddings, (_, _) = self.bilstm(token_embeddings, (h0, c0))  # [1, num_tokens, 768]
        token_embeddings = token_embeddings[0, :, :]  # Remove batch dim: [num_tokens, 768]


        # ====================================================================
        # STEP 3: Event Identification Task
        # ====================================================================
        # For each word/phrase in label_event, compute its embedding by
        # averaging the token embeddings within its span
        
        for token_i in range(label_event.shape[0]):
            # Get the span of tokens for this word
            start_in_input_ids = label_event[token_i, 0]
            end_in_input_ids = label_event[token_i, 1]
            
            # Average token embeddings within span
            word_embedding = torch.mean(token_embeddings[start_in_input_ids: end_in_input_ids, :], dim = 0).view(1, 768)
            
            # Build matrix of all word embeddings
            if token_i == 0:
                event_embeddings = word_embedding
            else:
                event_embeddings = torch.cat((event_embeddings, word_embedding), dim = 0)

        # Pass through event identification head
        # Shape: [num_words, 768] -> [num_words, 2] (logits for non-event vs event)
        event_raw_scores = self.event_head_2(self.relu(self.event_head_1(event_embeddings)))
        
        # Compute cross-entropy loss for each word
        event_loss = self.crossentropyloss(event_raw_scores, label_event[:,2])  # Shape: [num_words]

        # Apply class-specific weights to handle class imbalance
        # Weight negative examples (label=0) with weight 1
        event_loss_weight_0 = (label_event[:, 2] == 0).int()
        event_loss_0 = torch.mul(event_loss, event_loss_weight_0)

        # Weight positive examples (label=1) with configurable weight
        event_loss_weight_1 = torch.mul(label_event[:,2], event_weight_positive)
        event_loss_1 = torch.mul(event_loss, event_loss_weight_1)

        # Combine and sum to get total weighted loss
        event_weighted_loss = torch.add(event_loss_0, event_loss_1)
        event_weighted_loss = torch.sum(event_weighted_loss)


        # ====================================================================
        # STEP 4: Convert coreference pairs to clusters (for evaluation)
        # ====================================================================
        # Build gold coreference clusters from pairwise labels
        # This is used for computing CoNLL metrics later
        
        number_of_events = torch.sum((label_event[:, 2] == 1).int())  # Count actual events

        label_coreference_cluster = []  # Will store clusters as lists of event indices
        
        # Iterate through each event
        for event_idx in range(number_of_events):

            # Find which cluster this event belongs to (if any)
            event1_index_in_cluster = -1
            for cluster_i in range(len(label_coreference_cluster)):
                if event_idx in label_coreference_cluster[cluster_i]:
                    event1_index_in_cluster = cluster_i

            # Check all events after this one for coreference
            # Use prefix sum to find the correct index in label_coreference
            prefix_sum = int((2 * number_of_events - 1 - event_idx) * event_idx / 2)
            
            for j in range(0, number_of_events - 1 - event_idx):
                if label_coreference[prefix_sum + j] == 1:  # These events are coreferent
                    
                    # Find cluster of the second event
                    event2_index_in_cluster = -1
                    for cluster_i in range(len(label_coreference_cluster)):
                        if (event_idx + j + 1) in label_coreference_cluster[cluster_i]:
                            event2_index_in_cluster = cluster_i

                    # Merge clusters or create new one based on which events are already clustered
                    if event1_index_in_cluster == -1 and event2_index_in_cluster == -1:
                        # Neither event in a cluster - create new cluster
                        label_coreference_cluster.append([event_idx, event_idx + j + 1])
                        event1_index_in_cluster = len(label_coreference_cluster) - 1
                        event2_index_in_cluster = len(label_coreference_cluster) - 1
                    if event1_index_in_cluster != -1 and event2_index_in_cluster == -1:
                        # Add event2 to event1's cluster
                        label_coreference_cluster[event1_index_in_cluster].append(event_idx + j + 1)
                        event2_index_in_cluster = event1_index_in_cluster
                    if event1_index_in_cluster == -1 and event2_index_in_cluster != -1:
                        # Add event1 to event2's cluster
                        label_coreference_cluster[event2_index_in_cluster].append(event_idx)
                        event1_index_in_cluster = event2_index_in_cluster
                    if event1_index_in_cluster != -1 and event2_index_in_cluster != -1 and event1_index_in_cluster != event2_index_in_cluster:
                        # Merge two existing clusters
                        label_coreference_cluster[event1_index_in_cluster].extend(label_coreference_cluster[event2_index_in_cluster])
                        label_coreference_cluster.pop(event2_index_in_cluster)
                        # Recalculate event1's cluster index after removal
                        for cluster_i in range(len(label_coreference_cluster)):
                            if event_idx in label_coreference_cluster[cluster_i]:
                                event1_index_in_cluster = cluster_i
                        event2_index_in_cluster = event1_index_in_cluster

            # If event not in any cluster, create singleton cluster
            if event1_index_in_cluster == -1:
                label_coreference_cluster.append([event_idx])

        # Sort clusters for consistency
        for cluster_i in range(len(label_coreference_cluster)):
            label_coreference_cluster[cluster_i] = sorted(label_coreference_cluster[cluster_i])
        label_coreference_cluster = sorted(label_coreference_cluster, key = lambda x: x[0])



        # ====================================================================
        # STEP 5: Create Event Pair Embeddings
        # ====================================================================
        # For relation classification, we need to encode each pair of events.
        # Strategy: Concatenate [event1, event2, event1-event2, event1*event2]
        # This captures both individual events and their interaction.
        
        for event_pair_i in range(event_pairs.shape[0]):
            # Get embeddings for both events in this pair
            event_1_in_label_event = event_pairs[event_pair_i, 0]
            event_2_in_label_event = event_pairs[event_pair_i, 1]
            event_1_embedding = event_embeddings[event_1_in_label_event, :].view(1, 768)
            event_2_embedding = event_embeddings[event_2_in_label_event, :].view(1, 768)
            
            # Create interaction features
            pair_element_wise_sub = torch.sub(event_1_embedding, event_2_embedding)  # Difference
            pair_element_wise_mul = torch.mul(event_1_embedding, event_2_embedding)  # Element-wise product
            
            # Concatenate all features: [e1, e2, e1-e2, e1*e2] => 768*4 dimensions
            this_pair_embedding = torch.cat((event_1_embedding, event_2_embedding, 
                                            pair_element_wise_sub, pair_element_wise_mul), dim = 1)
            
            # Build matrix of all pair embeddings
            if event_pair_i == 0:
                event_pair_embeddings = this_pair_embedding  # Shape: [1, 768*4]
            else:
                event_pair_embeddings = torch.cat((event_pair_embeddings, this_pair_embedding), dim = 0)



        # event coreference relation task, training based on event pairs instead of based on event mention clusters

        if coreference_train_method == "event_pairs":

            coreference_raw_scores = self.coreference_head_3(self.relu(self.coreference_head_2(self.relu(self.coreference_head_1(event_pair_embeddings))))) # nrow = nrow(event_pairs), ncol = 2
            coreference_loss = self.crossentropyloss(coreference_raw_scores, label_coreference) # size = nrow(event_pairs)

            coreference_loss_weight_0 = (label_coreference == 0).int() # weight = 1 for negative examples with label_coreference = 0
            coreference_loss_0 = torch.mul(coreference_loss, coreference_loss_weight_0)

            coreference_loss_weight_1 = torch.mul(label_coreference, coreference_weight_positive) # weight = coreference_weight_positive for positive examples with label_coreference = 1
            coreference_loss_1 = torch.mul(coreference_loss, coreference_loss_weight_1)

            coreference_weighted_loss = torch.add(coreference_loss_0, coreference_loss_1)
            coreference_weighted_loss = torch.sum(coreference_weighted_loss)

            coreference_decision = torch.argmax(coreference_raw_scores, dim = 1)

            predicted_coreference_cluster = [] # use label_coreference_cluster and predicted_coreference_cluster can calculate CoNLL metric
            for event_idx in range(number_of_events):

                event1_index_in_cluster = -1  # event_idx
                for cluster_i in range(len(predicted_coreference_cluster)):
                    if event_idx in predicted_coreference_cluster[cluster_i]:
                        event1_index_in_cluster = cluster_i

                prefix_sum = int((2 * number_of_events - 1 - event_idx) * event_idx / 2)
                for j in range(0, number_of_events - 1 - event_idx):
                    if coreference_decision[prefix_sum + j] == 1:

                        event2_index_in_cluster = -1 # (event_idx + j + 1)
                        for cluster_i in range(len(predicted_coreference_cluster)):
                            if (event_idx + j + 1) in predicted_coreference_cluster[cluster_i]:
                                event2_index_in_cluster = cluster_i

                        if event1_index_in_cluster == -1 and event2_index_in_cluster == -1:
                            predicted_coreference_cluster.append([event_idx, event_idx + j + 1])
                            event1_index_in_cluster = len(predicted_coreference_cluster) - 1
                            event2_index_in_cluster = len(predicted_coreference_cluster) - 1
                        if event1_index_in_cluster != -1 and event2_index_in_cluster == -1:
                            predicted_coreference_cluster[event1_index_in_cluster].append(event_idx + j + 1)
                            event2_index_in_cluster = event1_index_in_cluster
                        if event1_index_in_cluster == -1 and event2_index_in_cluster != -1:
                            predicted_coreference_cluster[event2_index_in_cluster].append(event_idx)
                            event1_index_in_cluster = event2_index_in_cluster
                        if event1_index_in_cluster != -1 and event2_index_in_cluster != -1 and event1_index_in_cluster != event2_index_in_cluster:
                            predicted_coreference_cluster[event1_index_in_cluster].extend(predicted_coreference_cluster[event2_index_in_cluster])
                            predicted_coreference_cluster.pop(event2_index_in_cluster)
                            # recalculate event1_index_in_cluster for event_idx because one element pop out
                            for cluster_i in range(len(predicted_coreference_cluster)):
                                if event_idx in predicted_coreference_cluster[cluster_i]:
                                    event1_index_in_cluster = cluster_i
                            event2_index_in_cluster = event1_index_in_cluster

                if event1_index_in_cluster == -1: # singleton
                    predicted_coreference_cluster.append([event_idx])

            for cluster_i in range(len(predicted_coreference_cluster)):
                predicted_coreference_cluster[cluster_i] = sorted(predicted_coreference_cluster[cluster_i])
            predicted_coreference_cluster = sorted(predicted_coreference_cluster, key=lambda x: x[0])



        # event temporal relation task

        temporal_raw_scores = self.temporal_head_3(self.relu(self.temporal_head_2(self.relu(self.temporal_head_1(event_pair_embeddings))))) # nrow = nrow(event_pairs), ncol = 4
        temporal_loss = self.crossentropyloss(temporal_raw_scores, label_temporal) # size = nrow(event_pairs)

        temporal_loss_weight_0 = (label_temporal == 0).int() # weight = 1 for negative examples with label_temporal = 0
        temporal_loss_0 = torch.mul(temporal_loss, temporal_loss_weight_0)

        temporal_loss_weight_1 = torch.mul((label_temporal == 1).int(), temporal_weight_before)
        temporal_loss_1 = torch.mul(temporal_loss, temporal_loss_weight_1)

        temporal_loss_weight_2 = torch.mul((label_temporal == 2).int(), temporal_weight_after)
        temporal_loss_2 = torch.mul(temporal_loss, temporal_loss_weight_2)

        temporal_loss_weight_3 = torch.mul((label_temporal == 3).int(), temporal_weight_overlap)
        temporal_loss_3 = torch.mul(temporal_loss, temporal_loss_weight_3)

        temporal_weighted_loss = torch.add(torch.add(torch.add(temporal_loss_0, temporal_loss_1), temporal_loss_2), temporal_loss_3)
        temporal_weighted_loss = torch.sum(temporal_weighted_loss)


        # event causal relation task

        causal_raw_scores = self.causal_head_3(self.relu(self.causal_head_2(self.relu(self.causal_head_1(event_pair_embeddings))))) # nrow = nrow(event_pairs), ncol = 3
        causal_loss = self.crossentropyloss(causal_raw_scores, label_causal) # size = nrow(event_pairs)

        causal_loss_weight_0 = (label_causal == 0).int() # weight = 1 for negative examples with label_causal = 0
        causal_loss_0 = torch.mul(causal_loss, causal_loss_weight_0)

        causal_loss_weight_1 = torch.mul((label_causal == 1).int(), causal_weight_cause)
        causal_loss_1 = torch.mul(causal_loss, causal_loss_weight_1)

        causal_loss_weight_2 = torch.mul((label_causal == 2).int(), causal_weight_caused)
        causal_loss_2 = torch.mul(causal_loss, causal_loss_weight_2)

        causal_weighted_loss = torch.add(torch.add(causal_loss_0, causal_loss_1), causal_loss_2)
        causal_weighted_loss = torch.sum(causal_weighted_loss)


        # event subevent relation task

        subevent_raw_scores = self.subevent_head_3(self.relu(self.subevent_head_2(self.relu(self.subevent_head_1(event_pair_embeddings))))) # nrow = nrow(event_pairs), ncol = 3
        subevent_loss = self.crossentropyloss(subevent_raw_scores, label_subevent) # size = nrow(event_pairs)

        subevent_loss_weight_0 = (label_subevent == 0).int() # weight = 1 for negative examples with label_subevent = 0
        subevent_loss_0 = torch.mul(subevent_loss, subevent_loss_weight_0)

        subevent_loss_weight_1 = torch.mul((label_subevent == 1).int(), subevent_weight_contain)
        subevent_loss_1 = torch.mul(subevent_loss, subevent_loss_weight_1)

        subevent_loss_weight_2 = torch.mul((label_subevent == 2).int(), subevent_weight_contained)
        subevent_loss_2 = torch.mul(subevent_loss, subevent_loss_weight_2)

        subevent_weighted_loss = torch.add(torch.add(subevent_loss_0, subevent_loss_1), subevent_loss_2)
        subevent_weighted_loss = torch.sum(subevent_weighted_loss)


        # event coreference relation task, training based on event mention clusters instead of based on event pairs

        if coreference_train_method == "event_cluster":

            event_pairs_sorted_by_event2 = torch.stack(sorted(event_pairs, key = lambda x: (x[1], x[0])))

            for event_pair_i in range(event_pairs_sorted_by_event2.shape[0]):
                if event_pair_i == 0:
                    event_1_in_label_event = event_pairs_sorted_by_event2[event_pair_i, 0]
                    event_2_in_label_event = event_pairs_sorted_by_event2[event_pair_i, 1]
                    event_1_embedding = event_embeddings[event_1_in_label_event, :].view(1, 768)
                    event_2_embedding = event_embeddings[event_2_in_label_event, :].view(1, 768)
                    pair_element_wise_sub = torch.sub(event_1_embedding, event_2_embedding)
                    pair_element_wise_mul = torch.mul(event_1_embedding, event_2_embedding)
                    event_pair_embeddings = torch.cat((event_1_embedding, event_2_embedding, pair_element_wise_sub, pair_element_wise_mul), dim = 1) # 1 * (768 * 4)
                else:
                    event_1_in_label_event = event_pairs_sorted_by_event2[event_pair_i, 0]
                    event_2_in_label_event = event_pairs_sorted_by_event2[event_pair_i, 1]
                    event_1_embedding = event_embeddings[event_1_in_label_event, :].view(1, 768)
                    event_2_embedding = event_embeddings[event_2_in_label_event, :].view(1, 768)
                    pair_element_wise_sub = torch.sub(event_1_embedding, event_2_embedding)
                    pair_element_wise_mul = torch.mul(event_1_embedding, event_2_embedding)
                    this_event_pair_embedding = torch.cat((event_1_embedding, event_2_embedding, pair_element_wise_sub, pair_element_wise_mul), dim = 1)
                    event_pair_embeddings = torch.cat((event_pair_embeddings, this_event_pair_embedding), dim = 0)

            coreference_raw_scores = self.coreference_head_3(self.relu(self.coreference_head_2(self.relu(self.coreference_head_1(event_pair_embeddings))))) # nrow = nrow(event_pairs), ncol = 2
            coreference_raw_scores = coreference_raw_scores[:,0]
            split_scores = [to_cuda(torch.tensor([]))] + list(torch.split(coreference_raw_scores, [i for i in range(number_of_events) if i], dim=0))  # first event has no valid antecedent
            epsilon = to_var(torch.tensor([0.]))  # dummy score default to 0.0
            with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]  # dummy index default to same index as itself
            coreference_probs = [F.softmax(tensor, dim=0) for tensor in with_epsilon]
            coreference_probs = pad_and_stack(coreference_probs, value = -100.0) # use label_coreference_cluster and coreference_probs can calculate CoNLL metric

            filled_labels = fill_expand(label_coreference_cluster)
            filled_labels = to_cuda(filled_labels)
            eps = 1e-8
            prob_sum = torch.sum(torch.clamp(torch.mul(coreference_probs, filled_labels), eps, 1-eps), dim=1)
            coreference_cluster_loss = torch.sum(torch.log(prob_sum)) * -1 # torch.mean(torch.log(prob_sum)) * -1



        if coreference_train_method == "event_pairs":
            return event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
                   subevent_weighted_loss, subevent_raw_scores, coreference_weighted_loss, coreference_raw_scores, predicted_coreference_cluster, label_coreference_cluster

        if coreference_train_method == "event_cluster":
            return event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
                   subevent_weighted_loss, subevent_raw_scores, coreference_cluster_loss, coreference_probs, label_coreference_cluster






# ============================================================================
# EVALUATION METRICS AND FUNCTIONS
# ============================================================================
# This section implements various coreference evaluation metrics:
# - MUC: Measures link-based overlap between predicted and gold clusters
# - B-CUBED: Measures precision/recall for each mention
# - CEAFE: Entity-aligned F-measure using Hungarian algorithm
# - BLANC: Considers both coreference and non-coreference links
# CoNLL score is the average of these four metrics.
# ============================================================================

def get_event2cluster(clusters):
    """Map each event ID to its cluster tuple."""
    event2cluster = {}
    for cluster in clusters:
        for eid in cluster:
            event2cluster[eid] = tuple(cluster)
    return event2cluster

def get_clusters(event2cluster):
    """Extract unique clusters from event-to-cluster mapping."""
    clusters = list(set(event2cluster.values()))
    return clusters

def get_predicted_clusters(prob):
    """
    Convert antecedent probabilities to clusters.
    Used for cluster-based coreference method.
    
    Args:
        prob: Tensor of antecedent probabilities [num_events, max_antecedents+1]
        
    Returns:
        predicted_clusters: List of clusters
        idx_to_clusters: Mapping from event index to cluster
    """
    # Get most likely antecedent for each event
    predicted_antecedents = torch.argmax(prob, dim=-1).cpu().numpy().tolist()
    
    # Initialize: each event in its own cluster
    idx_to_clusters = {}
    for i in range(len(predicted_antecedents)):
        idx_to_clusters[i] = set([i])

    # Merge clusters based on predicted antecedents
    for i, predicted_index in enumerate(predicted_antecedents):
        if predicted_index >= i:
            # No antecedent or invalid (event is its own antecedent)
            assert predicted_index == i
            continue
        else:
            # Merge with antecedent's cluster
            union_cluster = idx_to_clusters[predicted_index] | idx_to_clusters[i]
            for j in union_cluster:
                idx_to_clusters[j] = union_cluster
                
    # Convert to sorted tuples
    idx_to_clusters = {i: tuple(sorted(idx_to_clusters[i])) for i in idx_to_clusters}
    predicted_clusters = get_clusters(idx_to_clusters)
    return predicted_clusters, idx_to_clusters


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem

def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p

def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))

def ceafe(clusters, gold_clusters):
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    row_id, col_id = linear_sum_assignment(-scores)
    similarity = sum(scores[row_id, col_id])
    return similarity, len(clusters), similarity, len(gold_clusters)

def blanc(mention_to_cluster, mention_to_gold):
    rc = 0
    wc = 0
    rn = 0
    wn = 0
    assert len(mention_to_cluster) == len(mention_to_gold)
    mentions = list(mention_to_cluster.keys())
    for i in range(len(mentions)):
        for j in range(i + 1, len(mentions)):
            if mention_to_cluster[mentions[i]] == mention_to_cluster[mentions[j]]:
                if mention_to_gold[mentions[i]] == mention_to_gold[mentions[j]]:
                    rc += 1
                else:
                    wc += 1
            else:
                if mention_to_gold[mentions[i]] == mention_to_gold[mentions[j]]:
                    wn += 1
                else:
                    rn += 1
    return rc, wc, rn, wn


class MUC:
    def __init__(self, beta = 1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = muc
        self.beta = beta
        self.rc = 0
        self.wc = 0
        self.rn = 0
        self.wn = 0

    def update(self, gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster):
        pn, pd = self.metric(pred_cluster, gold_event2cluster)
        rn, rd = self.metric(gold_cluster, pred_event2cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class B_CUBED:
    def __init__(self, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = b_cubed
        self.beta = beta
        self.rc = 0
        self.wc = 0
        self.rn = 0
        self.wn = 0

    def update(self, gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster):
        pn, pd = self.metric(pred_cluster, gold_event2cluster)
        rn, rd = self.metric(gold_cluster, pred_event2cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class CEAFE:
    def __init__(self, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = ceafe
        self.beta = beta
        self.rc = 0
        self.wc = 0
        self.rn = 0
        self.wn = 0

    def update(self, gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster):
        pn, pd, rn, rd = self.metric(pred_cluster, gold_cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class BLANC:
    def __init__(self, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = blanc
        self.beta = beta
        self.rc = 0
        self.wc = 0
        self.rn = 0
        self.wn = 0

    def update(self, gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster):
        rc, wc, rn, wn = self.metric(pred_event2cluster, gold_event2cluster)
        self.rc += rc
        self.wc += wc
        self.rn += rn
        self.wn += wn

    def get_f1(self):
        return (f1(self.rc, self.rc+self.wc, self.rc, self.rc+self.wn, beta=self.beta) + f1(self.rn, self.rn+self.wn, self.rn, self.rn+self.wc, beta=self.beta)) / 2

    def get_recall(self):
        return (self.rc/(self.rc+self.wn+1e-6) + self.rn/(self.rn+self.wc+1e-6)) / 2

    def get_precision(self):
        return (self.rc/(self.rc+self.wc+1e-6) + self.rn/(self.rn+self.wn+1e-6)) / 2

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()



def evaluate(event_relation_graph, eval_dataloader, verbose):

    event_relation_graph.eval()

    if coreference_train_method == "event_pairs":

        muc_evaluator = MUC()
        bcubed_evaluator = B_CUBED()
        ceafe_evaluator = CEAFE()
        blanc_evaluator = BLANC()

        for step, batch in enumerate(eval_dataloader):

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label_event = batch['label_event']
            event_pairs= batch['event_pairs']
            label_coreference = batch['label_coreference']
            label_temporal = batch['label_temporal']
            label_causal = batch['label_causal']
            label_subevent = batch['label_subevent']

            label_event = label_event[0, :, :]
            number_of_events = torch.sum((label_event[:, 2] == 1).int())
            if number_of_events == 1:
                continue
            event_pairs = event_pairs[0, :, :]
            label_coreference = label_coreference[0, :]
            label_temporal = label_temporal[0, :]
            label_causal = label_causal[0, :]
            label_subevent = label_subevent[0, :]

            input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent = \
                input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), \
                label_coreference.to(device), label_temporal.to(device), label_causal.to(device), label_subevent.to(device)

            with torch.no_grad():
                event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
                subevent_weighted_loss, subevent_raw_scores, coreference_weighted_loss, coreference_raw_scores, predicted_coreference_cluster, label_coreference_cluster = \
                    event_relation_graph(input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent, coreference_train_method)


            gold_event2cluster = get_event2cluster(label_coreference_cluster)
            gold_cluster = label_coreference_cluster
            pred_event2cluster = get_event2cluster(predicted_coreference_cluster)
            pred_cluster = predicted_coreference_cluster

            muc_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            bcubed_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            ceafe_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            blanc_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)


            decision_event = torch.argmax(event_raw_scores, dim = 1).view(event_raw_scores.shape[0], 1) # batch_size * 1
            true_label_event = label_event[:,2].view(event_raw_scores.shape[0], 1)

            if step == 0:
                decision_event_onetest = decision_event
                true_label_event_onetest = true_label_event
            else:
                decision_event_onetest = torch.cat((decision_event_onetest, decision_event), dim=0)
                true_label_event_onetest = torch.cat((true_label_event_onetest, true_label_event), dim=0)


            decision_temporal = torch.argmax(temporal_raw_scores, dim = 1).view(temporal_raw_scores.shape[0], 1) # batch_size * 1
            true_label_temporal = label_temporal.view(temporal_raw_scores.shape[0], 1)

            if step == 0:
                decision_temporal_onetest = decision_temporal
                true_label_temporal_onetest = true_label_temporal
            else:
                decision_temporal_onetest = torch.cat((decision_temporal_onetest, decision_temporal), dim=0)
                true_label_temporal_onetest = torch.cat((true_label_temporal_onetest, true_label_temporal), dim=0)


            decision_causal = torch.argmax(causal_raw_scores, dim = 1).view(causal_raw_scores.shape[0], 1) # batch_size * 1
            true_label_causal = label_causal.view(causal_raw_scores.shape[0], 1)

            if step == 0:
                decision_causal_onetest = decision_causal
                true_label_causal_onetest = true_label_causal
            else:
                decision_causal_onetest = torch.cat((decision_causal_onetest, decision_causal), dim=0)
                true_label_causal_onetest = torch.cat((true_label_causal_onetest, true_label_causal), dim=0)


            decision_subevent = torch.argmax(subevent_raw_scores, dim = 1).view(subevent_raw_scores.shape[0], 1) # batch_size * 1
            true_label_subevent = label_subevent.view(subevent_raw_scores.shape[0], 1)

            if step == 0:
                decision_subevent_onetest = decision_subevent
                true_label_subevent_onetest = true_label_subevent
            else:
                decision_subevent_onetest = torch.cat((decision_subevent_onetest, decision_subevent), dim=0)
                true_label_subevent_onetest = torch.cat((true_label_subevent_onetest, true_label_subevent), dim=0)


            decision_coreference = torch.argmax(coreference_raw_scores, dim = 1).view(coreference_raw_scores.shape[0], 1) # batch_size * 1
            true_label_coreference = label_coreference.view(coreference_raw_scores.shape[0], 1)

            if step == 0:
                decision_coreference_onetest = decision_coreference
                true_label_coreference_onetest = true_label_coreference
            else:
                decision_coreference_onetest = torch.cat((decision_coreference_onetest, decision_coreference), dim=0)
                true_label_coreference_onetest = torch.cat((true_label_coreference_onetest, true_label_coreference), dim=0)



        decision_event_onetest = decision_event_onetest.to('cpu').numpy()
        true_label_event_onetest = true_label_event_onetest.to('cpu').numpy()

        if verbose:
            print("======== Event Identification Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average=None)[:3])

        macro_F_event = precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average='macro')[2]


        decision_temporal_onetest = decision_temporal_onetest.to('cpu').numpy()
        true_label_temporal_onetest = true_label_temporal_onetest.to('cpu').numpy()

        if verbose:
            print("======== Temporal Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average=None)[:3])

        macro_F_temporal = precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average='macro')[2]


        decision_causal_onetest = decision_causal_onetest.to('cpu').numpy()
        true_label_causal_onetest = true_label_causal_onetest.to('cpu').numpy()

        if verbose:
            print("======== Causal Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average=None)[:3])

        macro_F_causal = precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average='macro')[2]


        decision_subevent_onetest = decision_subevent_onetest.to('cpu').numpy()
        true_label_subevent_onetest = true_label_subevent_onetest.to('cpu').numpy()

        if verbose:
            print("======== Subevent Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average=None)[:3])

        macro_F_subevent = precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average='macro')[2]


        decision_coreference_onetest = decision_coreference_onetest.to('cpu').numpy()
        true_label_coreference_onetest = true_label_coreference_onetest.to('cpu').numpy()

        if verbose:
            print("======== Coreference Relation Task - event pairs ========")
            print("Macro: ", precision_recall_fscore_support(true_label_coreference_onetest, decision_coreference_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_coreference_onetest, decision_coreference_onetest, average=None)[:3])

        macro_F_coreference = precision_recall_fscore_support(true_label_coreference_onetest, decision_coreference_onetest, average='macro')[2]


        muc_precision, muc_recall, muc_F = muc_evaluator.get_prf()
        bcubed_precision, bcubed_recall, bcubed_F = bcubed_evaluator.get_prf()
        ceafe_precision, ceafe_recall, ceafe_F = ceafe_evaluator.get_prf()
        blanc_precision, blanc_recall, blanc_F = blanc_evaluator.get_prf()
        conll_F_coreference = (muc_F + bcubed_F + ceafe_F + blanc_F) / 4

        if verbose:
            print("======== Coreference Relation Task - event mention clusters ========")
            print('MUC precision: {:.3f}, MUC recall: {:.3f}, MUC F: {:.3f}'.format(muc_precision, muc_recall, muc_F))
            print('BCUBED precision: {:.3f}, BCUBED recall: {:.3f}, BCUBED F: {:.3f}'.format(bcubed_precision, bcubed_recall, bcubed_F))
            print('CEAFE precision: {:.3f}, CEAFE recall: {:.3f}, CEAFE F: {:.3f}'.format(ceafe_precision, ceafe_recall, ceafe_F))
            print('BLANC precision: {:.3f}, BLANC recall: {:.3f}, BLANC F: {:.3f}'.format(blanc_precision, blanc_recall, blanc_F))


        macro_F_graph = (macro_F_event + macro_F_temporal + macro_F_causal + macro_F_subevent + conll_F_coreference) / 5

        if verbose:
            print('macro_F_graph: {:.3f}'.format(macro_F_graph))


        return macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph




    if coreference_train_method == "event_cluster":

        muc_evaluator = MUC()
        bcubed_evaluator = B_CUBED()
        ceafe_evaluator = CEAFE()
        blanc_evaluator = BLANC()

        for step, batch in enumerate(eval_dataloader):

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label_event = batch['label_event']
            event_pairs = batch['event_pairs']
            label_coreference = batch['label_coreference']
            label_temporal = batch['label_temporal']
            label_causal = batch['label_causal']
            label_subevent = batch['label_subevent']

            label_event = label_event[0, :, :]
            number_of_events = torch.sum((label_event[:, 2] == 1).int())
            if number_of_events == 1:
                continue
            event_pairs = event_pairs[0, :, :]
            label_coreference = label_coreference[0, :]
            label_temporal = label_temporal[0, :]
            label_causal = label_causal[0, :]
            label_subevent = label_subevent[0, :]

            input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent = \
                input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), \
                label_coreference.to(device), label_temporal.to(device), label_causal.to(device), label_subevent.to(device)

            with torch.no_grad():
                event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
                subevent_weighted_loss, subevent_raw_scores, coreference_cluster_loss, coreference_probs, label_coreference_cluster = \
                    event_relation_graph(input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent, coreference_train_method)


            pred_cluster, pred_event2cluster = get_predicted_clusters(coreference_probs)
            gold_event2cluster = get_event2cluster(label_coreference_cluster)
            gold_cluster = label_coreference_cluster

            muc_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            bcubed_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            ceafe_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            blanc_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)


            decision_event = torch.argmax(event_raw_scores, dim = 1).view(event_raw_scores.shape[0], 1) # batch_size * 1
            true_label_event = label_event[:,2].view(event_raw_scores.shape[0], 1)

            if step == 0:
                decision_event_onetest = decision_event
                true_label_event_onetest = true_label_event
            else:
                decision_event_onetest = torch.cat((decision_event_onetest, decision_event), dim=0)
                true_label_event_onetest = torch.cat((true_label_event_onetest, true_label_event), dim=0)


            decision_temporal = torch.argmax(temporal_raw_scores, dim = 1).view(temporal_raw_scores.shape[0], 1) # batch_size * 1
            true_label_temporal = label_temporal.view(temporal_raw_scores.shape[0], 1)

            if step == 0:
                decision_temporal_onetest = decision_temporal
                true_label_temporal_onetest = true_label_temporal
            else:
                decision_temporal_onetest = torch.cat((decision_temporal_onetest, decision_temporal), dim=0)
                true_label_temporal_onetest = torch.cat((true_label_temporal_onetest, true_label_temporal), dim=0)


            decision_causal = torch.argmax(causal_raw_scores, dim = 1).view(causal_raw_scores.shape[0], 1) # batch_size * 1
            true_label_causal = label_causal.view(causal_raw_scores.shape[0], 1)

            if step == 0:
                decision_causal_onetest = decision_causal
                true_label_causal_onetest = true_label_causal
            else:
                decision_causal_onetest = torch.cat((decision_causal_onetest, decision_causal), dim=0)
                true_label_causal_onetest = torch.cat((true_label_causal_onetest, true_label_causal), dim=0)


            decision_subevent = torch.argmax(subevent_raw_scores, dim = 1).view(subevent_raw_scores.shape[0], 1) # batch_size * 1
            true_label_subevent = label_subevent.view(subevent_raw_scores.shape[0], 1)

            if step == 0:
                decision_subevent_onetest = decision_subevent
                true_label_subevent_onetest = true_label_subevent
            else:
                decision_subevent_onetest = torch.cat((decision_subevent_onetest, decision_subevent), dim=0)
                true_label_subevent_onetest = torch.cat((true_label_subevent_onetest, true_label_subevent), dim=0)



        decision_event_onetest = decision_event_onetest.to('cpu').numpy()
        true_label_event_onetest = true_label_event_onetest.to('cpu').numpy()

        if verbose:
            print("======== Event Identification Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average=None)[:3])

        macro_F_event = precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average='macro')[2]


        decision_temporal_onetest = decision_temporal_onetest.to('cpu').numpy()
        true_label_temporal_onetest = true_label_temporal_onetest.to('cpu').numpy()

        if verbose:
            print("======== Temporal Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average=None)[:3])

        macro_F_temporal = precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average='macro')[2]


        decision_causal_onetest = decision_causal_onetest.to('cpu').numpy()
        true_label_causal_onetest = true_label_causal_onetest.to('cpu').numpy()

        if verbose:
            print("======== Causal Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average=None)[:3])

        macro_F_causal = precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average='macro')[2]


        decision_subevent_onetest = decision_subevent_onetest.to('cpu').numpy()
        true_label_subevent_onetest = true_label_subevent_onetest.to('cpu').numpy()

        if verbose:
            print("======== Subevent Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average=None)[:3])

        macro_F_subevent = precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average='macro')[2]


        muc_precision, muc_recall, muc_F = muc_evaluator.get_prf()
        bcubed_precision, bcubed_recall, bcubed_F = bcubed_evaluator.get_prf()
        ceafe_precision, ceafe_recall, ceafe_F = ceafe_evaluator.get_prf()
        blanc_precision, blanc_recall, blanc_F = blanc_evaluator.get_prf()
        conll_F_coreference = (muc_F + bcubed_F + ceafe_F + blanc_F) / 4

        if verbose:
            print("======== Coreference Relation Task - event mention clusters ========")
            print('MUC precision: {:.3f}, MUC recall: {:.3f}, MUC F: {:.3f}'.format(muc_precision, muc_recall, muc_F))
            print('BCUBED precision: {:.3f}, BCUBED recall: {:.3f}, BCUBED F: {:.3f}'.format(bcubed_precision, bcubed_recall, bcubed_F))
            print('CEAFE precision: {:.3f}, CEAFE recall: {:.3f}, CEAFE F: {:.3f}'.format(ceafe_precision, ceafe_recall, ceafe_F))
            print('BLANC precision: {:.3f}, BLANC recall: {:.3f}, BLANC F: {:.3f}'.format(blanc_precision, blanc_recall, blanc_F))


        macro_F_graph = (macro_F_event + macro_F_temporal + macro_F_causal + macro_F_subevent + conll_F_coreference) / 5

        if verbose:
            print('macro_F_graph: {:.3f}'.format(macro_F_graph))


        return macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph







# ============================================================================
# TRAINING SCRIPT
# ============================================================================
# This section sets up and executes the training loop for the event relation
# graph model. It includes:
# 1. Random seed setting for reproducibility
# 2. Model initialization and optimizer configuration
# 3. Dataset loading (20% subset for faster training)
# 4. Training loop with periodic evaluation
# 5. Model checkpointing for best performance on each task
# ============================================================================

import time
import datetime

def format_time(elapsed):
    """
    Convert elapsed time in seconds to readable format (HH:MM:SS).
    
    Args:
        elapsed (float): Elapsed time in seconds
        
    Returns:
        str: Formatted time string
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Suppress warnings for cleaner output
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()



# ============================================================================
# Set Random Seeds for Reproducibility
# ============================================================================
# Setting seeds ensures that the model training is reproducible
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# ============================================================================
# Initialize Model and Move to GPU
# ============================================================================
event_relation_graph = Event_Relation_Graph()
event_relation_graph.cuda()


# ============================================================================
# Configure Optimizer with Layer-Specific Learning Rates
# ============================================================================
# Different learning rates for pre-trained (Longformer) vs task-specific layers
# Weight decay only applied to weights, not biases or layer normalization

param_all = list(event_relation_graph.named_parameters())
optimizer_grouped_parameters = [
    # Longformer weights with weight decay
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('longformer' in n))],
     'lr': longformer_lr, 'weight_decay': longformer_weight_decay},
    # Task-specific weights with weight decay
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and (not 'longformer' in n))],
     'lr': non_longformer_lr, 'weight_decay': non_longformer_weight_decay},
    # Longformer biases/norms without weight decay
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('longformer' in n))],
     'lr': longformer_lr, 'weight_decay': 0.0},
    # Task-specific biases/norms without weight decay
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and (not 'longformer' in n))],
     'lr': non_longformer_lr, 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)


# ============================================================================
# Load Training and Validation Datasets
# ============================================================================
# Using 20% of data for faster training (remove sampling for full training)

train_path = "./MAVEN_ERE/train/"
train_file_names = os.listdir(train_path)
random.seed(42)
train_file_names = random.sample(train_file_names, int(len(train_file_names) * 0.2))  # 20% sample
train_file_paths = []
train_file_paths = create_file_path(train_path, train_file_names)

dev_path = "./MAVEN_ERE/dev/"
dev_file_names = os.listdir(dev_path)
random.seed(42)
dev_file_names = random.sample(dev_file_names, int(len(dev_file_names) * 0.2))  # 20% sample
dev_file_paths = []
dev_file_paths = create_file_path(dev_path, dev_file_names)

print(f"Training on {len(train_file_paths)} files (20% of dataset)")
print(f"Validating on {len(dev_file_paths)} files (20% of dataset)")

# COMMENTED OUT: test set loading (no test set in this dataset)
# test_path = "./MAVEN_ERE/test/"
# test_file_names = os.listdir(test_path)
# test_file_paths = []
# test_file_paths = create_file_path(test_path, test_file_names)

# Create PyTorch datasets
train_dataset = custom_dataset(train_file_paths)
dev_dataset = custom_dataset(dev_file_paths)
# COMMENTED OUT: test dataset
# test_dataset = custom_dataset(test_file_paths)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# COMMENTED OUT: test dataloader
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ============================================================================
# Configure Learning Rate Scheduler
# ============================================================================
# Linear warmup followed by linear decay

num_train_steps = num_epochs * len(train_dataloader)
warmup_steps = int(warmup_proportion * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)


# ============================================================================
# Initialize Best Score Tracking
# ============================================================================
# Track best performance on validation set for each metric
# Models are saved when a new best score is achieved

best_macro_F_event = 0             # Best F1 for event identification
best_macro_F_coreference = 0       # Best F1 for coreference (pair-based)
best_conll_F_coreference = 0       # Best CoNLL F1 (cluster-based, avg of 4 metrics)
best_macro_F_temporal = 0          # Best F1 for temporal relations
best_macro_F_causal = 0            # Best F1 for causal relations
best_macro_F_subevent = 0          # Best F1 for subevent relations
best_macro_F_graph = 0             # Best average F1 across all tasks



# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
# Training strategy:
# 1. Process each batch (one article at a time due to batch_size=1)
# 2. Compute forward pass and all task losses
# 3. Backpropagate with gradient clipping for stability
# 4. Periodically evaluate on validation set (check_times evaluations total)
# 5. Save model checkpoints when validation performance improves
# 6. Skip articles with only 1 event (no pairs to form)
# ============================================================================

for epoch_i in range(num_epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
    print('Training...')

    t0 = time.time()
    
    # Track cumulative losses for reporting
    total_event_loss = 0
    total_coreference_loss = 0
    total_temporal_loss = 0
    total_causal_loss = 0
    total_subevent_loss = 0
    num_batch = 0

    for step, batch in enumerate(train_dataloader):

        # ====================================================================
        # Periodic Evaluation on Validation Set
        # ====================================================================
        if step % ((len(train_dataloader) * num_epochs) // check_times) == 0:

            elapsed = format_time(time.time() - t0)

            # Print training loss averages
            if num_batch != 0:
                avg_event_loss = total_event_loss / num_batch
                avg_coreference_loss = total_coreference_loss / num_batch
                avg_temporal_loss = total_temporal_loss / num_batch
                avg_causal_loss = total_causal_loss / num_batch
                avg_subevent_loss = total_subevent_loss / num_batch

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Event Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_event_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Coreference Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_coreference_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Temporal Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_temporal_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Causal Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_causal_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Subevent Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_subevent_loss))

            else:
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Reset loss tracking
            total_event_loss = 0
            total_coreference_loss = 0
            total_temporal_loss = 0
            total_causal_loss = 0
            total_subevent_loss = 0
            num_batch = 0

            # Run evaluation on development set
            if coreference_train_method == "event_pairs":
                macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
                    evaluate(event_relation_graph, dev_dataloader, verbose = 1)

                # Save if best coreference pair F1
                if macro_F_coreference > best_macro_F_coreference:
                    torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_coreference.ckpt')
                    best_macro_F_coreference = macro_F_coreference

            if coreference_train_method == "event_cluster":
                macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
                    evaluate(event_relation_graph, dev_dataloader, verbose = 1)

            # Save model checkpoints for each task if performance improved
            if macro_F_event > best_macro_F_event:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_event.ckpt')
                best_macro_F_event = macro_F_event
            if macro_F_temporal > best_macro_F_temporal:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_temporal.ckpt')
                best_macro_F_temporal = macro_F_temporal
            if macro_F_causal > best_macro_F_causal:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_causal.ckpt')
                best_macro_F_causal = macro_F_causal
            if macro_F_subevent > best_macro_F_subevent:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_subevent.ckpt')
                best_macro_F_subevent = macro_F_subevent
            if conll_F_coreference > best_conll_F_coreference:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_conll_F_coreference.ckpt')
                best_conll_F_coreference = conll_F_coreference
            if macro_F_graph > best_macro_F_graph:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_graph.ckpt')
                best_macro_F_graph = macro_F_graph



        # ====================================================================
        # Training Step
        # ====================================================================

        event_relation_graph.train()

        # Extract batch data
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label_event = batch['label_event']
        event_pairs = batch['event_pairs']
        label_coreference = batch['label_coreference']
        label_temporal = batch['label_temporal']
        label_causal = batch['label_causal']
        label_subevent = batch['label_subevent']

        # Remove batch dimension (batch_size is 1)
        label_event = label_event[0, :, :]
        
        # Skip if only 1 event (no pairs can be formed)
        number_of_events = torch.sum((label_event[:, 2] == 1).int())
        if number_of_events == 1:
            continue
            
        event_pairs = event_pairs[0, :, :]
        label_coreference = label_coreference[0, :]
        label_temporal = label_temporal[0, :]
        label_causal = label_causal[0, :]
        label_subevent = label_subevent[0, :]

        # Move to GPU
        input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent = \
            input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), \
            label_coreference.to(device), label_temporal.to(device), label_causal.to(device), label_subevent.to(device)

        # Zero gradients
        optimizer.zero_grad()


        if coreference_train_method == "event_pairs":
            # Forward pass
            event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
            subevent_weighted_loss, subevent_raw_scores, coreference_weighted_loss, coreference_raw_scores, predicted_coreference_cluster, label_coreference_cluster = \
                event_relation_graph(input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent, coreference_train_method)

            # Track losses
            total_event_loss += event_weighted_loss.item()
            total_coreference_loss += coreference_weighted_loss.item()
            total_temporal_loss += temporal_weighted_loss.item()
            total_causal_loss += causal_weighted_loss.item()
            total_subevent_loss += subevent_weighted_loss.item()
            num_batch += 1

            # Backward pass (separate for each loss, retain_graph needed for shared parameters)
            event_weighted_loss.backward(retain_graph = True)
            coreference_weighted_loss.backward(retain_graph = True)
            temporal_weighted_loss.backward(retain_graph = True)
            causal_weighted_loss.backward(retain_graph = True)
            subevent_weighted_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(event_relation_graph.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()

        if coreference_train_method == "event_cluster":
            # Forward pass (cluster-based coreference)
            event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
            subevent_weighted_loss, subevent_raw_scores, coreference_cluster_loss, coreference_probs, label_coreference_cluster = \
                event_relation_graph(input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent, coreference_train_method)

            # Track losses
            total_event_loss += event_weighted_loss.item()
            total_coreference_loss += coreference_cluster_loss.item()
            total_temporal_loss += temporal_weighted_loss.item()
            total_causal_loss += causal_weighted_loss.item()
            total_subevent_loss += subevent_weighted_loss.item()
            num_batch += 1

            # Backward pass
            event_weighted_loss.backward(retain_graph = True)
            coreference_cluster_loss.backward(retain_graph = True)
            temporal_weighted_loss.backward(retain_graph = True)
            causal_weighted_loss.backward(retain_graph = True)
            subevent_weighted_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(event_relation_graph.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()



    elapsed = format_time(time.time() - t0)

    if num_batch != 0:
        avg_event_loss = total_event_loss / num_batch
        avg_coreference_loss = total_coreference_loss / num_batch
        avg_temporal_loss = total_temporal_loss / num_batch
        avg_causal_loss = total_causal_loss / num_batch
        avg_subevent_loss = total_subevent_loss / num_batch

        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Event Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_event_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Coreference Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_coreference_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Temporal Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_temporal_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Causal Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_causal_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Subevent Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_subevent_loss))

    else:
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

    total_event_loss = 0
    total_coreference_loss = 0
    total_temporal_loss = 0
    total_causal_loss = 0
    total_subevent_loss = 0
    num_batch = 0

    # evaluate on dev set

    if coreference_train_method == "event_pairs":
        macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
            evaluate(event_relation_graph, dev_dataloader, verbose = 1)

        if macro_F_coreference > best_macro_F_coreference:
            torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_coreference.ckpt')
            best_macro_F_coreference = macro_F_coreference

    if coreference_train_method == "event_cluster":
        macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
            evaluate(event_relation_graph, dev_dataloader, verbose = 1)

    if macro_F_event > best_macro_F_event:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_event.ckpt')
        best_macro_F_event = macro_F_event
    if macro_F_temporal > best_macro_F_temporal:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_temporal.ckpt')
        best_macro_F_temporal = macro_F_temporal
    if macro_F_causal > best_macro_F_causal:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_causal.ckpt')
        best_macro_F_causal = macro_F_causal
    if macro_F_subevent > best_macro_F_subevent:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_subevent.ckpt')
        best_macro_F_subevent = macro_F_subevent
    if conll_F_coreference > best_conll_F_coreference:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_conll_F_coreference.ckpt')
        best_conll_F_coreference = conll_F_coreference
    if macro_F_graph > best_macro_F_graph:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_graph.ckpt')
        best_macro_F_graph = macro_F_graph


print("")
print("Training complete!")
print("Best scores on dev set:")
print(f"  best_macro_F_event: {best_macro_F_event:.3f}")
print(f"  best_macro_F_temporal: {best_macro_F_temporal:.3f}")
print(f"  best_macro_F_causal: {best_macro_F_causal:.3f}")
print(f"  best_macro_F_subevent: {best_macro_F_subevent:.3f}")
print(f"  best_conll_F_coreference: {best_conll_F_coreference:.3f}")
if coreference_train_method == "event_pairs":
    print(f"  best_macro_F_coreference: {best_macro_F_coreference:.3f}")
print(f"  best_macro_F_graph: {best_macro_F_graph:.3f}")


# COMMENTED OUT: ALL TEST SET EVALUATION
# # test
# 
# print("Testing...")
# 
# print("best_macro_F_graph on dev is: {:}".format(best_macro_F_graph))
# 
# event_relation_graph = Event_Relation_Graph()
# event_relation_graph.cuda()
# event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_graph.ckpt', map_location=device))
# 
# if coreference_train_method == "event_pairs":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# if coreference_train_method == "event_cluster":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# 
# print("best_macro_F_event on dev is: {:}".format(best_macro_F_event))
# 
# event_relation_graph = Event_Relation_Graph()
# event_relation_graph.cuda()
# event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_event.ckpt', map_location=device))
# 
# if coreference_train_method == "event_pairs":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# if coreference_train_method == "event_cluster":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# 
# print("best_macro_F_temporal on dev is: {:}".format(best_macro_F_temporal))
# 
# event_relation_graph = Event_Relation_Graph()
# event_relation_graph.cuda()
# event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_temporal.ckpt', map_location=device))
# 
# if coreference_train_method == "event_pairs":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# if coreference_train_method == "event_cluster":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# 
# print("best_macro_F_causal on dev is: {:}".format(best_macro_F_causal))
# 
# event_relation_graph = Event_Relation_Graph()
# event_relation_graph.cuda()
# event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_causal.ckpt', map_location=device))
# 
# if coreference_train_method == "event_pairs":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# if coreference_train_method == "event_cluster":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# 
# print("best_macro_F_subevent on dev is: {:}".format(best_macro_F_subevent))
# 
# event_relation_graph = Event_Relation_Graph()
# event_relation_graph.cuda()
# event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_subevent.ckpt', map_location=device))
# 
# if coreference_train_method == "event_pairs":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# if coreference_train_method == "event_cluster":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# 
# print("best_conll_F_coreference on dev is: {:}".format(best_conll_F_coreference))
# 
# event_relation_graph = Event_Relation_Graph()
# event_relation_graph.cuda()
# event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_conll_F_coreference.ckpt', map_location=device))
# 
# if coreference_train_method == "event_pairs":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# if coreference_train_method == "event_cluster":
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)
# 
# 
# if coreference_train_method == "event_pairs":
# 
#     print("best_macro_F_coreference on dev is: {:}".format(best_macro_F_coreference))
# 
#     event_relation_graph = Event_Relation_Graph()
#     event_relation_graph.cuda()
#     event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_coreference.ckpt', map_location=device))
# 
#     macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
#         evaluate(event_relation_graph, test_dataloader, verbose=1)




# stop here