"""
Enhanced Event Relation Extraction - Implicit Relations Improvement
This script imports and extends your existing 2_train_event_extractors.py
without modifying the original code.

Usage:
    python 3_train_enhanced_implicit_relations.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import everything from your original training script
# Assuming the file is named '2_train_event_extractors.py'
import importlib.util

def import_original_code(file_path='2_train_event_extractors.py'):
    """Import all components from original training script"""
    spec = importlib.util.spec_from_file_location("original_trainer", file_path)
    original_module = importlib.util.module_from_spec(spec)
    sys.modules["original_trainer"] = original_module
    spec.loader.exec_module(original_module)
    return original_module

# Import original code
print("Importing original training code...")
orig = import_original_code()

# Now we have access to all original components:
# orig.Event_Relation_Graph, orig.custom_dataset, orig.tokenizer, etc.

# ============= ENHANCEMENT CONFIGURATIONS =============
USE_CONTRASTIVE_LEARNING = True
USE_DATA_AUGMENTATION = True
CONTRASTIVE_WEIGHT = 0.3

# ============= DATA AUGMENTATION MODULE =============
class ImplicitRelationAugmenter:
    """Augments training data by removing explicit discourse markers"""
    
    def __init__(self):
        self.explicit_markers = {
            'temporal': ['before', 'after', 'since', 'until', 'when', 'while', 
                         'following', 'preceding', 'then', 'next', 'previously'],
            'causal': ['because', 'therefore', 'thus', 'hence', 'consequently',
                       'as a result', 'due to', 'caused by', 'leads to', 'so'],
            'subevent': ['including', 'such as', 'consists of', 'comprises',
                         'contains', 'involves', 'composed of']
        }
    
    def augment_files(self, file_paths, output_dir="./MAVEN_ERE/train_augmented/"):
        """Create augmented versions of training files"""
        os.makedirs(output_dir, exist_ok=True)
        augmented_paths = []
        
        print(f"Augmenting {len(file_paths)} files for implicit relations...")
        
        for file_path in tqdm(file_paths):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                text = ' '.join(data['tokens_list']).lower()
                
                # Find all markers present
                found_markers = []
                for relation_type, markers in self.explicit_markers.items():
                    for marker in markers:
                        if marker in text:
                            found_markers.append(marker)
                
                if found_markers:
                    # Remove one random marker
                    marker_to_remove = random.choice(found_markers)
                    augmented_tokens = self._remove_marker(
                        data['tokens_list'], 
                        marker_to_remove
                    )
                    
                    # Create augmented data
                    augmented_data = data.copy()
                    augmented_data['tokens_list'] = augmented_tokens
                    
                    # Save
                    base_name = os.path.basename(file_path).replace('.json', '')
                    aug_path = os.path.join(output_dir, f"{base_name}_aug.json")
                    
                    with open(aug_path, 'w') as f:
                        json.dump(augmented_data, f)
                    
                    augmented_paths.append(aug_path)
            
            except Exception as e:
                print(f"Error augmenting {file_path}: {e}")
                continue
        
        print(f"Created {len(augmented_paths)} augmented files")
        return augmented_paths
    
    def _remove_marker(self, tokens, marker):
        """Remove marker from token list"""
        marker_words = marker.split()
        result = []
        i = 0
        
        while i < len(tokens):
            if i + len(marker_words) <= len(tokens):
                window = ' '.join(tokens[i:i+len(marker_words)]).lower()
                if window == marker:
                    i += len(marker_words)
                    continue
            result.append(tokens[i])
            i += 1
        
        return result


# ============= ENHANCED MODEL =============
class EnhancedEventRelationGraph(nn.Module):
    """
    Wraps original Event_Relation_Graph with implicit relation enhancements
    """
    
    def __init__(self, original_model):
        super().__init__()
        
        # Keep the original model
        self.original_model = original_model
        
        # Add contrastive learning projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(768 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Add implicit relation enhancement layers
        self.implicit_temporal = nn.Sequential(
            nn.Linear(768 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        self.implicit_causal = nn.Sequential(
            nn.Linear(768 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        self.implicit_subevent = nn.Sequential(
            nn.Linear(768 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        # Fusion layers (combine original + implicit features)
        self.temporal_fusion = nn.Linear(256 * 2, 4)
        self.causal_fusion = nn.Linear(256 * 2, 3)
        self.subevent_fusion = nn.Linear(256 * 2, 3)
    
    def compute_contrastive_loss(self, event_pair_embeddings, labels, temperature=0.07):
        """Contrastive learning for implicit relations"""
        # Project to contrastive space
        projected = F.normalize(self.contrastive_proj(event_pair_embeddings), dim=-1)
        
        # Similarity matrix
        similarity = torch.mm(projected, projected.t()) / temperature
        
        # Positive pairs (same label)
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        positive_mask.fill_diagonal_(0)
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity)
        positive_sum = (exp_sim * positive_mask).sum(dim=1)
        negative_sum = exp_sim.sum(dim=1) - torch.diag(exp_sim)
        
        loss = -torch.log(positive_sum / (negative_sum + 1e-8) + 1e-8)
        return loss.mean()
    
    def forward(self, input_ids, attention_mask, label_event, event_pairs, 
                label_coreference, label_temporal, label_causal, label_subevent, 
                coreference_train_method, use_enhanced=True):
        """
        Forward pass with optional implicit relation enhancements
        """
        # Get base model outputs (this calls your original model)
        if orig.coreference_train_method == "event_pairs":
            outputs = self.original_model(
                input_ids, attention_mask, label_event, event_pairs,
                label_coreference, label_temporal, label_causal, label_subevent,
                coreference_train_method
            )
            (event_weighted_loss, event_raw_scores, temporal_weighted_loss, 
             temporal_raw_scores, causal_weighted_loss, causal_raw_scores,
             subevent_weighted_loss, subevent_raw_scores, coreference_weighted_loss, 
             coreference_raw_scores, predicted_coreference_cluster, 
             label_coreference_cluster) = outputs
        else:
            outputs = self.original_model(
                input_ids, attention_mask, label_event, event_pairs,
                label_coreference, label_temporal, label_causal, label_subevent,
                coreference_train_method
            )
            (event_weighted_loss, event_raw_scores, temporal_weighted_loss, 
             temporal_raw_scores, causal_weighted_loss, causal_raw_scores,
             subevent_weighted_loss, subevent_raw_scores, coreference_cluster_loss, 
             coreference_probs, label_coreference_cluster) = outputs
        
        if not use_enhanced:
            return outputs
        
        # ===== ENHANCEMENT: Get event pair embeddings for implicit learning =====
        # We need to re-compute event pair embeddings from the model
        # Extract from original model's intermediate representations
        
        # Get token embeddings
        with torch.no_grad():
            token_embeddings = self.original_model.token_embedding(input_ids, attention_mask)
            token_embeddings = token_embeddings.view(1, token_embeddings.shape[0], token_embeddings.shape[1])
            h0 = torch.zeros(2, 1, 384).cuda()
            c0 = torch.zeros(2, 1, 384).cuda()
            token_embeddings, (_, _) = self.original_model.bilstm(token_embeddings, (h0, c0))
            token_embeddings = token_embeddings[0, :, :]
        
        # Get event embeddings
        event_embeddings = []
        for token_i in range(label_event.shape[0]):
            start_in_input_ids = label_event[token_i, 0]
            end_in_input_ids = label_event[token_i, 1]
            event_embeddings.append(
                torch.mean(token_embeddings[start_in_input_ids:end_in_input_ids, :], dim=0)
            )
        event_embeddings = torch.stack(event_embeddings)
        
        # Create event pair embeddings
        event_pair_embeddings = []
        for event_pair_i in range(event_pairs.shape[0]):
            event_1_in_label_event = event_pairs[event_pair_i, 0]
            event_2_in_label_event = event_pairs[event_pair_i, 1]
            event_1_embedding = event_embeddings[event_1_in_label_event, :]
            event_2_embedding = event_embeddings[event_2_in_label_event, :]
            
            pair_embed = torch.cat([
                event_1_embedding,
                event_2_embedding,
                torch.sub(event_1_embedding, event_2_embedding),
                torch.mul(event_1_embedding, event_2_embedding)
            ])
            event_pair_embeddings.append(pair_embed)
        
        event_pair_embeddings = torch.stack(event_pair_embeddings)
        
        # ===== Compute enhanced implicit relation features =====
        implicit_temp_feat = self.implicit_temporal(event_pair_embeddings)
        implicit_causal_feat = self.implicit_causal(event_pair_embeddings)
        implicit_subevent_feat = self.implicit_subevent(event_pair_embeddings)
        
        # ===== Compute contrastive losses =====
        contrastive_loss = 0.0
        if self.training and USE_CONTRASTIVE_LEARNING:
            contrastive_loss += self.compute_contrastive_loss(event_pair_embeddings, label_temporal)
            contrastive_loss += self.compute_contrastive_loss(event_pair_embeddings, label_causal)
            contrastive_loss += self.compute_contrastive_loss(event_pair_embeddings, label_subevent)
            contrastive_loss /= 3
        
        # Return enhanced outputs
        if orig.coreference_train_method == "event_pairs":
            return (event_weighted_loss, event_raw_scores, 
                    temporal_weighted_loss, temporal_raw_scores,
                    causal_weighted_loss, causal_raw_scores,
                    subevent_weighted_loss, subevent_raw_scores,
                    coreference_weighted_loss, coreference_raw_scores,
                    predicted_coreference_cluster, label_coreference_cluster,
                    contrastive_loss)  # Added contrastive loss
        else:
            return (event_weighted_loss, event_raw_scores,
                    temporal_weighted_loss, temporal_raw_scores,
                    causal_weighted_loss, causal_raw_scores,
                    subevent_weighted_loss, subevent_raw_scores,
                    coreference_cluster_loss, coreference_probs,
                    label_coreference_cluster,
                    contrastive_loss)  # Added contrastive loss


# ============= MAIN TRAINING FUNCTION =============
def train_enhanced():
    """Main training function with enhancements"""
    
    print("="*80)
    print("ENHANCED EVENT RELATION TRAINING - IMPLICIT RELATIONS IMPROVEMENT")
    print("="*80)
    
    # Set random seeds (same as original)
    seed_val = 42
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    # ===== STEP 1: Data Augmentation =====
    if USE_DATA_AUGMENTATION:
        augmenter = ImplicitRelationAugmenter()
        
        # Load original training files
        train_path = "./MAVEN_ERE/train/"
        train_file_names = os.listdir(train_path)
        random.seed(42)
        train_file_names = random.sample(train_file_names, int(len(train_file_names) * 0.2))
        train_file_paths = orig.create_file_path(train_path, train_file_names)
        
        # Augment
        augmented_paths = augmenter.augment_files(train_file_paths)
        all_train_paths = train_file_paths + augmented_paths
        
        print(f"\nOriginal training files: {len(train_file_paths)}")
        print(f"Augmented files: {len(augmented_paths)}")
        print(f"Total training files: {len(all_train_paths)}")
    else:
        # Use original training data
        train_path = "./MAVEN_ERE/train/"
        train_file_names = os.listdir(train_path)
        random.seed(42)
        train_file_names = random.sample(train_file_names, int(len(train_file_names) * 0.2))
        all_train_paths = orig.create_file_path(train_path, train_file_names)
    
    # Load dev data (same as original)
    dev_path = "./MAVEN_ERE/dev/"
    dev_file_names = os.listdir(dev_path)
    random.seed(42)
    dev_file_names = random.sample(dev_file_names, int(len(dev_file_names) * 0.2))
    dev_file_paths = orig.create_file_path(dev_path, dev_file_names)
    
    # Create datasets
    train_dataset = orig.custom_dataset(all_train_paths)
    dev_dataset = orig.custom_dataset(dev_file_paths)
    
    train_dataloader = orig.DataLoader(train_dataset, batch_size=1, shuffle=True)
    dev_dataloader = orig.DataLoader(dev_dataset, batch_size=1, shuffle=False)
    
    # ===== STEP 2: Initialize Enhanced Model =====
    print("\nInitializing enhanced model...")
    
    # Create original model
    original_model = orig.Event_Relation_Graph()
    
    # Wrap with enhancements
    enhanced_model = EnhancedEventRelationGraph(original_model)
    enhanced_model.cuda()
    
    # Setup optimizer (same as original)
    param_all = list(enhanced_model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in orig.no_decay)) and ('longformer' in n))],
         'lr': orig.longformer_lr, 'weight_decay': orig.longformer_weight_decay},
        {'params': [p for n, p in param_all if ((not any(nd in n for nd in orig.no_decay)) and (not 'longformer' in n))],
         'lr': orig.non_longformer_lr, 'weight_decay': orig.non_longformer_weight_decay},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in orig.no_decay)) and ('longformer' in n))],
         'lr': orig.longformer_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in param_all if ((any(nd in n for nd in orig.no_decay)) and (not 'longformer' in n))],
         'lr': orig.non_longformer_lr, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
    
    num_train_steps = orig.num_epochs * len(train_dataloader)
    warmup_steps = int(orig.warmup_proportion * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    
    # ===== STEP 3: Training Loop =====
    print("\nStarting training...")
    best_macro_F_graph = 0
    
    for epoch_i in range(orig.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch_i + 1} / {orig.num_epochs}")
        print(f"{'='*80}")
        
        enhanced_model.train()
        t0 = time.time()
        total_loss = 0
        total_contrastive_loss = 0
        num_batch = 0
        
        for step, batch in enumerate(train_dataloader):
            # Prepare inputs
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label_event = batch['label_event'][0, :, :]
            event_pairs = batch['event_pairs'][0, :, :]
            label_coreference = batch['label_coreference'][0, :]
            label_temporal = batch['label_temporal'][0, :]
            label_causal = batch['label_causal'][0, :]
            label_subevent = batch['label_subevent'][0, :]
            
            number_of_events = torch.sum((label_event[:, 2] == 1).int())
            if number_of_events == 1:
                continue
            
            # Move to device
            input_ids = input_ids.to(orig.device)
            attention_mask = attention_mask.to(orig.device)
            label_event = label_event.to(orig.device)
            event_pairs = event_pairs.to(orig.device)
            label_coreference = label_coreference.to(orig.device)
            label_temporal = label_temporal.to(orig.device)
            label_causal = label_causal.to(orig.device)
            label_subevent = label_subevent.to(orig.device)
            
            optimizer.zero_grad()
            
            # Forward pass with enhancements
            outputs = enhanced_model(
                input_ids, attention_mask, label_event, event_pairs,
                label_coreference, label_temporal, label_causal, label_subevent,
                orig.coreference_train_method,
                use_enhanced=True
            )
            
            # Unpack outputs (includes contrastive loss)
            if orig.coreference_train_method == "event_pairs":
                (event_weighted_loss, _, temporal_weighted_loss, _,
                 causal_weighted_loss, _, subevent_weighted_loss, _,
                 coreference_weighted_loss, _, _, _, contrastive_loss) = outputs
                
                # Combined loss
                combined_loss = (event_weighted_loss + temporal_weighted_loss +
                                 causal_weighted_loss + subevent_weighted_loss +
                                 coreference_weighted_loss +
                                 CONTRASTIVE_WEIGHT * contrastive_loss)
            else:
                (event_weighted_loss, _, temporal_weighted_loss, _,
                 causal_weighted_loss, _, subevent_weighted_loss, _,
                 coreference_cluster_loss, _, _, contrastive_loss) = outputs
                
                combined_loss = (event_weighted_loss + temporal_weighted_loss +
                                 causal_weighted_loss + subevent_weighted_loss +
                                 coreference_cluster_loss +
                                 CONTRASTIVE_WEIGHT * contrastive_loss)
            
            # Backward pass
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(enhanced_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += combined_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            num_batch += 1
            
            # Periodic evaluation
            if step % ((len(train_dataloader) * orig.num_epochs) // orig.check_times) == 0 and step > 0:
                elapsed = orig.format_time(time.time() - t0)
                avg_loss = total_loss / num_batch
                avg_cont_loss = total_contrastive_loss / num_batch
                
                print(f"  Batch {step:>5,} / {len(train_dataloader):>5,}  |  Elapsed: {elapsed}")
                print(f"  Avg Loss: {avg_loss:.4f}  |  Contrastive Loss: {avg_cont_loss:.4f}")
                
                # Evaluate on dev set
                print("\n  Evaluating on dev set...")
                if orig.coreference_train_method == "event_pairs":
                    results = orig.evaluate(enhanced_model.original_model, dev_dataloader, verbose=1)
                    macro_F_graph = results[6]
                else:
                    results = orig.evaluate(enhanced_model.original_model, dev_dataloader, verbose=1)
                    macro_F_graph = results[5]
                
                # Save best model
                if macro_F_graph > best_macro_F_graph:
                    best_macro_F_graph = macro_F_graph
                    os.makedirs('./saved_models/enhanced_event_relation_graph/', exist_ok=True)
                    torch.save(enhanced_model.state_dict(), 
                               './saved_models/enhanced_event_relation_graph/best_model.ckpt')
                    print(f"  ✓ New best model saved! F1: {best_macro_F_graph:.4f}")
                
                enhanced_model.train()
                total_loss = 0
                total_contrastive_loss = 0
                num_batch = 0
        
        # End of epoch evaluation
        print(f"\nEpoch {epoch_i + 1} completed. Final evaluation...")
        if orig.coreference_train_method == "event_pairs":
            results = orig.evaluate(enhanced_model.original_model, dev_dataloader, verbose=1)
            macro_F_graph = results[6]
        else:
            results = orig.evaluate(enhanced_model.original_model, dev_dataloader, verbose=1)
            macro_F_graph = results[5]
        
        if macro_F_graph > best_macro_F_graph:
            best_macro_F_graph = macro_F_graph
            torch.save(enhanced_model.state_dict(),
                       './saved_models/enhanced_event_relation_graph/best_model.ckpt')
            print(f"✓ New best model! F1: {best_macro_F_graph:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print(f"Best macro F1 on dev set: {best_macro_F_graph:.4f}")
    print("="*80)
    
    return enhanced_model, best_macro_F_graph


# ============= RUN TRAINING =============
if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMPLICIT RELATION ENHANCEMENT FOR EVENT RELATION EXTRACTION")
    print("="*80)
    print(f"\nConfigurations:")
    print(f"  - Contrastive Learning: {USE_CONTRASTIVE_LEARNING}")
    print(f"  - Data Augmentation: {USE_DATA_AUGMENTATION}")
    print(f"  - Contrastive Weight: {CONTRASTIVE_WEIGHT}")
    print(f"\nThis will train an enhanced version of your model.")
    print(f"Your original code remains unchanged.\n")
    
    # Run training
    model, best_f1 = train_enhanced()
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best model saved to: ./saved_models/enhanced_event_relation_graph/best_model.ckpt")
    print(f"✓ Best F1 score: {best_f1:.4f}")