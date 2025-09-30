import json
import os
import shutil
from tqdm import tqdm

def integrate_classifications_with_training_data():
    """Integrate factual/interpretive classifications back into MAVEN-ERE format for training"""
    
    print(f"{'='*60}")
    print(f"INTEGRATING CLASSIFICATIONS WITH TRAINING DATA")
    print(f"{'='*60}")
    
    # Check if classifications exist
    if not os.path.exists('./event_fact_interp_labels.json'):
        print("ERROR: ./event_fact_interp_labels.json not found!")
        print("Run 3_combine_results.py first")
        return False
    
    # Load classifications
    print("Loading event classifications...")
    with open('./event_fact_interp_labels.json', 'r') as f:
        classifications = json.load(f)
    
    print(f"Loaded classifications for {len(classifications)} files")
    
    # Create output directory for enhanced training data
    output_base = './MAVEN_ERE_with_fact_interp/'
    os.makedirs(output_base, exist_ok=True)
    
    # Process train, dev, AND test splits
    total_processed = 0
    total_events = 0
    total_classified = 0
    
    for split_name, input_dir in [('train', './MAVEN_ERE/train/'), 
                                   ('dev', './MAVEN_ERE/dev/'),
                                   ('test', './MAVEN_ERE/test/')]:
        
        print(f"\nProcessing {split_name} split...")
        
        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} not found, skipping...")
            continue
        
        # Create output directory for this split
        output_dir = f"{output_base}/{split_name}/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all files in this split
        files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        
        if len(files) == 0:
            print(f"  No files found in {input_dir}, skipping...")
            continue
        
        split_processed = 0
        split_events = 0
        split_classified = 0
        
        for filename in tqdm(files, desc=f"Processing {split_name}"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Load original processed file
                with open(input_path, 'r') as f:
                    article_json = json.load(f)
                
                # Add factual/interpretive labels
                article_json, events_count, classified_count = add_fact_interp_labels(
                    article_json, classifications.get(filename, {})
                )
                
                # Save enhanced file
                with open(output_path, 'w') as f:
                    json.dump(article_json, f, indent=2)
                
                split_processed += 1
                split_events += events_count
                split_classified += classified_count
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                # Copy original file if processing fails
                shutil.copy2(input_path, output_path)
        
        print(f"{split_name} split complete:")
        print(f"  Files processed: {split_processed}")
        print(f"  Total events: {split_events}")
        print(f"  Classified events: {split_classified}")
        print(f"  Classification coverage: {split_classified/split_events*100:.1f}%" if split_events > 0 else "  No events found")
        
        total_processed += split_processed
        total_events += split_events
        total_classified += split_classified
    
    print(f"\n{'='*60}")
    print(f"INTEGRATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {total_processed}")
    print(f"Total events: {total_events}")
    print(f"Successfully classified: {total_classified}")
    print(f"Overall coverage: {total_classified/total_events*100:.1f}%" if total_events > 0 else "No events found")
    print(f"Enhanced files saved to: {output_base}")
    
    # Create PyTorch-ready label files
    create_pytorch_labels(output_base)
    
    return True

def add_fact_interp_labels(article_json, file_classifications):
    """Add factual/interpretive labels to article JSON"""
    
    events_count = 0
    classified_count = 0
    
    # Add fact_interp_label field to each event_label entry
    for i, event_label in enumerate(article_json['event_label']):
        if event_label['event_label'] == 1:  # Is an event
            events_count += 1
            
            # Add classification if available
            if str(i) in file_classifications:
                event_label['fact_interp_label'] = file_classifications[str(i)]
                classified_count += 1
            else:
                event_label['fact_interp_label'] = 0  # Default to factual
        else:
            # Non-events get -1 (not applicable)
            event_label['fact_interp_label'] = -1
    
    return article_json, events_count, classified_count

def create_pytorch_labels(base_dir):
    """Create PyTorch-compatible label files for faster training data loading"""
    
    print(f"\nCreating PyTorch-compatible label tensors...")
    
    for split_name in ['train', 'dev', 'test']:
        directory = f"{base_dir}/{split_name}/"
        
        if not os.path.exists(directory):
            print(f"Skipping {split_name} - directory not found")
            continue
        
        files = [f for f in os.listdir(directory) if f.endswith('.json')]
        
        if len(files) == 0:
            print(f"Skipping {split_name} - no files found")
            continue
            
        all_fact_interp_labels = {}
        
        for filename in tqdm(files, desc=f"Creating {split_name} tensors"):
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r') as f:
                article_json = json.load(f)
            
            # Extract factual/interpretive labels as list
            labels = []
            for event_label in article_json['event_label']:
                if 'fact_interp_label' in event_label:
                    labels.append(event_label['fact_interp_label'])
                else:
                    labels.append(-1)  # Not classified
            
            all_fact_interp_labels[filename] = labels
        
        # Save for fast loading in training
        import pickle
        output_file = f'./fact_interp_labels_{split_name}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(all_fact_interp_labels, f)
        
        print(f"PyTorch labels saved: {output_file}")

def create_modification_guide():
    """Create guide for modifying training code"""
    
    guide = """
# TRAINING CODE MODIFICATION GUIDE

## 1. Update Dataset Class in training_event_relation_graph.py

Add this to custom_dataset.__getitem__():

```python
# Load factual/interpretive labels
import pickle

# Load cached labels (faster than JSON)
if not hasattr(self, 'fact_interp_cache'):
    split_name = 'train' if 'train' in file_path else 'dev'
    with open(f'./fact_interp_labels_{split_name}.pkl', 'rb') as f:
        self.fact_interp_cache = pickle.load(f)

filename = os.path.basename(file_path)
fact_interp_labels = self.fact_interp_cache.get(filename, [])

# Convert to tensor (only for events, filter out -1)
label_fact_interp = []
for i, event_label in enumerate(article_json['event_label']):
    if event_label['event_label'] == 1:  # Is an event
        if i < len(fact_interp_labels) and fact_interp_labels[i] != -1:
            label_fact_interp.append(fact_interp_labels[i])
        else:
            label_fact_interp.append(0)  # Default factual

label_fact_interp = torch.tensor(label_fact_interp)

# Add to return dict
dict["label_fact_interp"] = label_fact_interp
```

## 2. Update Model Architecture

Add to Event_Relation_Graph.__init__():

```python
# Factual/Interpretive classification head
self.fact_interp_head_1 = nn.Linear(768, 384, bias=True)
self.fact_interp_head_2 = nn.Linear(384, 2, bias=True)
```

Add to forward() method:

```python
# Factual/Interpretive classification
fact_interp_scores = self.fact_interp_head_2(
    self.relu(self.fact_interp_head_1(event_embeddings))
)
fact_interp_loss = self.crossentropyloss(fact_interp_scores, label_fact_interp)

# Add to return statement
return (..., fact_interp_loss, fact_interp_scores)
```

## 3. Update Training Loop

Add fact_interp_loss to the backward pass:

```python
fact_interp_weighted_loss.backward(retain_graph=True)
```

## 4. Update Evaluation

Add factual/interpretive evaluation metrics to evaluate() function.
"""

    with open('./training_modification_guide.txt', 'w') as f:
        f.write(guide)
    
    print(f"Training modification guide saved to: ./training_modification_guide.txt")

if __name__ == "__main__":
    print("Step 4: Integrating classifications with training data...")
    
    success = integrate_classifications_with_training_data()
    
    if success:
        create_modification_guide()
        
        print(f"\n{'='*60}")
        print(f"INTEGRATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Files ready for training:")
        print(f"  Enhanced MAVEN-ERE: ./MAVEN_ERE_with_fact_interp/")
        print(f"  Training labels: ./event_fact_interp_labels.json")
        print(f"  PyTorch labels: ./fact_interp_labels_train.pkl, ./fact_interp_labels_dev.pkl, ./fact_interp_labels_test.pkl")
        print(f"  Modification guide: ./training_modification_guide.txt")
        print(f"\nNext: Run training_event_relation_graph_enhanced.py")
    else:
        print("Integration failed - check member completion status")