import json
import os
import random
import time
from tqdm import tqdm

def generate_synthetic_member_classifications(member_id, member_dir):
    """
    Generate synthetic classification results that match the format of 
    2_classify_events_member.py output
    """
    
    print(f"{'='*50}")
    print(f"GENERATING SYNTHETIC DATA FOR MEMBER {member_id}")
    print(f"{'='*50}")
    
    # Load member configuration
    config_path = f"{member_dir}/member_config.json"
    if not os.path.exists(config_path):
        print(f"ERROR: {config_path} not found!")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Files to process: {config['total_files']}")
    print(f"Estimated events: {config['estimated_events']}")
    
    # Get list of files
    file_list_path = f"{member_dir}/file_list.txt"
    with open(file_list_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]
    
    all_results = {}
    total_events_processed = 0
    total_factual = 0
    total_interpretive = 0
    
    # Process each file
    for filename in tqdm(filenames, desc=f"Member {member_id} synthetic"):
        file_path = f"{member_dir}/{filename}"
        
        try:
            with open(file_path, 'r') as f:
                article_json = json.load(f)
            
            # Extract events
            events = []
            for i, event_label in enumerate(article_json['event_label']):
                if event_label['event_label'] == 1:  # Is an event
                    context_start = max(0, i-8)
                    context_end = min(len(article_json['event_label']), i+8)
                    
                    context_tokens = []
                    for j in range(context_start, context_end):
                        context_tokens.append(article_json['event_label'][j]['token'])
                    
                    context = " ".join(context_tokens)
                    
                    events.append({
                        'index': i,
                        'trigger': event_label['token'],
                        'context': context
                    })
            
            # Generate synthetic classifications
            # Use realistic distribution: ~70% factual, ~30% interpretive
            # with some bias based on trigger words
            classifications = []
            for event in events:
                trigger = event['trigger'].lower()
                
                # Interpretive indicators
                interpretive_words = [
                    'criticized', 'praised', 'condemned', 'celebrated', 'worried',
                    'threatened', 'rescued', 'devastated', 'triumphed', 'believed',
                    'felt', 'thought', 'claimed', 'alleged', 'suggested', 'implied'
                ]
                
                # Factual indicators
                factual_words = [
                    'voted', 'signed', 'met', 'traveled', 'died', 'born',
                    'passed', 'approved', 'filed', 'announced', 'declared',
                    'said', 'reported', 'stated', 'occurred', 'happened'
                ]
                
                # Determine classification with weighted randomness
                if any(word in trigger for word in interpretive_words):
                    # 80% chance interpretive if matches interpretive word
                    classification = 1 if random.random() < 0.8 else 0
                elif any(word in trigger for word in factual_words):
                    # 85% chance factual if matches factual word
                    classification = 0 if random.random() < 0.85 else 1
                else:
                    # Default: 70% factual, 30% interpretive
                    classification = 0 if random.random() < 0.7 else 1
                
                classifications.append(classification)
                
                if classification == 0:
                    total_factual += 1
                else:
                    total_interpretive += 1
            
            all_results[filename] = {
                'events': events,
                'classifications': classifications
            }
            
            total_events_processed += len(events)
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            all_results[filename] = {
                'events': [],
                'classifications': [],
                'error': str(e)
            }
    
    # Create final output matching the real format
    # Simulate realistic API usage (estimated events / 10 events per batch)
    simulated_requests = (total_events_processed + 9) // 10
    simulated_requests = min(simulated_requests, 990)  # Cap at 990
    
    final_output = {
        'member_id': member_id,
        'completion_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_files': len(filenames),
            'successful_files': len([f for f in all_results.values() if 'error' not in f]),
            'error_files': len([f for f in all_results.values() if 'error' in f]),
            'total_events': total_events_processed,
            'factual_events': total_factual,
            'interpretive_events': total_interpretive,
            'requests_used': simulated_requests,
            'factual_percentage': round(total_factual/total_events_processed*100, 1) if total_events_processed > 0 else 0,
            'interpretive_percentage': round(total_interpretive/total_events_processed*100, 1) if total_events_processed > 0 else 0,
            'note': 'SYNTHETIC DATA - Generated for testing purposes'
        },
        'classifications': all_results
    }
    
    # Save output
    output_file = f"{member_dir}/member_{member_id}_classifications.json"
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"MEMBER {member_id} SYNTHETIC DATA COMPLETE")
    print(f"{'='*50}")
    print(f"Files: {len(filenames)}")
    print(f"Events: {total_events_processed}")
    print(f"Factual: {total_factual} ({final_output['summary']['factual_percentage']}%)")
    print(f"Interpretive: {total_interpretive} ({final_output['summary']['interpretive_percentage']}%)")
    print(f"Simulated requests: {simulated_requests}/1000")
    print(f"Output saved: {output_file}")
    
    return True

def generate_all_members_synthetic(num_members=10):
    """Generate synthetic data for all members"""
    
    print(f"{'='*60}")
    print(f"GENERATING SYNTHETIC CLASSIFICATION DATA")
    print(f"{'='*60}")
    print(f"Purpose: Testing downstream pipeline (steps 3 & 4)")
    print(f"Members: {num_members}")
    print(f"\nNOTE: This is SYNTHETIC data for testing only!")
    print(f"Real classifications require running 2_classify_events_member.py")
    print(f"{'='*60}\n")
    
    # Verify group_split directory exists
    if not os.path.exists('./group_split/'):
        print("ERROR: ./group_split/ directory not found!")
        print("Run 1_split_dataset.py first")
        return False
    
    success_count = 0
    
    for member_id in range(1, num_members + 1):
        member_dir = f"./group_split/member_{member_id}"
        
        if not os.path.exists(member_dir):
            print(f"WARNING: {member_dir} not found, skipping...")
            continue
        
        print(f"\nProcessing Member {member_id}...")
        success = generate_synthetic_member_classifications(member_id, member_dir)
        
        if success:
            success_count += 1
        
        # Small delay to simulate processing time
        time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"SYNTHETIC DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully generated: {success_count}/{num_members} members")
    print(f"\nYou can now test:")
    print(f"  1. python 3_combine_results.py")
    print(f"  2. python 4_integrate_with_training.py")
    print(f"\nREMINDER: This is synthetic data!")
    print(f"For real classifications, run 2_classify_events_member.py with actual API keys")
    
    return success_count == num_members

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic classification data for testing'
    )
    parser.add_argument(
        '--num_members',
        type=int,
        default=10,
        help='Number of members (default: 10)'
    )
    
    args = parser.parse_args()
    
    success = generate_all_members_synthetic(args.num_members)
    
    if success:
        print(f"\nAll synthetic data generated successfully!")
        print(f"Ready to test steps 3 and 4")
    else:
        print(f"\nSome members failed - check output above")