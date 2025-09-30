import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def split_maven_dataset(num_members=10):
    """Split processed MAVEN-ERE files among 6 group members"""
    
    # Collect all processed files
    all_files = []
    base_dirs = ['./MAVEN_ERE/train/', './MAVEN_ERE/dev/']
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.json')]
            all_files.extend(files)
    
    print(f"Found {len(all_files)} total files to split")
    
    if len(all_files) == 0:
        print("ERROR: No files found!")
        print("Make sure you ran mavenere_event_relation_label.py first")
        return False
    
    # Create output directories
    os.makedirs('./group_split', exist_ok=True)
    
    # Calculate splits - ensure even distribution
    chunk_size = len(all_files) // num_members
    
    total_events_estimate = 0
    
    for member_id in range(1, num_members + 1):
        print(f"\nProcessing Member {member_id}...")
        
        # Calculate file range for this member
        start_idx = (member_id - 1) * chunk_size
        if member_id == num_members:  # Last member gets remainder
            end_idx = len(all_files)
        else:
            end_idx = member_id * chunk_size
            
        member_files = all_files[start_idx:end_idx]
        
        # Create member directory
        member_dir = f'./group_split/member_{member_id}'
        os.makedirs(member_dir, exist_ok=True)
        
        # Copy files and count events
        member_events = 0
        for file_path in tqdm(member_files, desc=f"Processing files for member {member_id}"):
            filename = os.path.basename(file_path)
            dest_path = f"{member_dir}/{filename}"
            shutil.copy2(file_path, dest_path)
            
            # Count events in this file
            try:
                with open(file_path, 'r') as f:
                    article_json = json.load(f)
                events_in_file = sum(1 for event in article_json['event_label'] if event['event_label'] == 1)
                member_events += events_in_file
            except:
                member_events += 5  # Rough estimate if file reading fails
        
        total_events_estimate += member_events
        
        # Create file list for member
        with open(f"{member_dir}/file_list.txt", 'w') as f:
            for file_path in member_files:
                f.write(f"{os.path.basename(file_path)}\n")
        
        # Calculate API requirements
        estimated_requests = (member_events + 5) // 6  # 6 events per batch
        
        # Create config file for member
        config = {
            "member_id": member_id,
            "total_files": len(member_files),
            "file_range": f"{start_idx}-{end_idx-1}",
            "estimated_events": member_events,
            "estimated_requests": estimated_requests,
            "daily_limit_usage": f"{estimated_requests}/1000",
            "estimated_runtime_minutes": int(estimated_requests / 15 * 60)  # 15 RPM
        }
        
        with open(f"{member_dir}/member_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Member {member_id}:")
        print(f"  Files: {len(member_files)}")
        print(f"  Events: ~{member_events}")
        print(f"  Requests: ~{estimated_requests}")
        print(f"  Runtime: ~{config['estimated_runtime_minutes']} minutes")
        print(f"  Directory: {member_dir}")

    print(f"\n{'='*50}")
    print(f"DATASET SPLIT COMPLETE")
    print(f"{'='*50}")
    print(f"Total files: {len(all_files)}")
    print(f"Total events: ~{total_events_estimate}")
    print(f"Total requests needed: ~{total_events_estimate // 6}")
    print(f"Requests per member: ~{total_events_estimate // 6 // num_members}")
    print(f"\nEach member should run:")
    print(f"python 2_classify_events_member.py --member_id [1-6] --api_key [their_key]")
    
    return True

if __name__ == "__main__":
    split_maven_dataset()