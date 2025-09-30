import json
import os
from collections import defaultdict
import time

def combine_all_member_results():
    """Combine classification results from all 10 members"""
    
    print(f"{'='*60}")
    print(f"COMBINING RESULTS FROM ALL 10 MEMBERS")
    print(f"{'='*60}")
    
    # Check all members completed
    missing_members = []
    completed_members = []
    
    for member_id in range(1, 11):
        results_file = f"./group_split/member_{member_id}/member_{member_id}_classifications.json"
        if not os.path.exists(results_file):
            missing_members.append(member_id)
        else:
            completed_members.append(member_id)
    
    if missing_members:
        print(f"WARNING: Missing results from members: {missing_members}")
        print(f"Completed members: {completed_members}")
        
        if len(completed_members) == 0:
            print("No member results found! Make sure members have run classification first.")
            return False
        
        print(f"Proceeding with {len(completed_members)} completed members...")
    
    # Combine all results
    combined_classifications = {}
    member_summaries = {}
    total_stats = {
        'files': 0,
        'events': 0,
        'factual': 0,
        'interpretive': 0,
        'requests': 0,
        'runtime_minutes': 0
    }
    
    for member_id in completed_members:
        results_file = f"./group_split/member_{member_id}/member_{member_id}_classifications.json"
        
        print(f"Loading Member {member_id} results...")
        
        with open(results_file, 'r') as f:
            member_data = json.load(f)
        
        # Store member summary
        member_summaries[f"member_{member_id}"] = member_data['summary']
        
        # Add classifications to combined dict
        combined_classifications.update(member_data['classifications'])
        
        # Update totals
        summary = member_data['summary']
        total_stats['files'] += summary['total_files']
        total_stats['events'] += summary['total_events']
        total_stats['factual'] += summary['factual_events']
        total_stats['interpretive'] += summary['interpretive_events']
        total_stats['requests'] += summary['requests_used']
        
        print(f"  Member {member_id}: {summary['total_files']} files, "
              f"{summary['total_events']} events, "
              f"{summary['requests_used']}/1000 requests")
    
    # Create final combined output
    final_output = {
        'combination_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_members_completed': len(completed_members),
        'missing_members': missing_members,
        'combined_summary': {
            'total_files': total_stats['files'],
            'total_events': total_stats['events'],
            'total_factual': total_stats['factual'],
            'total_interpretive': total_stats['interpretive'],
            'total_requests_used': total_stats['requests'],
            'factual_percentage': round(total_stats['factual']/total_stats['events']*100, 1) if total_stats['events'] > 0 else 0,
            'interpretive_percentage': round(total_stats['interpretive']/total_stats['events']*100, 1) if total_stats['events'] > 0 else 0
        },
        'member_summaries': member_summaries,
        'all_classifications': combined_classifications
    }
    
    # Save master results file
    output_file = './master_event_classifications.json'
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMBINATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total files processed: {total_stats['files']:,}")
    print(f"Total events classified: {total_stats['events']:,}")
    print(f"Factual events: {total_stats['factual']:,} ({final_output['combined_summary']['factual_percentage']}%)")
    print(f"Interpretive events: {total_stats['interpretive']:,} ({final_output['combined_summary']['interpretive_percentage']}%)")
    print(f"Total API requests: {total_stats['requests']:,}")
    print(f"Average requests per member: {total_stats['requests']//len(completed_members)}")
    print(f"Master results saved to: {output_file}")
    
    # Create training-ready format
    print(f"\nCreating training-ready format...")
    success = create_training_format(combined_classifications)
    
    return success

def create_training_format(classifications):
    """Create simplified format for training code integration"""
    
    training_format = {}
    successfully_processed = 0
    error_files = 0
    
    for filename, data in classifications.items():
        if 'error' in data:
            error_files += 1
            continue
            
        # Map event indices to classifications
        event_classifications = {}
        
        if 'events' in data and 'classifications' in data:
            for i, event in enumerate(data['events']):
                if i < len(data['classifications']):
                    event_classifications[event['index']] = data['classifications'][i]
        
        training_format[filename] = event_classifications
        successfully_processed += 1
    
    # Save training format
    training_file = './event_fact_interp_labels.json'
    with open(training_file, 'w') as f:
        json.dump(training_format, f, indent=2)
    
    print(f"Training format created:")
    print(f"  Successfully processed: {successfully_processed} files")
    print(f"  Error files skipped: {error_files}")
    print(f"  Training labels saved to: {training_file}")
    
    return True

def verify_completeness():
    """Verify all files were processed and show statistics"""
    
    print(f"\nVERIFYING COMPLETENESS...")
    
    # Count original files from Step 1
    original_files = []
    for base_dir in ['./MAVEN_ERE/train/', './MAVEN_ERE/dev/', './MAVEN_ERE/test/']:
        if os.path.exists(base_dir):
            files = [f for f in os.listdir(base_dir) if f.endswith('.json')]
            original_files.extend(files)
    
    # Load combined results
    if not os.path.exists('./master_event_classifications.json'):
        print("Master results file not found!")
        return False
        
    with open('./master_event_classifications.json', 'r') as f:
        combined_data = json.load(f)
    
    processed_files = set(combined_data['all_classifications'].keys())
    original_files_set = set(original_files)
    
    missing = original_files_set - processed_files
    extra = processed_files - original_files_set
    
    print(f"Original files (Step 1 output): {len(original_files):,}")
    print(f"Processed files (Member results): {len(processed_files):,}")
    
    if missing:
        print(f"Missing files: {len(missing)}")
        if len(missing) <= 10:
            print(f"Missing: {list(missing)}")
        else:
            print(f"First 10 missing: {list(missing)[:10]}")
    
    if extra:
        print(f"Extra files: {len(extra)}")
        if len(extra) <= 10:
            print(f"Extra: {list(extra)}")
    
    if not missing and not extra:
        print("Perfect match! All files processed successfully")
        return True
    elif len(missing) < len(original_files) * 0.05:  # Less than 5% missing
        print(f"Good coverage: {len(processed_files)/len(original_files)*100:.1f}% of files processed")
        return True
    else:
        print(f"Significant coverage gap: {len(missing)} files missing")
        return False

def generate_final_report():
    """Generate comprehensive final report"""
    
    with open('./master_event_classifications.json', 'r') as f:
        data = json.load(f)
    
    summary = data['combined_summary']
    
    report = f"""
{'='*80}
MAVEN-ERE EVENT CLASSIFICATION PROJECT - FINAL REPORT
{'='*80}
Completion Date: {data['combination_timestamp']}
Model Used: Gemini 2.5 Flash-Lite
Team Size: {data['total_members_completed']} members

DATASET STATISTICS:
- Total files processed: {summary['total_files']:,}
- Total events classified: {summary['total_events']:,}

CLASSIFICATION RESULTS:
- Factual events: {summary['total_factual']:,} ({summary['factual_percentage']}%)
- Interpretive events: {summary['total_interpretive']:,} ({summary['interpretive_percentage']}%)

API USAGE:
- Total requests across all members: {summary['total_requests_used']:,}
- Average requests per member: {summary['total_requests_used']//data['total_members_completed']:,}/1,000
- Total cost: $0.00 (Free tier usage)

MEMBER BREAKDOWN:
"""
    
    for member_id in range(1, 11):
        if f'member_{member_id}' in data['member_summaries']:
            member_data = data['member_summaries'][f'member_{member_id}']
            report += f"Member {member_id}: {member_data['total_events']:,} events, {member_data['requests_used']} requests\n"
        else:
            report += f"Member {member_id}: Not completed\n"
    
    report += f"""
NEXT STEPS:
1. Use 'event_fact_interp_labels.json' in training pipeline
2. Modify training_event_relation_graph.py to include factual/interpretive head
3. Proceed with hierarchical dual-view architecture training

OUTPUT FILES:
- master_event_classifications.json (complete results)
- event_fact_interp_labels.json (training-ready format)
{'='*80}
"""
    
    print(report)
    
    # Save report to file
    with open('./classification_final_report.txt', 'w') as f:
        f.write(report)
    
    print(f"Final report saved to: ./classification_final_report.txt")

if __name__ == "__main__":
    print("Step 3: Combining all member results...")
    
    success = combine_all_member_results()
    
    if success:
        verify_completeness()
        generate_final_report()
        
        print(f"\n{'='*60}")
        print(f"ALL STEPS COMPLETE!")
        print(f"{'='*60}")
        print(f"Ready to proceed with Step 4: Integration with training data")
        print(f"Use ./event_fact_interp_labels.json in your training pipeline")
    else:
        print("Combination failed - check that members completed their tasks")