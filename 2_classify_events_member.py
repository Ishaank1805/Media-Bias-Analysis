from google import genai
import json
import os
import time
import argparse
from tqdm import tqdm
from typing import List, Dict

class GeminiEventClassifier:
    def __init__(self, api_key: str, member_id: int):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash-lite"
        self.member_id = member_id
        self.daily_requests = 0
        self.max_daily_requests = 990  # Leave buffer under 1,000 RPD
        
    def build_sota_prompt(self, events_batch):
        """SOTA prompt for maximum accuracy with Gemini 2.5 Flash-Lite"""
        prompt = f"""You are an expert linguist specializing in factual vs interpretive language analysis.

FACTUAL events are objectively verifiable occurrences:
- Physical actions: voted, signed, met, traveled, died, born
- Official proceedings: passed, approved, filed, announced, declared
- Measurable changes: increased by X%, decreased to $Y, rose, fell
- Documented statements: said, reported, stated (the act of speaking)
- Temporal markers: occurred, happened, began, ended, concluded

INTERPRETIVE events involve subjective evaluation or characterization:
- Opinion verbs: criticized, praised, condemned, celebrated, worried
- Modal/speculative: could harm, might benefit, appears to, seems like
- Evaluative characterizations: threatened, rescued, devastated, triumphed
- Predictions: will lead to, expected to cause, likely to result
- Cognitive attributions: believed, felt, thought, assumed

CRITICAL EXAMPLES:
- "announced the policy" = FACTUAL (verifiable speech act)
- "admitted the mistake" = INTERPRETIVE ("admitted" implies wrongdoing)
- "voted 52-48" = FACTUAL (recorded action)
- "criticized the decision" = INTERPRETIVE (subjective evaluation)

Events to classify:
"""
        
        for j, event in enumerate(events_batch):
            prompt += f"{j+1}. '{event['trigger']}' in \"{event['context'][:50]}...\"\n"
        
        prompt += f"\nRespond exactly: 1.FACTUAL 2.INTERPRETIVE 3.FACTUAL etc. for all {len(events_batch)} events:"
        
        return prompt
    
    def process_member_files(self, member_dir: str):
        """Process all files assigned to this member"""
        
        # Initial wait to clear any rolling rate limit window
        print(f"Member {self.member_id}: Waiting 2 minutes to ensure clean rate limit window...")
        time.sleep(120)  # Wait 2 minutes to clear rolling window
        
        # Load member configuration
        config_path = f"{member_dir}/member_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"{'='*50}")
        print(f"MEMBER {self.member_id} STARTING")
        print(f"{'='*50}")
        print(f"Files to process: {config['total_files']}")
        print(f"Estimated events: {config['estimated_events']}")
        
        # Calculate requests with batch size 10 and 8-second delays
        estimated_requests = (config['estimated_events'] + 9) // 10  # Ceiling division
        estimated_minutes = estimated_requests * 10 / 60  # 8 seconds per request
        
        print(f"Batch size: 10 events per request")
        print(f"Estimated requests: {estimated_requests}")
        print(f"Daily quota usage: {estimated_requests}/1000")
        print(f"Estimated runtime: {estimated_minutes:.1f} minutes with 8s delays")
        
        # Get list of files to process
        file_list_path = f"{member_dir}/file_list.txt"
        with open(file_list_path, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        
        all_results = {}
        processed_count = 0
        total_events_processed = 0
        
        start_time = time.time()
        
        # Process each file
        for filename in tqdm(filenames, desc=f"Member {self.member_id}"):
            file_path = f"{member_dir}/{filename}"
            
            try:
                with open(file_path, 'r') as f:
                    article_json = json.load(f)
                
                # Extract events for classification
                events = self._extract_events_from_article(article_json)
                
                if events:
                    # Fixed batch size of 10
                    labels = self._classify_article_events(events, batch_size=10)
                    all_results[filename] = {
                        'events': events,
                        'classifications': labels
                    }
                    total_events_processed += len(events)
                else:
                    all_results[filename] = {
                        'events': [],
                        'classifications': []
                    }
                
                processed_count += 1
                
                # Save progress every 50 files
                if processed_count % 50 == 0:
                    self._save_progress(all_results, member_dir, processed_count)
                    elapsed = (time.time() - start_time) / 60
                    requests_so_far = self.daily_requests
                    print(f"Member {self.member_id}: {processed_count} files, {total_events_processed} events, {elapsed:.1f}min, {requests_so_far} requests")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                all_results[filename] = {'events': [], 'classifications': [], 'error': str(e)}
        
        # Save final results
        self._save_final_results(all_results, member_dir, total_events_processed)
        
        total_time = (time.time() - start_time) / 60
        print(f"\n{'='*50}")
        print(f"MEMBER {self.member_id} COMPLETED!")
        print(f"{'='*50}")
        print(f"Files processed: {processed_count}")
        print(f"Events classified: {total_events_processed}")
        print(f"Total time: {total_time:.1f} minutes")
        print(f"API requests used: {self.daily_requests}/1000")
        
        return True
        
    def _extract_events_from_article(self, article_json):
        """Extract events that need classification from processed MAVEN-ERE format"""
        events = []
        
        for i, event_label in enumerate(article_json['event_label']):
            if event_label['event_label'] == 1:  # Is an event (not background token)
                # Get context (surrounding tokens for better classification)
                context_start = max(0, i-8)
                context_end = min(len(article_json['event_label']), i+8)
                
                context_tokens = []
                for j in range(context_start, context_end):
                    context_tokens.append(article_json['event_label'][j]['token'])
                
                context = " ".join(context_tokens)
                
                events.append({
                    'index': i,  # Original index in event_label array
                    'trigger': event_label['token'],
                    'context': context
                })
        
        return events
    
    def _classify_article_events(self, events: List[Dict], batch_size=10) -> List[int]:
        """Classify events in batches of 10 with conservative 8-second delays"""
        all_labels = []
        
        for i in range(0, len(events), batch_size):
            # Check daily request limit
            if self.daily_requests >= self.max_daily_requests:
                print(f"Member {self.member_id}: Daily request limit reached ({self.daily_requests})")
                break
            
            batch = events[i:i+batch_size]
            prompt = self.build_sota_prompt(batch)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            batch_labels = self._parse_compact_response(response.text, len(batch))
            all_labels.extend(batch_labels)
            
            self.daily_requests += 1
            
            # Conservative rate limiting: 8 seconds = 7.5 requests per minute (well under 15 RPM)
            time.sleep(10.0)
        
        print(f"Member {self.member_id}: Used {self.daily_requests}/1000 daily requests")
        return all_labels
    
    def _parse_compact_response(self, response: str, expected_count: int) -> List[int]:
        """Parse response format: 1.FACTUAL 2.INTERPRETIVE etc."""
        labels = []
        
        # Split into lines and process
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip().upper()
            # Look for patterns like "1.FACTUAL", "2.INTERPRETIVE", "1.F", "2.I"
            if 'FACTUAL' in line or '.F ' in line or line.endswith('.F'):
                labels.append(0)  # factual
            elif 'INTERPRETIVE' in line or '.I ' in line or line.endswith('.I'):
                labels.append(1)  # interpretive
        
        # Ensure correct count
        while len(labels) < expected_count:
            labels.append(0)  # Default factual for missing
        
        return labels[:expected_count]
    
    def _save_progress(self, results, member_dir, count):
        """Save progress checkpoint"""
        progress_file = f"{member_dir}/progress_checkpoint_{count}.json"
        with open(progress_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Member {self.member_id}: Progress checkpoint saved")
    
    def _save_final_results(self, results, member_dir, total_events):
        """Save final classification results"""
        output_file = f"{member_dir}/member_{self.member_id}_classifications.json"
        
        # Calculate summary statistics
        total_files = len(results)
        total_factual = 0
        total_interpretive = 0
        errors = 0
        
        for data in results.values():
            if 'error' in data:
                errors += 1
            elif 'classifications' in data:
                total_factual += data['classifications'].count(0)
                total_interpretive += data['classifications'].count(1)
        
        # Create final output
        final_output = {
            'member_id': self.member_id,
            'completion_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_files': total_files,
                'successful_files': total_files - errors,
                'error_files': errors,
                'total_events': total_events,
                'factual_events': total_factual,
                'interpretive_events': total_interpretive,
                'requests_used': self.daily_requests,
                'factual_percentage': round(total_factual/total_events*100, 1) if total_events > 0 else 0,
                'interpretive_percentage': round(total_interpretive/total_events*100, 1) if total_events > 0 else 0
            },
            'classifications': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        print(f"\nMember {self.member_id} FINAL SUMMARY:")
        print(f"  Files: {total_files} ({total_files-errors} successful)")
        print(f"  Events: {total_events}")
        print(f"  Factual: {total_factual} ({final_output['summary']['factual_percentage']}%)")
        print(f"  Interpretive: {total_interpretive} ({final_output['summary']['interpretive_percentage']}%)")
        print(f"  API requests: {self.daily_requests}/1000")
        print(f"  Results saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Classify events for group member using Gemini 2.5 Flash-Lite')
    parser.add_argument('--member_id', type=int, required=True, choices=[1,2,3,4,5,6,7,8,9,10],
                       help='Member ID (1-10)')
    parser.add_argument('--api_key', type=str, required=True,
                       help='Gemini API key for this member')
    
    args = parser.parse_args()
    
    # Validate member directory exists
    member_dir = f"./group_split/member_{args.member_id}"
    if not os.path.exists(member_dir):
        print(f"ERROR: Member directory {member_dir} not found!")
        print("Run 1_split_dataset.py first with num_members=10")
        return False
    
    # Initialize classifier
    classifier = GeminiEventClassifier(args.api_key, args.member_id)
    
    # Process files
    success = classifier.process_member_files(member_dir)
    
    if success:
        print(f"\nMember {args.member_id} classification complete!")
        print(f"Share your results file with the team for combination.")
    
    return success

if __name__ == "__main__":
    main()