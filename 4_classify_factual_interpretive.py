import os
import json
import time
from datetime import datetime, date
from google import genai
import pickle

class GeminiEventClassifier:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash-lite"
        self.daily_requests = 0
        self.max_daily_requests = 99000
        self.requests_this_minute = 0
        self.max_requests_per_minute = 14
        self.minute_start_time = time.time()
        self.today = date.today()
        self.request_log_file = "gemini_request_log.pkl"
        self._load_request_log()
    
    def _load_request_log(self):
        if os.path.exists(self.request_log_file):
            try:
                with open(self.request_log_file, 'rb') as f:
                    log = pickle.load(f)
                    if log['date'] == self.today:
                        self.daily_requests = log['count']
                        print(f"Loaded request log: {self.daily_requests} requests used today")
                    else:
                        print("New day - resetting request counter")
                        self.daily_requests = 0
            except:
                self.daily_requests = 0
        else:
            print("No existing request log - starting fresh")
    
    def _save_request_log(self):
        with open(self.request_log_file, 'wb') as f:
            pickle.dump({'date': self.today, 'count': self.daily_requests}, f)
    
    def _check_rate_limits(self):
        if self.daily_requests >= self.max_daily_requests:
            raise Exception(f"Daily limit reached ({self.max_daily_requests}). Resume tomorrow.")
        
        current_time = time.time()
        elapsed_time = current_time - self.minute_start_time
        
        if elapsed_time >= 60:
            self.requests_this_minute = 0
            self.minute_start_time = current_time
        elif self.requests_this_minute >= self.max_requests_per_minute:
            sleep_time = 60 - elapsed_time + 1
            print(f"    [Rate Limit] Waiting {sleep_time:.0f}s...")
            time.sleep(sleep_time)
            self.requests_this_minute = 0
            self.minute_start_time = time.time()
    
    def classify_event(self, event_token, sentence_text, max_retries=5):
        prompt = f"""You are classifying events in news articles as FACTUAL or INTERPRETIVE.

FACTUAL events are objectively verifiable occurrences:
- Physical actions: met, signed, traveled, departed, arrived, walked
- Official proceedings: voted, passed, ruled, approved, enacted, confirmed
- Measurable changes: increased by 10%, reached $5B, dropped to 8%, rose, fell
- Formal declarations: announced, published, filed, released, stated, reported
- Scheduled events: began, concluded, occurred, took place, started, ended

Examples: "The Senate voted 52-48", "The president signed the order", "Unemployment reached 8%"

INTERPRETIVE events involve subjective characterization or evaluation:
- Opinion attributions: critics claimed, experts warned, supporters argued, analysts believe
- Modal/speculative: could harm, might benefit, may lead to, would cause, appears to
- Evaluative characterizations: slammed, celebrated, threatened, rescued, devastated, praised
- Cognitive/perceptual: worried, hoped, feared, expected, anticipated, concerned
- Predictions: will cause, expected to result in, likely to affect, poised to

Examples: "Experts warned it could fail", "Critics slammed the decision", "threatened democracy"

Key distinctions:
- "announced" = FACTUAL (official declaration occurred)
- "warned" = INTERPRETIVE (implies negative judgment)
- "voted" = FACTUAL (official action with record)
- "condemned" = INTERPRETIVE (evaluative response)
- "increased to 8%" = FACTUAL (measurable)
- "soared to 8%" = INTERPRETIVE ("soared" is evaluative)

Sentence: "{sentence_text}"
Event: "{event_token['token_text']}"

Classify as FACTUAL or INTERPRETIVE. Output ONLY valid JSON:
{{"classification": "FACTUAL or INTERPRETIVE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
        
        for attempt in range(max_retries):
            try:
                self._check_rate_limits()
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                
                self.daily_requests += 1
                self.requests_this_minute += 1
                self._save_request_log()
                
                response_text = response.text.strip()
                
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                
                if 'classification' not in result or 'confidence' not in result:
                    raise ValueError("Missing required fields in response")
                if result['classification'] not in ['FACTUAL', 'INTERPRETIVE']:
                    raise ValueError(f"Invalid classification: {result['classification']}")
                if not (0.0 <= result['confidence'] <= 1.0):
                    raise ValueError(f"Invalid confidence: {result['confidence']}")
                
                return result
                
            except Exception as e:
                print(f"    [Error] Attempt {attempt + 1}/{max_retries}: {str(e)[:80]}")
                
                if attempt < max_retries - 1:
                    print(f"    [Retry] Waiting 60s...")
                    time.sleep(60)
                else:
                    raise Exception(f"Classification failed after {max_retries} attempts: {e}")


def find_sentence_containing_event(article_json, event_token):
    event_index = event_token['index_of_token']
    
    for sentence in article_json['sentences']:
        for token in sentence['tokens']:
            if token['index_of_token'] == event_index:
                return sentence['sentence_text'], sentence.get('sentence_id', 0)
    
    return "Sentence not found", -1


def classify_events_in_article(article_json, classifier, stats):
    if 'event_tokens' not in article_json or len(article_json['event_tokens']) == 0:
        return article_json
    
    num_events = len(article_json['event_tokens'])
    
    for i, event in enumerate(article_json['event_tokens']):
        sentence_text, sent_id = find_sentence_containing_event(article_json, event)
        
        result = classifier.classify_event(event, sentence_text)
        
        event['fi_classification'] = result['classification']
        event['fi_confidence'] = float(result['confidence'])
        event['fi_reasoning'] = result.get('reasoning', '')
        
        stats['total_events'] += 1
        if result['classification'] == 'FACTUAL':
            stats['factual_events'] += 1
        else:
            stats['interpretive_events'] += 1
        
        fact_pct = 100*stats['factual_events']/stats['total_events']
        interp_pct = 100*stats['interpretive_events']/stats['total_events']
        print(f"      [{i+1}/{num_events}] '{event['token_text']}' → {result['classification']} ({result['confidence']:.2f}) | Total: F={stats['factual_events']}({fact_pct:.1f}%) I={stats['interpretive_events']}({interp_pct:.1f}%)")
    
    return article_json


def process_dataset(input_path, output_path, classifier, dataset_name, start_from=0):
    file_names = sorted(os.listdir(input_path))
    total_files = len(file_names)
    
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name}: {total_files} articles")
    print(f"Starting from article {start_from}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    stats = {
        'total_articles': 0,
        'total_events': 0,
        'factual_events': 0,
        'interpretive_events': 0,
        'failed_articles': []
    }
    
    for file_idx in range(start_from, total_files):
        file_name = file_names[file_idx]
        
        try:
            output_file = file_name.replace('_event_graph.json', '_event_graph_classified.json')
            if not output_file.endswith('_classified.json'):
                output_file = file_name.replace('.json', '_classified.json')
            
            output_full_path = f"{output_path}/{output_file}"
            if os.path.exists(output_full_path):
                print(f"  [{file_idx+1}/{total_files}] SKIP: {file_name}")
                continue
            
            with open(f"{input_path}/{file_name}", "r") as f:
                article_json = json.load(f)
            
            num_events = len(article_json.get('event_tokens', []))
            
            print(f"\n  {'='*66}")
            print(f"  [{file_idx+1}/{total_files}] {file_name} | Events: {num_events} | API: {classifier.daily_requests}/{classifier.max_daily_requests}")
            print(f"  {'='*66}")
            
            article_json = classify_events_in_article(article_json, classifier, stats)
            
            stats['total_articles'] += 1
            
            with open(output_full_path, "w", encoding="utf-8") as f:
                json.dump(article_json, f, indent=2, ensure_ascii=False)
            
            fact_pct = 100*stats['factual_events']/max(stats['total_events'],1)
            interp_pct = 100*stats['interpretive_events']/max(stats['total_events'],1)
            
            print(f"\n  SAVED: {output_file}")
            print(f"  {'─'*66}")
            print(f"  CUMULATIVE: Articles={stats['total_articles']}/{total_files-start_from} | Events={stats['total_events']} | F={stats['factual_events']}({fact_pct:.1f}%) I={stats['interpretive_events']}({interp_pct:.1f}%) | API={classifier.daily_requests}/{classifier.max_daily_requests}")
            print(f"  {'─'*66}\n")
            
            if classifier.daily_requests >= classifier.max_daily_requests - 50:
                print(f"\nAPPROACHING DAILY LIMIT! Used {classifier.daily_requests}/{classifier.max_daily_requests}")
                print(f"Resume tomorrow with start_from={file_idx + 1}")
                break
                
        except Exception as e:
            print(f"\n  FAILED: {file_name} - {str(e)[:100]}")
            stats['failed_articles'].append({'file': file_name, 'index': file_idx, 'error': str(e)})
            with open(f"{dataset_name}_failed_log.json", "w") as f:
                json.dump(stats['failed_articles'], f, indent=2)
            continue
    
    print(f"\n{'='*70}")
    print(f"{dataset_name} SUMMARY: {stats['total_articles']} articles, {stats['total_events']} events")
    if stats['total_events'] > 0:
        print(f"Factual: {stats['factual_events']} ({100*stats['factual_events']/stats['total_events']:.1f}%)")
        print(f"Interpretive: {stats['interpretive_events']} ({100*stats['interpretive_events']/stats['total_events']:.1f}%)")
    print(f"API calls: {classifier.daily_requests}/{classifier.max_daily_requests}")
    print(f"{'='*70}\n")
    
    return stats


def main():
    print("="*70)
    print("Factual-Interpretive Event Classification")
    print("Model: gemini-2.5-flash-lite | Limits: 14 RPM, 99000 RPD")
    print("="*70 + "\n")
    
    api_key = input("Enter Gemini API key: ").strip()
    if not api_key:
        print("ERROR: No API key")
        return
    
    classifier = GeminiEventClassifier(api_key=api_key)
    
    os.makedirs("./BASIL_event_graph_classified", exist_ok=True)
    os.makedirs("./BiasedSents_event_graph_classified", exist_ok=True)
    
    print(f"Starting with {classifier.daily_requests} requests used today\n")
    
    start_time = time.time()
    
    # BASIL
    print("="*70)
    print("PHASE 1: BASIL (300 articles)")
    print("="*70)
    basil_start = int(input("Start from article index (Enter for 0): ").strip() or "0")
    
    try:
        basil_stats = process_dataset("./BASIL_event_graph", "./BASIL_event_graph_classified", classifier, "BASIL", basil_start)
    except Exception as e:
        print(f"\nBASIL stopped: {e}\nProgress saved.")
        return
    
    if classifier.daily_requests >= classifier.max_daily_requests - 50:
        print("\nDaily limit reached. Resume BiasedSents tomorrow.")
        return
    
    # BiasedSents
    print("\n" + "="*70)
    print("PHASE 2: BiasedSents (46 articles)")
    print("="*70)
    bs_start = int(input("Start from article index (Enter for 0): ").strip() or "0")
    
    try:
        bs_stats = process_dataset("./BiasedSents_event_graph", "./BiasedSents_event_graph_classified", classifier, "BiasedSents", bs_start)
    except Exception as e:
        print(f"\nBiasedSents stopped: {e}\nProgress saved.")
        return
    
    # Final summary
    elapsed = time.time() - start_time
    total_events = basil_stats['total_events'] + bs_stats['total_events']
    total_factual = basil_stats['factual_events'] + bs_stats['factual_events']
    total_interpretive = basil_stats['interpretive_events'] + bs_stats['interpretive_events']
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Events: {total_events} | F={total_factual}({100*total_factual/total_events:.1f}%) I={total_interpretive}({100*total_interpretive/total_events:.1f}%)")
    print(f"API: {classifier.daily_requests}")
    print("="*70)


if __name__ == "__main__":
    main()