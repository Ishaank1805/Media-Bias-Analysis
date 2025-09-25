# event_extraction.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import spacy
from allennlp.predictors import Predictor

@dataclass
class Event:
    """Representation of an extracted event"""
    trigger: str
    trigger_span: Tuple[int, int]
    event_type: Optional[str]
    arguments: Dict[str, List[Tuple[str, int, int]]]  # role -> [(text, start, end)]
    confidence: float
    sentence_idx: int
    embeddings: Optional[torch.Tensor] = None


class EventTriggerDetector(nn.Module):
    """Detect event triggers in text"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Binary classifier for each token
        self.trigger_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # is_trigger or not
        )
        
        # Event type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 33)  # 33 event types in ACE
        )
        
    def forward(self, sentences: List[str]) -> List[List[Event]]:
        """Detect events in sentences"""
        
        all_events = []
        
        for sent_idx, sentence in enumerate(sentences):
            # Tokenize
            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                padding=True,
                truncation=True,
                return_offsets_mapping=True
            )
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.bert(**{k:v for k,v in inputs.items() 
                                    if k != 'offset_mapping'})
                
            hidden_states = outputs.last_hidden_state
            
            # Classify each token
            trigger_logits = self.trigger_classifier(hidden_states)
            trigger_probs = torch.softmax(trigger_logits, dim=-1)
            
            # Find trigger tokens
            trigger_mask = trigger_probs[:, :, 1] > 0.5
            
            sentence_events = []
            
            # For each trigger
            trigger_indices = torch.where(trigger_mask[0])[0]
            
            for idx in trigger_indices:
                # Get trigger span
                offset = inputs['offset_mapping'][0][idx]
                trigger_text = sentence[offset[0]:offset[1]]
                
                # Classify event type
                type_logits = self.type_classifier(hidden_states[0, idx])
                type_probs = torch.softmax(type_logits, dim=-1)
                event_type_idx = torch.argmax(type_probs)
                
                event = Event(
                    trigger=trigger_text,
                    trigger_span=(offset[0].item(), offset[1].item()),
                    event_type=self._idx_to_event_type(event_type_idx.item()),
                    arguments={},  # Will be filled by argument extractor
                    confidence=trigger_probs[0, idx, 1].item(),
                    sentence_idx=sent_idx,
                    embeddings=hidden_states[0, idx].detach()
                )
                
                sentence_events.append(event)
                
            all_events.append(sentence_events)
            
        return all_events
    
    def _idx_to_event_type(self, idx: int) -> str:
        """Map index to event type string"""
        # ACE event types
        event_types = [
            'Movement:Transport', 'Personnel:Start-Position', 'Conflict:Attack',
            'Contact:Meet', 'Personnel:End-Position', 'Life:Die',
            # ... (full list of 33 types)
        ]
        return event_types[idx] if idx < len(event_types) else 'Other'


class EventArgumentExtractor(nn.Module):
    """Extract arguments for detected events"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Role classifier
        self.role_classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),  # Concatenated trigger and candidate
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 36)  # Number of argument roles
        )
        
        # Load SpaCy for entity detection
        self.nlp = spacy.load('en_core_web_sm')
        
    def extract_arguments(self, sentence: str, event: Event) -> Dict[str, List[Tuple[str, int, int]]]:
        """Extract arguments for a given event"""
        
        # Parse sentence
        doc = self.nlp(sentence)
        
        # Get candidate arguments (entities and noun phrases)
        candidates = []
        
        # Add named entities
        for ent in doc.ents:
            candidates.append((ent.text, ent.start_char, ent.end_char))
            
        # Add noun chunks
        for chunk in doc.noun_chunks:
            candidates.append((chunk.text, chunk.start_char, chunk.end_char))
            
        # Remove duplicates and trigger overlap
        candidates = self._filter_candidates(candidates, event.trigger_span)
        
        # Encode sentence
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True)
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            
        sentence_embedding = outputs.last_hidden_state
        
        # Get trigger embedding
        trigger_start_token = self._char_to_token(event.trigger_span[0], inputs)
        trigger_embedding = sentence_embedding[0, trigger_start_token]
        
        arguments = {}
        
        # Classify each candidate
        for cand_text, start, end in candidates:
            # Get candidate embedding
            cand_start_token = self._char_to_token(start, inputs)
            cand_embedding = sentence_embedding[0, cand_start_token]
            
            # Concatenate trigger and candidate
            combined = torch.cat([trigger_embedding, cand_embedding])
            
            # Predict role
            role_logits = self.role_classifier(combined)
            role_probs = torch.softmax(role_logits, dim=-1)
            
            # Get top role
            role_idx = torch.argmax(role_probs)
            role_conf = role_probs[role_idx]
            
            if role_conf > 0.5:  # Confidence threshold
                role = self._idx_to_role(role_idx.item())
                
                if role not in arguments:
                    arguments[role] = []
                    
                arguments[role].append((cand_text, start, end))
                
        return arguments
    
    def _filter_candidates(self, candidates: List[Tuple[str, int, int]], 
                          trigger_span: Tuple[int, int]) -> List[Tuple[str, int, int]]:
        """Filter out invalid candidates"""
        
        filtered = []
        seen = set()
        
        for text, start, end in candidates:
            # Skip if overlaps with trigger
            if not (end <= trigger_span[0] or start >= trigger_span[1]):
                continue
                
            # Skip duplicates
            key = (text, start, end)
            if key in seen:
                continue
                
            seen.add(key)
            filtered.append((text, start, end))
            
        return filtered
    
    def _char_to_token(self, char_idx: int, encoding) -> int:
        """Convert character index to token index"""
        # Simple implementation - would need proper alignment
        return min(char_idx // 4, encoding['input_ids'].shape[1] - 1)
    
    def _idx_to_role(self, idx: int) -> str:
        """Map index to argument role"""
        roles = [
            'Agent', 'Patient', 'Time', 'Place', 'Instrument',
            'Source', 'Destination', 'Beneficiary',
            # ... (full list of roles)
        ]
        return roles[idx] if idx < len(roles) else 'Other'


class EventRepresentationEncoder:
    """Encode events into vector representations"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode_events(self, events: List[Event], sentences: List[str]) -> List[torch.Tensor]:
        """Create embeddings for events"""
        
        event_embeddings = []
        
        for event in events:
            sentence = sentences[event.sentence_idx]
            
            # Create event context string
            context = self._create_event_context(event, sentence)
            
            # Encode
            inputs = self.tokenizer(
                context,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use CLS token as event representation
            event_embedding = outputs.last_hidden_state[0, 0]
            
            event_embeddings.append(event_embedding)
            
        return event_embeddings
    
    def _create_event_context(self, event: Event, sentence: str) -> str:
        """Create context string for event"""
        
        # Include trigger and surrounding context
        context_parts = []
        
        # Add event type
        context_parts.append(f"[EVENT: {event.event_type}]")
        
        # Add sentence with trigger marked
        marked_sentence = (
            sentence[:event.trigger_span[0]] +
            f"[TRIGGER: {event.trigger}]" +
            sentence[event.trigger_span[1]:]
        )
        context_parts.append(marked_sentence)
        
        # Add arguments
        for role, args in event.arguments.items():
            arg_texts = [arg[0] for arg in args]
            context_parts.append(f"[{role}: {', '.join(arg_texts)}]")
            
        return " ".join(context_parts)


class PretrainedEventExtractor:
    """Wrapper for pre-trained event extraction models"""
    
    def __init__(self, model_type: str = 'dygiepp'):
        self.model_type = model_type
        
        if model_type == 'dygiepp':
            # Load DyGIE++ model
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/dygiepp-ace2005.tar.gz"
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def extract_events(self, sentences: List[str]) -> List[List[Event]]:
        """Extract events using pre-trained model"""
        
        all_events = []
        
        for sent_idx, sentence in enumerate(sentences):
            # Run prediction
            prediction = self.predictor.predict(sentence=sentence)
            
            sentence_events = []
            
            # Parse DyGIE++ output
            if 'events' in prediction:
                for event_data in prediction['events']:
                    trigger_span = event_data['trigger']
                    trigger_text = sentence[trigger_span[0]:trigger_span[1]]
                    
                    # Extract arguments
                    arguments = {}
                    for arg in event_data.get('arguments', []):
                        role = arg['role']
                        span = arg['span']
                        text = sentence[span[0]:span[1]]
                        
                        if role not in arguments:
                            arguments[role] = []
                        arguments[role].append((text, span[0], span[1]))
                    
                    event = Event(
                        trigger=trigger_text,
                        trigger_span=tuple(trigger_span),
                        event_type=event_data.get('event_type', 'Unknown'),
                        arguments=arguments,
                        confidence=event_data.get('score', 1.0),
                        sentence_idx=sent_idx
                    )
                    
                    sentence_events.append(event)
                    
            all_events.append(sentence_events)
            
        return all_events
    
    def extract_relations(self, events: List[Event], sentences: List[str]) -> List[Dict]:
        """Extract relations between events"""
        
        # This would integrate with MAVEN-ERE relation extraction
        # For now, return placeholder
        relations = []
        
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events[i+1:], i+1):
                # Would call relation classification model here
                relations.append({
                    'source': i,
                    'target': j,
                    'type': 'temporal',  # placeholder
                    'subtype': 'before',
                    'confidence': 0.8
                })
                
        return relations


# Integration class that combines all extractors
class EventExtractionPipeline:
    """Complete pipeline for event extraction"""
    
    def __init__(self, use_pretrained: bool = True):
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            self.extractor = PretrainedEventExtractor()
        else:
            self.trigger_detector = EventTriggerDetector()
            self.argument_extractor = EventArgumentExtractor()
            
        self.representation_encoder = EventRepresentationEncoder()
        
    def process_document(self, document: Document) -> Tuple[List[List[Event]], List[torch.Tensor]]:
        """Extract events from document and encode them"""
        
        # Extract events
        if self.use_pretrained:
            events_by_sentence = self.extractor.extract_events(document.sentences)
        else:
            # Detect triggers
            events_by_sentence = self.trigger_detector(document.sentences)
            
            # Extract arguments
            for sent_idx, (sentence, events) in enumerate(zip(document.sentences, events_by_sentence)):
                for event in events:
                    event.arguments = self.argument_extractor.extract_arguments(sentence, event)
        
        # Flatten events and create embeddings
        all_events = []
        for events in events_by_sentence:
            all_events.extend(events)
            
        event_embeddings = self.representation_encoder.encode_events(all_events, document.sentences)
        
        return events_by_sentence, event_embeddings