# dataset_loaders.py

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

@dataclass
class Document:
    """Structured representation of a document"""
    doc_id: str
    text: str
    sentences: List[str]
    paragraphs: List[List[int]]  # List of sentence indices per paragraph
    label: Optional[str] = None  # neutral/left/right
    sentence_labels: Optional[List[int]] = None  # For BASIL/BiasedSents

@dataclass
class EventAnnotation:
    """Event annotation structure"""
    event_id: str
    trigger: str
    trigger_pos: Tuple[int, int]
    event_type: str
    arguments: Dict[str, str]
    sentence_idx: int


class BASILDatasetLoader:
    """Loader for BASIL media bias dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.documents = []
        
    def load(self) -> List[Document]:
        """Load BASIL dataset with bias annotations"""
        
        # BASIL has 300 articles with sentence-level annotations
        basil_file = self.data_path / 'basil_annotations.jsonl'
        
        with open(basil_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                # Extract document info
                doc_id = data['doc_id']
                text = data['full_text']
                
                # Process sentences
                sentences = data['sentences']
                sentence_texts = [s['text'] for s in sentences]
                
                # Extract bias labels (0: non-bias, 1: bias)
                # BASIL annotates both lexical and informational bias
                sentence_labels = []
                for s in sentences:
                    has_lexical = s.get('lexical_bias', 0)
                    has_informational = s.get('informational_bias', 0)
                    # Sentence is biased if it has either type
                    sentence_labels.append(1 if (has_lexical or has_informational) else 0)
                
                # Infer document label from sentence labels
                bias_ratio = sum(sentence_labels) / len(sentence_labels)
                if bias_ratio > 0.3:
                    doc_label = data.get('political_lean', 'unknown')
                else:
                    doc_label = 'neutral'
                
                # Extract paragraph structure
                paragraphs = self._extract_paragraphs(data.get('paragraph_breaks', []), 
                                                    len(sentences))
                
                doc = Document(
                    doc_id=doc_id,
                    text=text,
                    sentences=sentence_texts,
                    paragraphs=paragraphs,
                    label=doc_label,
                    sentence_labels=sentence_labels
                )
                
                self.documents.append(doc)
                
        return self.documents
    
    def _extract_paragraphs(self, paragraph_breaks: List[int], 
                           num_sentences: int) -> List[List[int]]:
        """Convert paragraph breaks to sentence groupings"""
        paragraphs = []
        start = 0
        
        for break_idx in paragraph_breaks:
            paragraphs.append(list(range(start, break_idx)))
            start = break_idx
            
        # Last paragraph
        if start < num_sentences:
            paragraphs.append(list(range(start, num_sentences)))
            
        return paragraphs


class BiasedSentsDatasetLoader:
    """Loader for BiasedSents dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.documents = []
        
    def load(self) -> List[Document]:
        """Load BiasedSents with multi-annotator labels"""
        
        biased_sents_file = self.data_path / 'biasedsents.csv'
        df = pd.read_csv(biased_sents_file)
        
        # Group by article
        for article_id, article_df in df.groupby('article_id'):
            
            sentences = article_df['sentence'].tolist()
            
            # BiasedSents has 4 scales: 0=not biased, 1=slightly, 2=biased, 3=very biased
            # Binary: 0,1 -> 0 (non-bias), 2,3 -> 1 (bias)
            # Take majority vote from 5 annotators
            sentence_labels = []
            for _, row in article_df.iterrows():
                votes = [row[f'annotator_{i}'] for i in range(1, 6)]
                binary_votes = [1 if v >= 2 else 0 for v in votes]
                majority = 1 if sum(binary_votes) >= 3 else 0
                sentence_labels.append(majority)
            
            # Reconstruct full text
            text = ' '.join(sentences)
            
            # Simple paragraph detection (every 5 sentences)
            paragraphs = []
            for i in range(0, len(sentences), 5):
                paragraphs.append(list(range(i, min(i+5, len(sentences)))))
            
            # Infer document label
            doc_label = self._infer_document_label(sentence_labels)
            
            doc = Document(
                doc_id=f"biasedsents_{article_id}",
                text=text,
                sentences=sentences,
                paragraphs=paragraphs,
                label=doc_label,
                sentence_labels=sentence_labels
            )
            
            self.documents.append(doc)
            
        return self.documents
    
    def _infer_document_label(self, sentence_labels: List[int]) -> str:
        """Infer document-level label from sentences"""
        bias_ratio = sum(sentence_labels) / len(sentence_labels)
        if bias_ratio < 0.2:
            return 'neutral'
        else:
            # Would need additional info to determine left/right
            return 'biased'  # Generic bias label


class MAVENEREDatasetLoader:
    """Loader for MAVEN-ERE event relation dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.documents = []
        
    def load(self) -> List[Tuple[Document, List[EventAnnotation]]]:
        """Load MAVEN-ERE for pre-training event extractors"""
        
        maven_file = self.data_path / 'maven_ere_train.jsonl'
        documents_with_events = []
        
        with open(maven_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                # Extract document
                doc_id = data['doc_id']
                sentences = [s['text'] for s in data['sentences']]
                text = ' '.join(sentences)
                
                # Simple paragraph grouping
                paragraphs = []
                para_size = 5
                for i in range(0, len(sentences), para_size):
                    paragraphs.append(list(range(i, min(i+para_size, len(sentences)))))
                
                doc = Document(
                    doc_id=doc_id,
                    text=text,
                    sentences=sentences,
                    paragraphs=paragraphs,
                    label=None  # No bias labels in MAVEN
                )
                
                # Extract event annotations
                events = []
                for event in data['events']:
                    for mention in event['mentions']:
                        event_ann = EventAnnotation(
                            event_id=event['id'],
                            trigger=mention['trigger_word'],
                            trigger_pos=(mention['start'], mention['end']),
                            event_type=event['type'],
                            arguments=mention.get('arguments', {}),
                            sentence_idx=mention['sentence_id']
                        )
                        events.append(event_ann)
                
                documents_with_events.append((doc, events))
                
        return documents_with_events


class DocumentPreprocessor:
    """Preprocess documents for consistent format"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
    def preprocess(self, raw_text: str) -> Document:
        """Convert raw text to structured document"""
        
        # Split into sentences
        sentences = sent_tokenize(raw_text)
        
        # Detect paragraphs (simple heuristic: double newlines)
        paragraphs = self._detect_paragraphs(raw_text, sentences)
        
        # Generate doc ID
        doc_id = f"doc_{hash(raw_text[:100])}"
        
        return Document(
            doc_id=doc_id,
            text=raw_text,
            sentences=sentences,
            paragraphs=paragraphs,
            label=None
        )
    
    def _detect_paragraphs(self, text: str, sentences: List[str]) -> List[List[int]]:
        """Detect paragraph boundaries"""
        
        # Split by double newlines
        raw_paragraphs = text.split('\n\n')
        
        paragraphs = []
        sent_idx = 0
        
        for para_text in raw_paragraphs:
            para_sentences = []
            
            # Find which sentences belong to this paragraph
            while sent_idx < len(sentences):
                if sentences[sent_idx] in para_text:
                    para_sentences.append(sent_idx)
                    sent_idx += 1
                else:
                    break
                    
            if para_sentences:
                paragraphs.append(para_sentences)
                
        # Handle case where no clear paragraphs found
        if not paragraphs:
            # Default: treat every 5 sentences as paragraph
            for i in range(0, len(sentences), 5):
                paragraphs.append(list(range(i, min(i+5, len(sentences)))))
                
        return paragraphs


class BiasDataset(Dataset):
    """PyTorch Dataset wrapper for bias detection"""
    
    def __init__(self, documents: List[Document], tokenizer=None):
        self.documents = documents
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        
        # Return document with all necessary info
        item = {
            'doc_id': doc.doc_id,
            'sentences': doc.sentences,
            'paragraphs': doc.paragraphs,
            'label': self._label_to_idx(doc.label) if doc.label else -1
        }
        
        if doc.sentence_labels is not None:
            item['sentence_labels'] = torch.tensor(doc.sentence_labels)
            
        # Optionally tokenize if tokenizer provided
        if self.tokenizer:
            item['input_ids'] = []
            item['attention_masks'] = []
            
            for sent in doc.sentences:
                encoded = self.tokenizer(
                    sent,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                item['input_ids'].append(encoded['input_ids'])
                item['attention_masks'].append(encoded['attention_mask'])
                
        return item
    
    def _label_to_idx(self, label: str) -> int:
        """Convert string label to index"""
        label_map = {
            'neutral': 0,
            'left': 1,
            'right': 2,
            'biased': 3  # Generic bias
        }
        return label_map.get(label, -1)