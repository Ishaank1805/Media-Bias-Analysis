# Two-Level Dual-View Event Relation Graph for Media Bias Detection

## Project Overview

This project implements a hierarchical dual-view event relation graph architecture for sentence-level media bias detection. The approach separates factual events from interpretive events and processes them through paragraph-level and document-level graph neural networks.

## Pre-trained Models and Datasets

All trained models and processed datasets are available in our Google Drive:

**📁 Google Drive Link:** https://drive.google.com/drive/folders/1R6fncQVNWPjmbIGHg8PZY0pQZEU441Oq?usp=sharing

This drive contains:
- Trained event extraction models (6 checkpoints)
- Preprocessed MAVEN-ERE files
- BASIL and BiasedSents event graphs
- Classified event graphs with factual/interpretive labels

You can download these to skip training stages and start directly from the classified data.

---

## Prerequisites

### Software Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Python Libraries
```bash
pip install torch torchvision
pip install transformers
pip install scikit-learn
pip install nltk
pip install google-generativeai
```

### Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt')"
```

### API Requirements
- Gemini API key (get from Google AI Studio)
- Rate limits: 14 requests/minute, 99,000 requests/day

---

## Dataset Setup

### 1. Download Datasets

**MAVEN-ERE Dataset:**

- Files needed: `train.jsonl`, `valid.jsonl`
- Place in: `./MAVEN_ERE/`

**BASIL Dataset:**

- Place original JSON files in: `./BASIL/`

**BiasedSents Dataset:**

- Place original JSON files in: `./BiasedSents/`

### 2. Create Output Directories

```bash
mkdir -p MAVEN_ERE/train
mkdir -p MAVEN_ERE/dev
mkdir -p BASIL_event_graph
mkdir -p BiasedSents_event_graph
mkdir -p BASIL_event_graph_classified
mkdir -p BiasedSents_event_graph_classified
mkdir -p saved_models/event_relation_graph
```

---

## Pipeline Execution

### Stage 1: MAVEN-ERE Preprocessing

**Purpose:** Convert MAVEN-ERE from JSONL format to individual article JSONs

**Script:** Data preprocessing module

**Input:**
- `./MAVEN_ERE/train.jsonl` (2,913 articles)
- `./MAVEN_ERE/valid.jsonl` (710 articles)

**Output:**
- `./MAVEN_ERE/train/*.json` (2,913 files)
- `./MAVEN_ERE/dev/*.json` (710 files)

**Run:**
```bash
python 1_preprocess_mavenere.py
```

**What it does:**
- Tokenizes articles into words
- Extracts event mentions with triggers and spans
- Creates all possible event pairs
- Assigns relation labels (coreference, temporal, causal, subevent)
- Saves one JSON per article

**Expected runtime:** 10-15 minutes

**Status:** ✅ COMPLETED

---

### Stage 2: Event Extractor Training

**Purpose:** Train joint model for event identification and relation extraction

**Script:** Training module for event relation graph

**Input:**
- `./MAVEN_ERE/train/*.json`
- `./MAVEN_ERE/dev/*.json`

**Output:**
- `./saved_models/event_relation_graph/best_macro_F_event.ckpt`
- `./saved_models/event_relation_graph/best_macro_F_coreference.ckpt`
- `./saved_models/event_relation_graph/best_macro_F_temporal.ckpt`
- `./saved_models/event_relation_graph/best_macro_F_causal.ckpt`
- `./saved_models/event_relation_graph/best_macro_F_subevent.ckpt`
- `./saved_models/event_relation_graph/best_macro_F_graph.ckpt`

**Configuration:**
- Uses 20% of MAVEN-ERE data (583 train, 142 dev)
- 5 training epochs
- Batch size: 1
- Learning rates: Longformer 1e-5, new layers 1e-4

**Run:**
```bash
python 2_train_event_extractors.py
```

**Expected runtime:** 4-6 hours on GPU

**Expected performance:**
- Event Identification: Macro F1 ≈ 89.40
- Coreference: CoNLL F1 ≈ 88.30
- Temporal: Macro F1 ≈ 47.04
- Causal: Macro F1 ≈ 56.01
- Subevent: Macro F1 ≈ 46.21

**Status:** ✅ COMPLETED

---

### Stage 3: Event Graph Construction

**Purpose:** Extract events and relations from BASIL and BiasedSents

**Script:** Event graph builder

**Input:**
- `./BASIL/*.json` (300 original articles)
- `./BiasedSents/*.json` (46 original articles)
- Trained models from Stage 2

**Output:**
- `./BASIL_event_graph/*.json` (300 files with extracted events)
- `./BiasedSents_event_graph/*.json` (46 files with extracted events)

**Run:**
```bash
python 3_extract_event_graphs.py
```

**What it does:**
- Tokenizes BASIL and BiasedSents articles
- Identifies event triggers (threshold: probability > 0.5)
- Creates all event pairs
- Predicts four relation types with probabilities
- Stores events, event pairs, and relation predictions

**Expected runtime:** 30-45 minutes

**Status:** ✅ COMPLETED

---

### Stage 4: Factual/Interpretive Classification

**Purpose:** Classify each extracted event as FACTUAL or INTERPRETIVE

**Script:** Gemini-based event classifier

**Input:**
- `./BASIL_event_graph/*.json`
- `./BiasedSents_event_graph/*.json`
- Gemini API key

**Output:**
- `./BASIL_event_graph_classified/*.json` (300 files)
- `./BiasedSents_event_graph_classified/*.json` (46 files)
- `gemini_request_log.pkl` (tracks daily API usage)

**Run:**
```bash
python 4_classify_factual_interpretive.py
```

**Interactive prompts:**
1. Enter your Gemini API key
2. Enter starting article index for BASIL (0 if starting fresh)
3. Enter starting article index for BiasedSents (0 if starting fresh)

**What it does:**
- Reads each article's event graph
- For each event, sends context + trigger to Gemini API
- Receives classification (FACTUAL/INTERPRETIVE), confidence, reasoning
- Handles rate limiting automatically (14 RPM, 99K RPD)
- Saves enhanced JSONs with three new fields per event:
  - `fi_classification`: "FACTUAL" or "INTERPRETIVE"
  - `fi_confidence`: 0.0-1.0
  - `fi_reasoning`: Brief explanation

**Resume capability:** If interrupted, restart and provide the last completed article index to continue

**Expected runtime:** 1-2 hours (depending on API speed)

**Expected distribution:** ~60-70% factual, ~30-40% interpretive

**Status:** ✅ COMPLETED

---

## Current Status Summary

### ✅ Completed Pipeline (Stages 1-4)

All data preparation is complete:

| Stage | Task | Input | Output | Status |
|-------|------|-------|--------|--------|
| 1 | MAVEN-ERE Preprocessing | JSONL files | 3,623 individual JSONs | ✅ Done |
| 2 | Event Extractor Training | MAVEN-ERE JSONs | 6 trained models | ✅ Done |
| 3 | Event Graph Construction | BASIL/BiasedSents | 346 event graphs | ✅ Done |
| 4 | F/I Classification | Event graphs | 346 classified graphs | ✅ Done |

**Data Ready For:** Two-level dual-view model implementation

---

## Next Steps (Upcoming Implementation)

### Stage 5: Two-Level Dual-View Model (Weeks 4-5)

**Timeline:** October 3-16, 2025

**What needs to be built:**
1. Custom data loader reading classified event graphs
2. Paragraph boundary detection system
3. Paragraph-level dual-view GNN module
4. Document-level GNN module
5. End-to-end model integration
6. Helper functions for graph organization

**Estimated implementation:** 700-900 lines of new code

### Stage 6: Training & Evaluation (Weeks 6-7)

**Timeline:** October 17-30, 2025

**Tasks:**
- 10-fold cross-validation on BASIL and BiasedSents
- Baseline comparisons
- Ablation studies
- Error analysis and visualization

---

