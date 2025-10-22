# Document-Level Dual-View Architecture Implementation

## Overview
Extended the paragraph-level dual-view GNN to include **document-level dual-view processing**, creating a complete two-level hierarchical architecture where both levels maintain the Factual/Interpretive paradigm.

## Architecture Changes

### Before (Original)
```
Level 1 (Paragraph): Dual-view events → R-GAT → Cross-view attention
Level 2 (Document): Single GAT on all paragraphs (no F/I distinction)
```

### After (New Document-Level Dual-View)
```
Level 1 (Paragraph): Dual-view events → R-GAT → Cross-view attention → F/I aggregation
                     ↓
                   Paragraph Dominance Classification (F or I)
                     ↓
Level 2 (Document): Dual-view paragraphs → Document R-GAT → Document Cross-view attention
                     ↓
                   Final sentence classification
```

## Implementation Details

### 1. New Layer: `Cross_View_Attention_Document`
- **Location**: Lines ~500-555
- **Purpose**: Cross-view attention between Factual-dominant and Interpretive-dominant paragraphs
- **Input**: 
  - `F_paragraphs`: (N_F_para, 768) - Factual-dominant paragraph embeddings
  - `I_paragraphs`: (N_I_para, 768) - Interpretive-dominant paragraph embeddings
- **Output**: Updated F and I paragraph embeddings after cross-view interaction

### 2. Updated Model Components
**File**: `5_train_bias_classifier_dual_view.py`

**New Layers Added**:
```python
# Level 2: Document-Level Dual-View GNN
self.R_GAT_Document_Factual = R_GAT_Layer(feature_dim, self.relation_types)
self.R_GAT_Document_Interpretive = R_GAT_Layer(feature_dim, self.relation_types)
self.Cross_View_Attn_Document = Cross_View_Attention_Document(feature_dim)
```

### 3. Forward Pass Updates

#### Paragraph Dominance Classification
**Location**: Lines ~690-695
```python
# Determine paragraph dominance based on event counts
if F_mask.size(0) >= I_mask.size(0):
    paragraph_dominance[p_idx] = 0  # Factual-dominant
else:
    paragraph_dominance[p_idx] = 1  # Interpretive-dominant
```

#### Document-Level Dual-View Processing
**Location**: Lines ~698-760

**Steps**:
1. **Separate paragraphs** into F-dominant and I-dominant views
2. **Create adjacency matrices** for each view (sequential connections)
3. **Apply Document R-GATs** to each view separately
4. **Cross-view attention** between F and I paragraph graphs
5. **Reconstruct** full paragraph representations
6. **Map to sentences** for final classification

## Key Features

### 1. Paragraph Dominance Heuristic
- Compares number of factual vs interpretive events in each paragraph
- If `#Factual >= #Interpretive` → Factual-dominant
- Otherwise → Interpretive-dominant

### 2. Document-Level Graph Structure
- **Sequential adjacency**: P_i ↔ P_i+1 (narrative flow)
- **Separate graphs**: One for F-paragraphs, one for I-paragraphs
- **Self-loops**: Added for attention stability

### 3. Cross-View Interaction
- Factual paragraphs attend to Interpretive paragraphs
- Interpretive paragraphs attend to Factual paragraphs
- Enables modeling of factual-interpretive interplay at document level

## Benefits

1. **Hierarchical Consistency**: Dual-view paradigm maintained across both levels
2. **Richer Representation**: Document structure captures both factual and interpretive narrative flows
3. **Better Bias Detection**: Cross-document patterns in how factual/interpretive content is organized
4. **Improved from Baseline**: Builds on proven paragraph-level improvements

## Training & Testing

### No Changes Required for:
- Dataset loading
- Training loop
- Evaluation metrics
- 10-fold cross-validation setup

### Model is backwards compatible:
- If document has only F or I paragraphs, code handles gracefully
- Empty views are handled with zero-sized tensor checks

## Next Steps

1. **Install dependencies** (if not already done):
   ```bash
   pip install torch transformers sklearn numpy
   ```

2. **Test the architecture**:
   ```bash
   python3 test_architecture.py  # Already verified ✓
   ```

3. **Run training** (single fold test):
   ```bash
   python3 5_train_bias_classifier_dual_view.py --fold 0 --gpu 0
   ```

4. **Full 10-fold CV** (if using SLURM):
   ```bash
   sbatch run_all_folds.sh
   ```

## Expected Improvements

Based on paragraph-level gains, document-level dual-view should provide:
- Better capture of document-wide bias patterns
- Improved modeling of how factual/interpretive content is structured
- Enhanced sentence-level predictions through richer contextual information

## Architecture Diagram

```
Input Tokens (Longformer)
        ↓
   BiLSTM Encoding
        ↓
Event Embeddings (F/I labeled)
        ↓
┌─────────────────────────────────────┐
│  Level 1: Paragraph-Level Dual-View │
├─────────────────────────────────────┤
│  ┌─────────┐      ┌─────────┐      │
│  │ F Events│      │ I Events│      │
│  │  R-GAT  │      │  R-GAT  │      │
│  └────┬────┘      └────┬────┘      │
│       │                 │           │
│       └──Cross-View─────┘           │
│               ↓                     │
│      Paragraph Aggregation          │
│         (F + I Summary)             │
│               ↓                     │
│    Paragraph Dominance (F or I)    │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  Level 2: Document-Level Dual-View  │
├─────────────────────────────────────┤
│  ┌──────────┐     ┌──────────┐     │
│  │ F-Para   │     │ I-Para   │     │
│  │ Doc R-GAT│     │ Doc R-GAT│     │
│  └────┬─────┘     └────┬─────┘     │
│       │                 │           │
│       └──Cross-View─────┘           │
│               ↓                     │
│   Updated Paragraph Reps            │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  Final Sentence Classification      │
├─────────────────────────────────────┤
│  Sentence Emb + Para Context +      │
│  Event Aggregation → Bias Head      │
└─────────────────────────────────────┘
```

## File Changes Summary

- **Modified**: `5_train_bias_classifier_dual_view.py`
  - Added `Cross_View_Attention_Document` class
  - Added document-level R-GAT layers to model
  - Updated forward pass for dual-view document processing
  - Added paragraph dominance tracking

- **Created**: `test_architecture.py` (for verification)
- **Created**: `DOCUMENT_LEVEL_DUAL_VIEW_SUMMARY.md` (this file)

---
**Implementation Date**: October 22, 2025
**Status**: ✅ Complete and ready for training
