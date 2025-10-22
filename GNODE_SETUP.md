# Running on GNode (1 GPU, 10 CPUs)

## Changes Made for Single GPU Setup

### 1. **Removed Multi-GPU Logic**
- Changed `--gpu` from required choices `[0,1,2,3]` to optional with default `0`
- Removed `CUDA_VISIBLE_DEVICES` environment variable manipulation
- Simplified device setup to `cuda:0`

### 2. **Updated Output Directory**
- Changed from `/scratch/atharv.johar/...` to `./results_dual_view`
- You can customize this with `--output_dir` argument

### 3. **Current Hyperparameters** (Already Optimized)
```python
MAX_LEN = 2048              # Token length
num_epochs = 10             # Training epochs
batch_size = 1              # Due to memory constraints
```

## How to Run

### Single Fold Test
```bash
# Run fold 0
python3 5_train_bias_classifier_dual_view.py --fold 0

# Or explicitly specify GPU (same as default)
python3 5_train_bias_classifier_dual_view.py --fold 0 --gpu 0

# With custom output directory
python3 5_train_bias_classifier_dual_view.py --fold 0 --output_dir /path/to/results
```

### Run All 10 Folds Sequentially
Since you have **1 GPU**, you should run folds **sequentially** (one after another):

```bash
#!/bin/bash
# run_all_folds_sequential.sh

for fold in {0..9}
do
    echo "Starting fold $fold"
    python3 5_train_bias_classifier_dual_view.py --fold $fold
    echo "Completed fold $fold"
done
```

Make it executable and run:
```bash
chmod +x run_all_folds_sequential.sh
./run_all_folds_sequential.sh
```

### Using SLURM (If Available on GNode)
Create a SLURM script to run all folds:

```bash
#!/bin/bash
#SBATCH --job-name=bias_cv
#SBATCH --output=logs/fold_%a.out
#SBATCH --error=logs/fold_%a.err
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Create logs directory
mkdir -p logs

# Activate your environment
source venv/bin/activate

# Run the fold corresponding to the array task ID
python3 5_train_bias_classifier_dual_view.py --fold $SLURM_ARRAY_TASK_ID
```

Submit with:
```bash
sbatch run_slurm.sh
```

## Resource Usage Estimates

### Single Fold Training Time
- **Estimated time**: 2-4 hours per fold (depends on GPU speed)
- **GPU Memory**: ~8-12 GB (with MAX_LEN=2048, batch_size=1)
- **CPU Usage**: Mainly for data loading (10 CPUs will help)

### Full 10-Fold CV
- **Sequential**: ~20-40 hours total (all folds one after another)
- **With SLURM array**: Can parallelize if your cluster allows multiple jobs

## Monitoring GPU Usage

While training is running, monitor GPU in another terminal:
```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Or more detailed
nvitop  # if installed: pip install nvitop
```

## Optimization Tips for Single GPU

### If Running Out of Memory
1. **Reduce MAX_LEN**: Change from 2048 to 1024
   ```python
   MAX_LEN = 1024
   ```

2. **Gradient Accumulation** (if you want effective batch_size > 1):
   ```python
   accumulation_steps = 4  # Effective batch_size = 4
   # Then in training loop, accumulate gradients
   ```

### If Training is Too Slow
1. **Reduce num_epochs**: From 10 to 5 or 8
   ```python
   num_epochs = 8
   ```

2. **Use mixed precision training** (add to your code):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   # In training loop:
   with autocast():
       loss = model(batch)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

## Expected Output

After each fold completes, you'll get:
- `results_dual_view/fold_X_best.ckpt` - Best model checkpoint
- `results_dual_view/fold_X_results.json` - Test results with metrics

Example output structure:
```
results_dual_view/
├── fold_0_best.ckpt
├── fold_0_results.json
├── fold_1_best.ckpt
├── fold_1_results.json
...
└── fold_9_results.json
```

## Aggregating Results

After all folds complete, aggregate results:

```python
import json
import numpy as np

results = []
for fold in range(10):
    with open(f'results_dual_view/fold_{fold}_results.json') as f:
        results.append(json.load(f))

# Calculate average metrics
macro_f1s = [r['test_macro_F'] for r in results]
bias_f1s = [r['test_biased_F'] for r in results]

print(f"Average Macro F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
print(f"Average Bias F1: {np.mean(bias_f1s):.4f} ± {np.std(bias_f1s):.4f}")
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Check what's using GPU memory
nvidia-smi

# Kill any hanging processes
kill -9 <PID>

# Then reduce MAX_LEN or clear cache in your code
torch.cuda.empty_cache()
```

### Slow Data Loading
- 10 CPUs should be enough
- If still slow, check if data is on fast storage (not network drive)

### Connection Lost / Training Interrupted
Use `tmux` or `screen` to keep training running:
```bash
# Start a tmux session
tmux new -s training

# Run your training
python3 5_train_bias_classifier_dual_view.py --fold 0

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

---
**Setup Date**: October 22, 2025  
**Hardware**: 1 GPU, 10 CPUs  
**Status**: ✅ Ready to run
