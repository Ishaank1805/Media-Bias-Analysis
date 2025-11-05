#!/bin/bash
#SBATCH --job-name=a1_unified
#SBATCH --output=slurm_logs/a1_%j.out
#SBATCH --error=slurm_logs/a1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Job [A1] started on $(hostname) at $(date)"
source /home2/lakshmiprajna.p/miniconda3/bin/activate base
cd /scratch/lakshmiprajna.p/Media-Bias-Analysis
python ablation_1_unified_graph.py --results_dir ./results_ablation_1 --gpu_num 4
echo "Job [A1] finished at $(date)"