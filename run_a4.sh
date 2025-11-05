#!/bin/bash
#SBATCH --job-name=a4_no_innov
#SBATCH --output=slurm_logs/a4_%j.out
#SBATCH --error=slurm_logs/a4_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Job [A4] started on $(hostname) at $(date)"
source /home2/lakshmiprajna.p/miniconda3/bin/activate base
cd /scratch/lakshmiprajna.p/Media-Bias-Analysis
python 5_train_bias_classifier_dual_view.py --edge_dropout 0.0 --contrastive_weight 0.0 --results_dir ./results_ablation_4 --gpu_num 4
echo "Job [A4] finished at $(date)"