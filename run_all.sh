#!/bin/bash

echo "Submitting all 7 experiments..."
echo "---"

sbatch run_b1.sh
sbatch run_b2.sh
sbatch run_a1.sh
sbatch run_a2.sh
sbatch run_a3.sh
sbatch run_a4.sh

echo "---"
echo "All jobs submitted."
echo "Check status with: squeue -u lakshmiprajna.p"