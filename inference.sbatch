#!/bin/bash
#
#SBATCH --job-name=KongNet_Infer # Job name for tracking
#SBATCH --partition=dgxa100 # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20     # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=60000             # Memory (mb) to use 
#SBATCH --time=24:00:00      # Job time limit
#
#SBATCH --output=/home/u1910100/logs/inference_%j.out # Standard out from your job
#SBATCH --error=/home/u1910100/logs/inference_%j.err  # Standard error from your job

## Activate Conda Environmnet ##
echo "init env ... "
conda init
source /home/u1910100/.bashrc
cd ~
source activate /home/u1910100/cloud_workspace/conda/tiatoolbox
echo $PATH
echo "done"

export HF_TOKEN="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"

## Run Python Program ##
cd /home/u1910100/cloud_workspace/GitHub/KongNet_Inference_Main
echo "starting inference script"
nvidia-smi
lscpu

./example_scripts.sh

echo "finished"