#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -p "ug-gpu-small"
#SBATCH --qos="short"
#SBATCH --job-name="River Level Model Training"
#SBATCH -t 00-02:00:00

source /etc/profile
source $HOME/river-level/bin/activate

python $HOME/river-level-analysis/level_prediction_training/train.py
