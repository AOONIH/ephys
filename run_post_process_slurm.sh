#! /bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 40G # memory pool for all cores
#SBATCH --gpus-per-node=1
#SBATCH -n 30
#SBATCH -t 5-0:0 # time
#SBATCH --mail-type=END,FAIL

echo "loading conda env"
module load miniconda
module load cuda/11.6

source activate si

echo "running script"
python ~/ephys/post_process.py "$1"