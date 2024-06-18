#! /bin/bash

#SBATCH -p fast # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 40G # memory pool for all cores
#SBATCH --gpus-per-node=1
#SBATCH -n 30
#SBATCH -t 0-2:0 # time
#SBATCH --mail-type=END,FAIL

echo "loading conda env"
module load miniconda
# module load cuda/11.8

source activate si

echo "running script"
python ~/ephys/split_concat.py "$1"  "$2"  --ow_flag "${3:-1}"