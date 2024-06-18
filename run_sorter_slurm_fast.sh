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
module load cuda/11.8

source activate si

echo "running script"
echo $2 $3
python ~/ephys/sorting_functions.py "$1"  --datadir "${2:''}" --extra_datadirs "${3:-''}"