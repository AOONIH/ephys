#! /bin/bash

#SBATCH -p cpu # partition (queue)
#SBATCH --mem 32 # memory pool for all cores
#SBATCH -n 30
#SBATCH -t 2-00:0 # time
#SBATCH --mail-type=END,FAIL

echo "loading conda env"
module load miniconda


source activate si

echo "running script"
python ~/ephys/ephys_analysis_multisess.py config.yaml "$1" --sorter_dirname "${2:-'from_concat'}" --sess_top_filts "${3:-}"