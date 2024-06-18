#! /bin/bash

#SBATCH -N 1   # number of nodes
#SBATCH --mem 60G # memory pool for all cores
#SBATCH --gpus-per-node=1
#SBATCH -n 30
#SBATCH --mail-type=END,FAIL

echo "loading conda env"
module load miniconda
module load cuda/11.8

source activate si

echo "running script"
echo $2 $3
echo ~/ephys/sorting_functions.py "$1"  --datadir "${2:-}" --extra_datadirs "${3:-}" --ow_flag "${5:-0}"
# python ~/ephys/sorting_functions.py "$1"  --datadir "${2:-}" --extra_datadirs "${3:-}" --ow_flag "${5:-0}"

# echo "running split and ephys analysis scripys"
# echo  ~/ephys/split_concat.py "$1"  "$4"  --ow_flag "${6:-1}"
echo  ~/ephys/ephys_analysis_multisess.py config.yaml "$4" --sorter_dirname "${7:-'from_concat'}" --sess_top_filts "${8:-}"
python ~/ephys/split_concat.py "$1"  "$4"  --ow_flag "${6:-1}" --sess_top_filts "${8:-}"
python ~/ephys/ephys_analysis_multisess.py config.yaml "$4" --sorter_dirname "${7:-'from_concat'}" --sess_top_filts "${8:-}"