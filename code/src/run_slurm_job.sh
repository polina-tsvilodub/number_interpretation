#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=20gb
#SBATCH --gres=gpu:A40:1

echo 'Running simulation'

# The directory name is the 
# first command line argument
dir_name=$1

# Now, change to that directory 
# (assuming it's a subdirectory of 
# the parent directory of the current script)

# print the current working directory
# (should be the directory of the script)
echo "Current working directory:"
echo $(pwd)

# Load conda
module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda deactivate
# Activate the conda environment
conda activate griceChain

echo "Conda environment activated:"
echo $(conda env list)
echo " "
echo "Python version:"
echo $(which python)
echo " "

# activate CUDA
module load devel/cuda/11.6

# iterate over tasks
tasks=("2" "3a" "3b") # "1b" "2" "3a" "3b")
# iterate over models
models=("claude-3-5-sonnet-20241022" "gemini-1.5-pro" "gpt-4o-mini")

for i in ${!models[*]}; do
    for j in ${!tasks[*]}; do
        echo "model: ${models[$i]}"
        echo "task: ${tasks[$j]}"
        python3 -u evaluate_log_p_hyperbole.py \
            --model="${models[$i]}" \
            --expt_num="${tasks[$j]}" \
            --num=1
    done
done
