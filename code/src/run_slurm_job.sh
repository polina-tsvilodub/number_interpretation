#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --mem=20gb

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
tasks=("1b") # "1b" "2" "3a" "3b")
# iterate over models
models=("meta-llama/Llama-3.1-8B-Instruct") # "claude-3-5-sonnet-20241022" "google/gemma-1.1-7b-it" "allenai/OLMo-2-1124-13B-Instruct" "gemini-1.5-pro" "meta-llama/Llama-3.3-70B-Instruct")

for i in ${!models[*]}; do
    for j in ${!tasks[*]}; do
        echo "model: ${models[$i]}"
        echo "task: ${tasks[$j]}"
        python3 -u evaluate_log_p_hyperbole.py \
            --model="${models[$i]}" 
            --expt_num="${tasks[$j]}" \
            --num=1
    done
done
