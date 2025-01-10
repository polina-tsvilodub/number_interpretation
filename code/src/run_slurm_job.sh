#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
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
conda activate llmlink

echo "Conda environment activated:"
echo $(conda env list)
echo " "
echo "Python version:"
echo $(which python)
echo " "

# activate CUDA
module load devel/cuda/11.6

# iterate over models
models=("gpt-4o-mini") # "google/gemma-1.1-7b-it" "meta-llama/Llama-3.3-70B-Instruct" "allenai/OLMo-2-1124-13B-Instruct" "gemini-1.5-pro" "claude-3-5-sonnet-20241022")

for i in ${!models[*]}; do
    echo "model: ${models[$i]}"
    python3 -u evaluate_llm_hyperbole.py \
        --model="${models[$i]}" \
        --expt_num="1a" \
        --num=10
done