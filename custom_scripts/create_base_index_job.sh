#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --gres gpu:1
#SBATCH --qos=vesta
#SBATCH --partition=volta

############################
# This hjob script can be executed in place of step 3/4 in `scripts/mine_sequences.py`.
# If runs a seperate script to create the base index using SLURM rather than SUBMITIT.
############################

module load volta anaconda3
module list
echo "activating conda env"
source activate muss

resource_monitor -O  create_index_job -i 600 -- python scripts/create_base_index.py en