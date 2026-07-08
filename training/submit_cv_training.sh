#!/bin/bash
#SBATCH -J vessel_cv              #Job name
#SBATCH -n 1                      #Total number of tasks
#SBATCH -G 1                      #Number of GPUs
#SBATCH -c 4                      #Threads per process (4, since data loading/
                                   #augmentation benefits from a bit of CPU
                                   #parallelism -- bump if you have many more
                                   #slides and tiling feels slow)
#SBATCH -t 12:00:00                #Walltime limit -- 5 folds x (Phase1+Phase2)
                                   #plus one final full-data training run;
                                   #adjust based on how long your interactive
                                   #notebook run took for ONE fold
#SBATCH -o slurm-%j.out-%N        #Standard output
#SBATCH -e slurm-%j.err-%N        #Standard error
#SBATCH -p hotel-gpu              #Partition name
#SBATCH -q hotel-gpu              #QOS name
#SBATCH -A htl158                 #Allocation name
#SBATCH --mem=100G
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user ssharvey@health.ucsd.edu   # <-- change to your real email

# Activate conda the same robust way setup_environment.sh does -- works
# whether or not `conda init` has touched this particular shell.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vessel-seg

# Run from the directory containing train.py and the data/ folder.
# Change this to wherever you actually put the pipeline + data on TSCC.
cd ~/vessel_seg

python train.py
