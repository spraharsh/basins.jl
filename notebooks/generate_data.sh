#!/bin/bash
#SBATCH --job-name=blah_generate_data            # Job name
#SBATCH --output=blah_generate_data_%A_%a.out    # Output log file
#SBATCH --error=blah_generate_data_%A_%a.err     # Error log file
#SBATCH --array=1-100                       # Array of 100 jobs
#SBATCH --time=0-02:00:00                   # Maximum runtime (2 days)
#SBATCH --mem=8G                            # Memory allocation per job (4 GB)

# Capture natoms from the command line argument
NATOMS=$1

# Limit OpenMP to one thread for each task
export OMP_NUM_THREADS=1;

# Execute the Julia script using Singularity, passing natoms and task ID
singularity exec --overlay /scratch/ps4586/conda/overlay-10GB-400K.ext3:ro \
    /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
    /bin/bash -c "
    source ~/.bashrc;
    export OMP_NUM_THREADS=1;
    conda activate cb3-3.9;                  # Activate the conda environment
    /ext3/.juliaup/bin/julia generate_data.jl $SLURM_ARRAY_TASK_ID $NATOMS  # Run Julia script with task ID and natoms
    "