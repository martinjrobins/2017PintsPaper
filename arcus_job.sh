#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set the partition where the job will run
#SBATCH --partition=devel

#SBATCH --cpus-per-task=8

# set max wallclock time
#SBATCH --time=0:10:00

# set name of job
#SBATCH --job-name=pints_matrix

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=martin.robinson@cs.ox.ac.uk

# Set OMP_NUM_THREADS to the same value as -c
# with a fallback in case it isn't set.
# SLURM_CPUS_PER_TASK is set to the value of -c, but only if -c is explicitly set

#module unload intel-mkl
#module unload intel-compilers
module load python/3.5
pip install --user ./pints

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    omp_threads=$SLURM_CPUS_PER_TASK
else
    omp_threads=1
fi
export set OMP_NUM_THREADS=$omp_threads

python3 main.py --integer ${SLURM_ARRAY_TASK_ID}
