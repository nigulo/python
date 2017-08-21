#!/bin/bash -l
###
### parallel job script example
###
## name of your job
#SBATCH -J GPR_stan
## system error message output file
#SBATCH -e output_err_%j
## system message output file
#SBATCH -o output_%j
## a per-process (soft) memory limit
## limit is specified in MB
## example: 1 GB is 1000
#SBATCH --mem-per-cpu=1000
## how long a job takes, wallclock time hh:mm:ss
#SBATCH -t 3:00:00
##the number of processes (number of cores)
##SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
## queue name
#SBATCH -p parallel
## run my MPI executable
export KMP_AFFINITY=compact
export KMP_DETERMINISTIC_REDUCTION=yes    #(if necessary and intel compiler version is 13 or later)
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WRKDIR}/libs/opencv/lib:${WRKDIR}/cpp/utils:${WRKDIR}/cpp/pcdl
srun python test_QP.py $1
