import pandas as pd
import sys
import numpy as np
import os.path

num_iters = 200
num_chains = 8
down_sample_factor = 1
queue = "longrun"
time_limit = "336:00:00"

peak_index = 0
if len(sys.argv) > 1:
    peak_index = int(sys.argv[1])


def load_BGLST_results():
    data = pd.read_csv("BGLST_results.txt", names=['star', 'cyc', 'sigma', 'normality', 'bic'], header=0, dtype=None, sep='\s+', engine='python').as_matrix()
    bglst_cycles = dict()
    for [star, cyc, std, normality, bic] in data:
        if not bglst_cycles.has_key(star):
            bglst_cycles[star] = list()
        all_cycles = bglst_cycles[star]
        cycles = list()
        if not np.isnan(cyc):
            cycles.append(cyc)
            cycles.append(std)
        all_cycles.append(np.asarray(cycles))
    return bglst_cycles

bglst_cycles = load_BGLST_results()

stars = np.array([])
if os.path.isfile("stars.txt"):
    #stars_to_recalculate = pd.read_csv("GPR_stan_stars.txt", names=['star'], dtype=None, sep='\s+', engine='python').as_matrix()
    stars = np.genfromtxt("stars.txt", dtype='str', delimiter=' ')


output = open("run.sh", "w")
output.write("module load python-env/2.7.13\n")
output.write("module load openblas\n")
for star in bglst_cycles.keys():
    if len(stars) > 0:
        star_indices, = np.where(stars == star)
        if len(star_indices) == 0:
            continue
    if len(bglst_cycles[star]) <= peak_index:
        continue
    output.write("sbatch job.sh " + star + " " + str(peak_index) + " " + str(num_iters) + " " + str(num_chains) + " " + str(down_sample_factor) + "\n")

output.close()

output = open("job.sh", "w")

output.write(
"""#!/bin/bash -l
###
### parallel job script example
###
## name of your job
#SBATCH -J GP_quasi
## system error message output file
#SBATCH -e output_err_%j
## system message output file
#SBATCH -o output_%j
## a per-process (soft) memory limit
## limit is specified in MB
## example: 1 GB is 1000
#SBATCH --mem-per-cpu=8000
## how long a job takes, wallclock time hh:mm:ss
""" +
"#SBATCH -t " + time_limit +
"""
##the number of processes (number of cores)
##SBATCH -N 1
#SBATCH -n 1
""" + 
"#SBATCH --cpus-per-task=" + str(num_chains) +
"""
## queue name
""" + 
"#SBATCH -p " + queue + 
"""
## run my MPI executable
export KMP_AFFINITY=compact
export KMP_DETERMINISTIC_REDUCTION=yes    #(if necessary and intel compiler version is 13 or later)
""" +
"export OMP_NUM_THREADS=" + str(num_chains) +
"""
srun python GPR_stan.py $1 $2 $3 $4 $5
""")

output.close()
