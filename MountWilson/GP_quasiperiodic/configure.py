import pandas as pd
import sys
import numpy as np
import os.path

num_iters = 200
num_chains = 8
down_sample_factor = 1

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
    stars = data = np.genfromtxt("stars.txt", dtype=None, delimiter=' ')



output = open("run.sh", "w")

for star in bglst_cycles.keys():
    if len(stars) > 0:
        star_indices, = np.where(stars == star)
        if len(star_indices) == 0:
            continue
    if len(bglst_cycles[star]) <= peak_index:
        continue
    output.write("sbatch job.sh " + star + " " + str(peak_index) + " " + str(num_iters) + " " + str(num_chains) + " " + str(down_sample_factor) + "\n")

output.close()