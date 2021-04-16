import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
from scipy.stats import gaussian_kde

# Two random Gaussians as data
data1 = random.normal(size=100)
data2 = random.normal(size=100)+5.
data = np.concatenate((data1, data2))

sigma_start = 0.01 # Minimum kernel width to try
sigma_end = 0.5 # Maximum kernel to try
num_sigmas = 100 # Number of kernel widths to try

num_cv_sets = 10 # 10-fold cross-validation

def calc_kde(data):
    max_lik = None
    opt_sigma = None
    set_len = int(len(data) / num_cv_sets)
    for sigma in np.linspace(sigma_start, sigma_end, num=num_sigmas):
        log_lik = 0
        for i in np.arange(0, num_cv_sets):
            # Training data
            train_data = np.concatenate((data[:i*set_len], data[(i+1)*set_len:]))
            # Validation data
            if i == num_cv_sets - 1:
                valid_data = data[i*set_len:]
            else:
                valid_data = data[i*set_len:(i+1)*set_len]

            density = gaussian_kde(train_data, bw_method = sigma)
            d = density(valid_data)
            log_lik += np.sum(np.log(d))
        if max_lik is None or log_lik > max_lik:
            max_lik = log_lik
            opt_sigma = sigma
    return gaussian_kde(data, bw_method = opt_sigma), opt_sigma

x_grid = np.linspace(min(data), max(data), 500)
density, sigma = calc_kde(data)
d = density(x_grid)
print("Optimal kernel width: %f" % (sigma))

fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(6, 4)


hist, bin_edges = np.histogram(data, bins = 20)
ax.bar(bin_edges[:-1], hist)

# The area under density is unity, so normaize it
# to the scame scale as the histogram
d *= max(hist)/max(d)
ax.plot(x_grid, d, "r-")

plt.show()
