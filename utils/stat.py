import numpy as np
from scipy.stats import gaussian_kde

def estimate_with_se(x, f, num_bootstrap=1000):
    estimate = f(x)
    bs_estimates = np.zeros(num_bootstrap)
    for j in np.arange(0, num_bootstrap):
        x_bs = x[np.random.choice(np.shape(x)[0], np.shape(x)[0], replace=True, p=None)]
        bs_estimates[j] = f(x_bs)
    return (estimate, np.std(bs_estimates))

def mean_with_se(x, num_bootstrap=1000):
    mean = np.mean(x)
    bs_means = np.zeros(num_bootstrap)
    for j in np.arange(0, num_bootstrap):
        x_bs = x[np.random.choice(np.shape(x)[0], np.shape(x)[0], replace=True, p=None)]
        bs_means[j] = np.mean(x_bs)
    return (mean, np.std(bs_means))
    
def mode_with_se(samples, num_bootstrap=1000):
    x_freqs = gaussian_kde(samples)
    x = np.linspace(min(samples), max(samples), 1000)
    #density.covariance_factor = lambda : .25
    #density._compute_covariance()
    mode = x[np.argmax(x_freqs(x))]
    bs_modes = np.zeros(num_bootstrap)
    for j in np.arange(0, num_bootstrap):
        samples_bs = x[np.random.choice(np.shape(x)[0], np.shape(x)[0], replace=True, p=None)]
        x_bs_freqs = gaussian_kde(samples_bs)
        x_bs = np.linspace(min(samples_bs), max(samples_bs), 1000)
        #density.covariance_factor = lambda : .25
        #density._compute_covariance()
        bs_modes[j] = x_bs[np.argmax(x_bs_freqs(x_bs))]
    return (mode, np.std(bs_modes))

