from pymc3 import *
import sys
sys.path.append('../kalman')
from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.linalg import solve_lyapunov as solve_continuous_lyapunov
except ImportError:  # pragma: no cover; github.com/scipy/scipy/pull/8082
    from scipy.linalg import solve_continuous_lyapunov

from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans

num_samples = 1000
num_cores = 6

def calc_y(x, alphas, betas, ws):
    y = 0.
    for i in np.arange(len(ws)):
        y += ws[i]*x**i
    for i in np.arange(len(alphas)):
        alpha = alphas[i]
        beta = betas[i]
        y += 1./(np.pi*beta*(1+((x-alpha)/beta)**2))
    return y

def loglik(y, y_true, sigma):
    loglik = -0.5 * np.sum((y - y_true)**2/sigma) - 0.5*np.log(sigma) - 0.5*np.log(2.0*np.pi)
    return loglik        
    
def bic(loglik, n, k):
    return np.log(n)*k-2*loglik
    
def mode_with_se(samples, num_bootstrap=100):
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

def find_local_maxima(x):
    maxima_inds = argrelextrema(x, np.greater_equal)[0]
    maxima_inds = maxima_inds[np.where(maxima_inds > 0)] # Omit leftmost point
    maxima_inds = maxima_inds[np.where(maxima_inds < len(x) - 1)] # Omit rightmost point
    filtered_maxima_inds = list()
    if len(maxima_inds) > 0:
        start = 0
        for i in np.arange(1, len(maxima_inds)):
            if maxima_inds[i] != maxima_inds[i-1] + 1:
                filtered_maxima_inds.append(int((maxima_inds[start] + maxima_inds[i-1]))/2)
                start = i
        filtered_maxima_inds.append(int((maxima_inds[start] + maxima_inds[-1]))/2)
    return np.asarray(filtered_maxima_inds, dtype='int')

def mode_samples(samples, mode):
    x_freqs = gaussian_kde(samples)
    x = np.linspace(min(samples), max(samples), 1000)
    local_maxima_inds = find_local_maxima(x_freqs(x))
    #print("local_maxima_inds", local_maxima_inds)
    x_kmeans = KMeans(n_clusters=len(local_maxima_inds)).fit(samples.reshape((-1, 1)))
    #print(np.array([mode]).reshape((-1, 1)))
    opt_x_label = x_kmeans.predict(np.array([mode]).reshape((-1, 1)))
    inds = np.where(x_kmeans.labels_ == opt_x_label)[0]
    samples_ = samples[inds]
    #print(len(samples), len(samples_))
    #print("samples_", samples_)
    #inds = np.searchsorted(samples, samples_)
    #inds[inds >= len(samples)] = len(samples) - 1
    return samples_, inds

dat = readsav("FT_kyo_kx0_00_12732.pow", idict=None, python_dict=False, uncompressed_file_name=None, verbose=False)
levels = np.linspace(np.min(np.log(dat.p_kyom_kx0))+2, np.max(np.log(dat.p_kyom_kx0))-2, 42)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.contour(dat.k_y, dat.nu, np.log(dat.p_kyom_kx0), levels=levels)

x = np.asarray(dat.nu, dtype='float')
k_index = np.min(np.where(dat.k_y >= 750)[0])
# Average over 3 neigbouring slices to reduce noise
y = np.log(dat.p_kyom_kx0[:, k_index-1] + dat.p_kyom_kx0[:, k_index] + dat.p_kyom_kx0[:, k_index+1]) - np.log(3)
y -= np.mean(y)
inds = np.where(x > 1.5)[0]
x = x[inds]
y = y[inds]
inds = np.where(x < 8.)[0]
x = x[inds]
y = y[inds]

noise_var = 0.

num_segments = 10
for i in np.arange(num_segments):
    start_index = int(i * len(y) / num_segments)
    end_index = int((i+1) * len(y) / num_segments)
    noise_var += np.var(y[start_index:end_index])
noise_var /= num_segments
sig_var = np.var(y) - noise_var
true_sigma = np.sqrt(noise_var)
num_w = 3

x_range = max(x) - min(x)
x_left = min(x)
#waics = []
min_bic = sys.float_info.max
for num_components in np.arange(3, 6):
    with Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
        alphas = []
        betas = []
        ws = []
        # Define priors
        for i in np.arange(num_components):
            sigma = x_range/3
            alphas.append(Normal('alpha' + str(i), x_left+x_range*(i+1)/(num_components+1), sigma=sigma))
            betas.append(HalfNormal('beta' + str(i), sigma=1.))
        for i in np.arange(num_w):
            ws.append(Normal('w' + str(i), 0., sigma=0.5))
        sigma = HalfNormal('sigma', sigma=true_sigma)
    
        # Define likelihood
        likelihood = Normal('y', mu=calc_y(x, alphas, betas, ws), sigma=sigma, observed=y)
    
        # Inference!
        trace = sample(num_samples, cores=num_cores) # draw 3000 posterior samples using NUTS sampling
    
        #w = waic(trace, model).WAIC
        #waics.append(w)
        #print("WAIC(", num_components, ")", w)
    
    #with Model() as model:
    #    # specify glm and pass in data. The resulting linear model, its likelihood and
    #    # and all its parameters are automatically added to our model.
    #    glm.GLM.from_formula('y ~ x', data)
    #    trace = sample(3000, cores=2) # draw 3000 posterior samples using NUTS sampling
    
    plt.figure(figsize=(7, 7))
    traceplot(trace[100:])
    plt.tight_layout()
    
    
    plt.figure(figsize=(7, 7))
    plt.plot(x, y, 'x', label='data')
    
    alphas_est = []
    betas_est = []
    ws_est = []
    for i in np.arange(num_components):
    
        alpha_samples = trace["alpha" + str(i), num_samples//2:]
    
        print("len(alpha_samples)", len(alpha_samples))
        print("alpha_samples range", np.min(alpha_samples), "-", np.max(alpha_samples))
        # Just in case we happen to have multimodal posteriors
        # Remove all samples belonging to the previously detected mode
        for j in np.arange(len(alphas_est)):
            alpha = alphas_est[j]
            if alpha >= np.min(alpha_samples) and alpha <= np.max(alpha_samples):
                mode_alphas, inds = mode_samples(alpha_samples, alpha)
                print("mode_alphas", len(mode_alphas))
                print("mode_alphas range", np.min(mode_alphas), "-", np.max(mode_alphas))
                mask = np.zeros(alpha_samples.shape,dtype=bool)
                mask[inds] = True
                print(len(mask[mask > 0]))
                print(len(inds))
                alpha_samples = alpha_samples[~mask]
    
        print("len(alpha_samples)", len(alpha_samples))
        print("alpha_samples range", np.min(alpha_samples), "-", np.max(alpha_samples))
        (alpha, alpha_se) = mode_with_se(alpha_samples)
        alphas_est.append(alpha)
    
        beta_samples = trace["beta" + str(i), num_samples//2:]
        # Remove all samples belonginh to the previously detected mode
        for j in np.arange(len(betas_est)):
            beta = betas_est[j]
            if beta >= np.min(beta_samples) and beta <= np.max(beta_samples):
                mode_betas, inds = mode_samples(beta_samples, beta)
                mask = np.zeros(beta_samples.shape,dtype=bool)
                mask[inds] = True
                beta_samples = beta_samples[~mask]
        (beta, beta_se) = mode_with_se(beta_samples)
        betas_est.append(beta)
    
    for i in np.arange(num_w):
        w_samples = trace["w" + str(i), num_samples//2:]
        ws_est.append(np.mean(w_samples))
    
    #plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')
    y_mean_est = calc_y(x, alphas_est, betas_est, ws_est)
    b = bic(loglik(y_mean_est, y, true_sigma), len(y), 2*num_components + num_w)
    print("BIC", b)
    plt.plot(x, y_mean_est, label='estimated regression line', lw=3., c='r')
    
    plt.title('Num. clusters ' + str(num_components))
    plt.legend(loc=0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    if b < min_bic:
        min_bic = b
        opt_alphas = alphas_est
        opt_betas = betas_est
        opt_ws = ws_est
        opt_num_components = num_components

plt.figure(figsize=(7, 7))
plt.plot(x, y, 'x', label='data')
#plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')
y_mean_est = calc_y(x, opt_alphas, opt_betas, opt_ws)
print("Lowest BIC", min_bic)
plt.plot(x, y_mean_est, label='estimated regression line', lw=3., c='r')
plt.title("Spectrum at k=" + str(dat.k_y[k_index]))
plt.legend(loc=0)
plt.xlabel(r'$\nu$')
plt.ylabel('Amplitude')
plt.show()

