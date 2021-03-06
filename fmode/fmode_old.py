import matplotlib as mpl
mpl.use('Agg')
from pymc3 import *
from sklearn.cluster import KMeans
import sys
import os
from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

import scipy.optimize

k_min = 700
k_max = sys.maxsize
if len(sys.argv) > 1:
    k_min = float(sys.argv[1])

if len(sys.argv) > 2:
    k_max = float(sys.argv[2])

num_samples = 1000
num_cores = 6
colors = ['blue', 'red', 'green', 'peru', 'purple']

sample_or_optimize = False

num_optimizations = 1

# F-mode = 0
# P modes = 1 ... 
def get_alpha_prior(i, k_y):
    assert(i >= 0 and i <= 3)
    g_sun=274.*1e-6 # Mm s^(-2)
    R_sun=696. # Mm
    A=g_sun/R_sun
    if i == 0:
        return (1000./(2*np.pi))*np.sqrt(A*k_y) #units=mHz
    else:
        return (1000./(2*np.pi))*np.sqrt((float(i) +.5)*A*k_y)

def calc_y(x, alphas, betas, ws, scale):
    y = 0.
    for i in np.arange(len(ws)):
        y += ws[i]*x**i
    for i in np.arange(len(alphas)):
        alpha = alphas[i]
        beta = betas[i]
        y += 1./(np.pi*beta*(1+((x-alpha)/beta)**2))
    return y*scale

def calc_loglik(y, y_true, sigma):
    loglik = -0.5 * np.sum((y - y_true)**2/sigma) - 0.5*np.log(sigma) - 0.5*np.log(2.0*np.pi)
    return loglik        
    
def bic(loglik, n, k):
    return np.log(n)*k - 2.*loglik
    
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

#def find_areas_analytical(x, alphas, betas, ws, scale, noise_std):
#    y_base = calc_y(x, [], [], ws, scale)
#    areas = []
#    for i in np.arange(len(alphas)):
#        y = calc_y(x, alphas[i:i+1], betas[i:i+1], ws, scale)
#        y -= y_base
#        inds = np.where(y > noise_std)[0]
#        y_gt_noise = y[inds]
#        x_gt_noise = x[inds]
#        x_left = np.min(x_gt_noise)
#        x_right = np.max(x_gt_noise)
#        areas.append((np.sum(y_gt_noise)*(x_right-x_left)/len(y_gt_noise), x_left, x_right))
#    return areas

def find_areas(x, y, alphas, betas, ws, scale, noise_std):
    ###########################################################################
    # Remove noise 
    #y_base = calc_y(x, [], [], ws, scale)
    #y -= y_base
    #inds = np.where(y > noise_std)[0]
    #y = y[inds]
    #x = x[inds]
    ###########################################################################
    y_base = calc_y(x, [], [], ws, scale)
    ys_fit = []
    num_components = len(alphas)
    for i in np.arange(num_components):
        y_fit = calc_y(x, alphas[i:i+1], betas[i:i+1], ws, scale)
        y_fit -= y_base
        ys_fit.append(y_fit)
    ys_fit = np.asarray(ys_fit)
    components_inds = np.argmax(ys_fit, axis=0)
    
    #print("components_inds", components_inds)
    areas = np.zeros(num_components)
    counts = np.zeros(num_components)
    ranges = np.zeros((num_components, 2))
    ranges[:, 0] = sys.float_info.max
    ranges[:, 1] = sys.float_info.min

    for i in np.arange(len(y)):
        component_ind = components_inds[i]
        areas[component_ind] += y[i]
        #######################################################################
        # Subtract contribution from polynomial base and other components
        areas[component_ind] -= y_base[i]
        for j in np.arange(num_components):
            if j != component_ind:
                areas[component_ind] -= ys_fit[j][i]
        #######################################################################
        counts[component_ind] += 1.
        # Update component ranges
        if x[i] < ranges[component_ind, 0]:
            ranges[component_ind, 0] = x[i]
        if x[i] > ranges[component_ind, 1]:
            ranges[component_ind, 1] = x[i]
        
    for i in np.arange(num_components):
        areas[i] *= (ranges[i, 1]-ranges[i, 0])/counts[i]
    
    return areas, ranges

def smooth(x=None, y=None):
    if x is not None:
        x = x[:-2]+x[1:-1]+x[2:]
        x /= 3
    
    if y is not None:
        y = y[:-2]+y[1:-1]+y[2:]
        y /= 3
    
    return x, y

    
def get_noise_var(dat):
    k_indices = np.where(np.logical_and(dat.k_y >= 3500, dat.k_y <= 4500))[0]
    nu_indices = np.where(np.logical_and(dat.nu >= 2, dat.nu <= 4))[0]
    y = dat.p_kyom_kx0[nu_indices]
    y = y[:, k_indices]

    _, y = smooth(None, y)
    return np.var(y)
    
for root, dirs, files in os.walk("data"):
    for file in files:
        if file[-4:] != ".pow":
            continue
        file_prefix = file[:-4]
        
        output_dir = "results/" + file_prefix
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        print("==================================================================")
        print(file_prefix)
        print("==================================================================")
        
    
        dat = readsav("data/" + file, idict=None, python_dict=False, uncompressed_file_name=None, verbose=False)
        levels = np.linspace(np.min(np.log(dat.p_kyom_kx0))+2, np.max(np.log(dat.p_kyom_kx0))-2, 42)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.contour(dat.k_y, dat.nu, np.log(dat.p_kyom_kx0), levels=levels)
        fig.savefig(output_dir + "/spectrum.png")
        plt.close(fig)
        
        f1 = open(output_dir + '/areas.txt', 'w')
        f1.write('k num_components f_mode_area\n')
        
        results = []
        k_index = np.min(np.where(dat.k_y >= k_min)[0])
        k_max_ = dat.k_y[dat.k_y < k_max][-1]
        while k_index < dat.p_kyom_kx0.shape[1]:
            y = dat.p_kyom_kx0[:, k_index]
            
            if np.min(dat.k_y[k_index:]) > k_max_:
                break
            
            print("------------------------------------------------------------------")
            print("Fitting for k = " + str(dat.k_y[k_index]))
            print("------------------------------------------------------------------")
            #y = dat.p_kyom_kx0[:, k_index-1] + dat.p_kyom_kx0[:, k_index] + dat.p_kyom_kx0[:, k_index+1]
            # Average over 3 neigbouring slices to reduce noise
            #y = np.log(dat.p_kyom_kx0[:, k_index-1] + dat.p_kyom_kx0[:, k_index] + dat.p_kyom_kx0[:, k_index+1]) - np.log(3)
            #y = y - np.mean(y)
            
            x = np.asarray(dat.nu, dtype='float')
            inds = np.where(x > 2.)[0]
            x = x[inds]
            y = y[inds]
            inds = np.where(x < 10.)[0]
            x = x[inds]
            y = y[inds]

            x, y = smooth(x, y)
            
            #noise_var = 0.
            #num_segments = 10
            #for i in np.arange(num_segments):
            #    start_index = int(i * len(y) / num_segments)
            #    end_index = int((i+1) * len(y) / num_segments)
            #    noise_var += np.var(y[start_index:end_index])
            #noise_var /= num_segments
            #noise_var /= 2
            
            noise_var = get_noise_var(dat)
            
            sig_var = np.var(y) - noise_var
            true_sigma = np.sqrt(noise_var)
            print("noise_std", true_sigma)
            num_w = 1
            
            x_range = max(x) - min(x)
            x_left = min(x)
            #waics = []
            min_bic = sys.float_info.max

            for num_components in np.arange(1, 5):
                scale =  np.sum(y)*x_range/len(y)/num_components
                print("scale", scale)
                
                alphas_est = []
                betas_est = []
                ws_est = []
                
                ###########################################################################
                # Sampling
                ###########################################################################
                if sample_or_optimize:
                    with Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
                        alphas = []
                        betas = []
                        ws = []
                        # Define priors
                        for i in np.arange(num_components):
                            sigma = x_range/3/num_components
                            alpha_prior = get_alpha_prior(i, dat.k_y[k_index])
                            print("alpha_prior", alpha_prior, i)
                            #alphas.append(Normal('alpha' + str(i), x_left+x_range*(i+1)/(num_components+1), sigma=sigma))
                            alphas.append(Normal('alpha' + str(i), alpha_prior, sigma=sigma))
                            betas.append(HalfNormal('beta' + str(i), sigma=1./num_components))
                        for i in np.arange(num_w):
                            ws.append(Normal('w' + str(i), 0., sigma=1.))
                        sigma = HalfNormal('sigma', sigma=true_sigma)
                    
                        # Define likelihood
                        likelihood = Normal('y', mu=calc_y(x, alphas, betas, ws, scale), sigma=sigma, observed=y)
                    
                        # Inference!
                        trace = sample(num_samples, cores=num_cores) # draw 3000 posterior samples using NUTS sampling
                    
                    for i in np.arange(num_components):
                        print("--------------------------------------------------------------")
                        print("Component", i)
                        alpha_samples = trace["alpha" + str(i), :]#num_samples//2:]
                    
                        #print("len(alpha_samples)", len(alpha_samples))
                        #print("alpha_samples range", np.min(alpha_samples), "-", np.max(alpha_samples))
                        # Just in case we happen to have multimodal posteriors
                        # Remove all samples belonging to the previously detected mode
                        for j in np.arange(len(alphas_est)):
                            alpha = alphas_est[j]
                            if alpha >= np.min(alpha_samples) and alpha <= np.max(alpha_samples):
                                mode_alphas, inds = mode_samples(alpha_samples, alpha)
                                #print("mode_alphas", len(mode_alphas))
                                #print("mode_alphas range", np.min(mode_alphas), "-", np.max(mode_alphas))
                                mask = np.zeros(alpha_samples.shape,dtype=bool)
                                mask[inds] = True
                                #print(len(mask[mask > 0]))
                                #print(len(inds))
                                alpha_samples = alpha_samples[~mask]
                        print("len(alpha_samples)", len(alpha_samples))
                        print("alpha_samples range", np.min(alpha_samples), "-", np.max(alpha_samples))
                        (alpha, alpha_se) = mode_with_se(alpha_samples)
                        alphas_est.append(alpha)
                    
                        beta_samples = trace["beta" + str(i), :]#num_samples//2:]
                        #print("len(beta_samples)", len(beta_samples))
                        #print("beta_samples range", np.min(beta_samples), "-", np.max(beta_samples))
                        ## Remove all samples belonginh to the previously detected mode
                        #for j in np.arange(len(betas_est)):
                        #    beta = betas_est[j]
                        #    if beta >= np.min(beta_samples) and beta <= np.max(beta_samples):
                        #        mode_betas, inds = mode_samples(beta_samples, beta)
                        #        #print("mode_betas", len(mode_betas))
                        #        #print("mode_betas range", np.min(mode_betas), "-", np.max(mode_betas))
                        #        mask = np.zeros(beta_samples.shape,dtype=bool)
                        #        mask[inds] = True
                        #        beta_samples = beta_samples[~mask]
                        #print("len(beta_samples)", len(beta_samples))
                        #print("beta_samples range", np.min(beta_samples), "-", np.max(beta_samples))
                        (beta, beta_se) = mode_with_se(beta_samples)
                        betas_est.append(beta)
                    
                    for i in np.arange(num_w):
                        w_samples = trace["w" + str(i), num_samples//2:]
                        ws_est.append(np.mean(w_samples))
                ###########################################################################
                # Optimization
                ###########################################################################
                else:
                    def lik_fn(params):
                        alphas = params[:num_components]
                        betas = params[num_components:2*num_components]
                        ws = params[2*num_components:2*num_components+num_w]
                        
                        y_mean_est=calc_y(x, alphas, betas, ws, scale)
                        return -calc_loglik(y_mean_est, y, true_sigma)
            
                    #def grad_fn(params):
                    #    return self.psf.likelihood_grad(params, [Ds, self.gamma])
                    
                    min_loglik = None
                    min_res = None
                    for trial_no in np.arange(0, num_optimizations):
                        params = []
                        bounds = []
                        for i in np.arange(num_components):
                            alpha_prior = get_alpha_prior(i, dat.k_y[k_index])
                            params.append(alpha_prior)
                            print("alpha_prior", alpha_prior, i)
                            bounds.append((alpha_prior-.5 , alpha_prior+.5))
                        for i in np.arange(num_components):
                            beta_prior = .2/num_components
                            params.append(beta_prior)
                            #params.append(1./100)
                            bounds.append((.0001 , 2*beta_prior))
                        for i in np.arange(num_w):
                            params.append(0.)
                            bounds.append((-100 , 100))
                            
                        print("params", params)
                            
                        initial_lik = lik_fn(params)
                        #res = scipy.optimize.minimize(lik_fn, params, method='CG', jac=None, options={'disp': True, 'gtol':initial_lik*1e-7})#, 'eps':.1})
                        res = scipy.optimize.minimize(lik_fn, params, method='L-BFGS-B', jac=None, bounds=bounds, options={'disp': True, 'gtol':1e-7})
                        loglik = res['fun']
                        if min_loglik is None or loglik < min_loglik:
                            min_loglik = loglik
                            min_res = res['x']
            
                    for i in np.arange(num_components):
                        alphas_est.append(min_res[i])
                        betas_est.append(min_res[i+num_components])
                    for i in np.arange(num_w):
                        ws_est.append(min_res[i+2*num_components])
                ###########################################################################
                    
                
                #plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')
                y_mean_est = calc_y(x, alphas_est, betas_est, ws_est, scale)
                b = bic(calc_loglik(y_mean_est, y, true_sigma), len(y), 2*num_components + num_w)
                print("BIC", b)
                
                #fig, ax = plt.subplots(nrows=1, ncols=1)
                #ax.plot(x, y, 'x', label='data')
                #ax.plot(x, y_mean_est, label='estimated regression line', lw=3., c='r')
                #ax.set_title('Num. clusters ' + str(num_components))
                #ax.legend(loc=0)
                #ax.set_xlabel('x')
                #ax.set_ylabel('y')
                #fig.savefig(output_dir + "/fit" + str(k_index) + "_" + str(num_components) + ".png")
                #plt.close(fig)
                
                if b < min_bic:
                    min_bic = b
                    opt_alphas = alphas_est
                    opt_betas = betas_est
                    opt_ws = ws_est
                    opt_num_components = num_components
            
            scale =  np.sum(y)*x_range/len(y)/opt_num_components
            
            print("alphas", opt_alphas)
            print("betas", opt_betas)
            print("ws", opt_ws)
            
            fig, ax = plt.subplots(nrows=1, ncols=1)
            plt.figure(figsize=(7, 7))
            ax.plot(x, y, 'x', label='data')
            #plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')
            y_mean_est = calc_y(x, opt_alphas, opt_betas, opt_ws, scale)
            ax.plot(x, y_mean_est, label='estimated regression line', lw=3., c='r')
            
            #areas = find_areas_analytical(x, opt_alphas, opt_betas, opt_ws, scale, true_sigma)
            #for i in np.arange(len(areas)):
            #    area, x_left, x_right = areas[i]
            #    ax.axvspan(x_left, x_right, alpha=0.5, color=colors[i])
            
            areas, ranges = find_areas(x, y, opt_alphas, opt_betas, opt_ws, scale, true_sigma)
            for i in np.arange(len(areas)):
                ax.axvspan(ranges[i, 0], ranges[i, 1], alpha=0.5, color=colors[i])
            
            k_value = dat.k_y[k_index]
            ax.set_title("Spectrum at k=" + str(k_value) + ", num. components=" + str(opt_num_components))
            ax.legend(loc=0)
            ax.set_xlabel(r'$\nu$')
            ax.set_ylabel('Amplitude')
            fig.savefig(output_dir + "/areas" + str(k_index) + ".png")

            plt.close(fig)
            
            f1.write('%s %s %s' % (str(k_value), opt_num_components, areas[0]) + "\n")
            print("Lowest BIC", min_bic)
            print("Num components", opt_num_components)
            print("Areas", areas)
            f1.flush()
            results.append([k_value, opt_num_components, areas[0]])
        
            k_index += 1
        
        f1.close()
        
        results = np.asarray(results)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.figure(figsize=(7, 7))
        ax.plot(results[:, 0], results[:, 2], 'k-')
        ax.set_xlabel(r'$k$')
        ax.set_ylabel('F-mode area')
        fig.savefig(output_dir + "/areas.png")

        plt.close(fig)
