import numpy as np
import scipy.misc
import sys
sys.path.append('../utils')
import sampling

import pymc3 as pm
import pyhdust.triangle as triangle
import scipy.optimize
import matplotlib.pyplot as plt


class psf_sampler():
    
    def __init__(self, psf, gamma, num_samples=10, num_chains = 0, full_posterior = False):
        self.psf = psf
        self.gamma = gamma
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.full_posterior = full_posterior

    def sample(self, D, D_d, plot_file = None):
    
        jmax = self.psf.coh_trans_func.phase_aberr.jmax
        alphas_est = np.zeros(jmax)    
    
        s = sampling.Sampling()
        if self.full_posterior:
            alphas = [None] * jmax
            with s.get_model():
                for i in np.arange(0, jmax):
                    alphas[i] = pm.Normal('alpha' + str(i), sd=1.0)
    
            trace = s.sample(self.psf.likelihood, alphas, [D, D_d, self.gamma], self.num_samples, self.num_chains, self.psf.likelihood_grad)
            samples = []
            var_names = []
            for i in np.arange(0, jmax):
                var_name = 'alpha' + str(i)
                samples.append(trace[var_name])
                var_names.append(r"$\alpha_" + str(i) + r"$")
                alphas_est[i] = np.mean(trace[var_name])
            
            samples = np.asarray(samples).T
            if plot_file is not None:
                fig, ax = plt.subplots(nrows=jmax, ncols=jmax)
                fig.set_size_inches(3*jmax, 3*jmax)
                triangle.corner(samples[:,:], labels=var_names, fig=fig)
                fig.savefig(plot_file)
    
        else:
            def lik_fn(params):
                return self.psf.likelihood(params, [D, D_d, self.gamma])
    
            def grad_fn(params):
                return self.psf.likelihood_grad(params, [D, D_d, self.gamma])
            
            min_loglik = None
            min_res = None
            if False:
                for trial_no in np.arange(0, self.num_samples):
                    res = scipy.optimize.minimize(lik_fn, np.random.normal(size=jmax), method='BFGS', jac=grad_fn, options={'disp': True})
                    loglik = res['fun']
                    #assert(loglik == lik_fn(res['x']))
                    if min_loglik is None or loglik < min_loglik:
                        min_loglik = loglik
                        min_res = res
                for i in np.arange(0, jmax):
                    alphas_est[i] = min_res['x'][i]
            else:
                for trial_no in np.arange(0, self.num_samples):
                    res = scipy.optimize.fmin_cg(lik_fn, np.random.normal(size=jmax), fprime=grad_fn, args=(), full_output=True)
                    #lower_bounds = np.zeros(jmax*2)
                    #upper_bounds = np.ones(jmax*2)*1e10
                    #res = scipy.optimize.minimize(lik_fn, np.random.normal(size=jmax*2), method='L-BFGS-B', jac=grad_fn, bounds = zip(lower_bounds, upper_bounds), options={'disp': True, 'gtol':1e-7})
                    loglik = res[1]
                    #assert(loglik == lik_fn(res['x']))
                    if min_loglik is None or loglik < min_loglik:
                        min_loglik = loglik
                        min_res = res
                for i in np.arange(0, jmax):
                    alphas_est[i] = min_res[0][i]
            
        print(alphas_est)
        #betas_est = np.random.normal(size=psf.jmax) + np.random.normal(size=psf.jmax)*1.j
        
        return alphas_est
