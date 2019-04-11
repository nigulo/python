import os
os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=6


import numpy as np
import scipy.misc
import sys
sys.path.append('../utils')
import sampling

import pymc3 as pm
import pyhdust.triangle as triangle
import scipy.optimize
import matplotlib.pyplot as plt


class psf_basis_sampler():
    
    def __init__(self, psf_b, gamma, num_samples=10, num_chains = 0, full_posterior = False):
        self.psf_b = psf_b
        self.gamma = gamma
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.full_posterior = full_posterior

    def sample(self, D, D_d, plot_file = None):
    
        jmax = self.psf_b.jmax
        betas_est = np.zeros(jmax, dtype='complex')    
    
        s = sampling.Sampling()
        if self.full_posterior:
            betas_real = [None] * jmax
            betas_imag = [None] * jmax
            with s.get_model():
                for i in np.arange(0, jmax):
                    betas_real[i] = pm.Normal('beta_r' + str(i), sd=1.0)
                    betas_imag[i] = pm.Normal('beta_i' + str(i), sd=1.0)
    
            trace = s.sample(self.psf_b.likelihood, betas_real + betas_imag, [D, D_d, self.gamma], self.num_samples, self.num_chains, self.psf_b.likelihood_grad)
            samples = []
            var_names = []
            for i in np.arange(0, jmax):
                var_name_r = 'beta_r' + str(i)
                var_name_i = 'beta_i' + str(i)
                samples.append(trace[var_name_r])
                samples.append(trace[var_name_i])
                var_names.append(r"$\Re \beta_" + str(i) + r"$")
                var_names.append(r"$\Im \beta_" + str(i) + r"$")
                betas_est[i] = np.mean(trace[var_name_r]) + 1.j*np.mean(trace[var_name_i])
            
            samples = np.asarray(samples).T
            if plot_file is not None:
                fig, ax = plt.subplots(nrows=jmax*2, ncols=jmax*2)
                fig.set_size_inches(6*jmax, 6*jmax)
                triangle.corner(samples[:,:], labels=var_names, fig=fig)
                fig.savefig(plot_file)
    
        else:
            def lik_fn(params):
                return self.psf_b.likelihood(params, [D, D_d, self.gamma])
    
            def grad_fn(params):
                return self.psf_b.likelihood_grad(params, [D, D_d, self.gamma])
            
            
            # Optional methods:
            # Nelder-Mead Simplex algorithm (method='Nelder-Mead')
            # Broyden-Fletcher-Goldfarb-Shanno algorithm (method='BFGS')
            # Powell's method (method='powell')
            # Newton-Conjugate-Gradient algorithm (method='Newton-CG')
            # Trust-Region Newton-Conjugate-Gradient Algorithm (method='trust-ncg')
            # Trust-Region Truncated Generalized Lanczos / Conjugate Gradient Algorithm (method='trust-krylov')
            # Trust-Region Nearly Exact Algorithm (method='trust-exact')
            # Trust-Region Constrained Algorithm (method='trust-constr')
            # Sequential Least SQuares Programming (SLSQP) Algorithm (method='SLSQP')
            # Unconstrained minimization (method='brent')
            # Bounded minimization (method='bounded')
            #
            # Not all of them use the given gradient function.
    
            min_loglik = None
            min_res = None
            for trial_no in np.arange(0, self.num_samples):
                res = scipy.optimize.minimize(lik_fn, np.random.normal(size=jmax*2), method='BFGS', jac=grad_fn, options={'disp': True, 'gtol':1e-7})
                #lower_bounds = np.zeros(jmax*2)
                #upper_bounds = np.ones(jmax*2)*1e10
                #res = scipy.optimize.minimize(lik_fn, np.random.normal(size=jmax*2), method='L-BFGS-B', jac=grad_fn, bounds = zip(lower_bounds, upper_bounds), options={'disp': True, 'gtol':1e-7})
                loglik = res['fun']
                #assert(loglik == lik_fn(res['x']))
                if min_loglik is None or loglik < min_loglik:
                    min_loglik = loglik
                    min_res = res
            for i in np.arange(0, jmax):
                betas_est[i] = min_res['x'][i] + 1.j*min_res['x'][jmax + i]
            
        print(betas_est)
        #betas_est = np.random.normal(size=psf.jmax) + np.random.normal(size=psf.jmax)*1.j
        
        return betas_est
