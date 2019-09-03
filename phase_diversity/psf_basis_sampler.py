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

    def sample(self, Ds, plot_file = None):
        L = Ds.shape[0]
        jmax = self.psf_b.jmax
        betas_est = np.zeros((L, jmax), dtype='complex')    

        tt = self.psf_b.tip_tilt
        if tt is not None:
            a_est = np.zeros(((L+1), 2))

        s = sampling.Sampling()
        if self.full_posterior:
            betas = [None] * 2*L*jmax
            with s.get_model():
                for i in np.arange(0, len(betas)):
                    betas[i] = pm.Normal('beta' + str(i), sd=1.0)
            if tt is not None:
                a = [None] * 2*(L+1)
                for l in np.arange(0, len(a)):
                    a[l] = pm.Normal('a' + str(l), sd=1.0)
            else:
                a = None
    
            trace = s.sample(self.psf_b.likelihood, [betas, a], [Ds, self.gamma], self.num_samples, self.num_chains, self.psf_b.likelihood_grad)
            samples = []
            var_names = []
            for l in np.arange(0, L):
                for i in np.arange(0, jmax):
                    var_name_r = 'beta' + str(l*2*jmax + i)
                    var_name_i = 'beta' + str(l*2*jmax + jmax + i)
                    samples.append(trace[var_name_r])
                    samples.append(trace[var_name_i])
                    var_names.append(r"$\Re \beta_" + str(i) + r"$")
                    var_names.append(r"$\Im \beta_" + str(i) + r"$")
                    betas_est[l, i] = np.mean(trace[var_name_r]) + 1.j*np.mean(trace[var_name_i])
            if tt is not None:
                for i in np.arange(0, 2*L):
                    var_name = 'a' + str(i)
                    samples.append(trace[var_name])
                    var_names.append(r"$a_" + str(i) + r"$")
                    a_est[i] = np.mean(trace[var_name])

            
            samples = np.asarray(samples).T
            if plot_file is not None:
                fig, ax = plt.subplots(nrows=jmax*2, ncols=jmax*2)
                fig.set_size_inches(6*jmax, 6*jmax)
                triangle.corner(samples[:,:], labels=var_names, fig=fig)
                fig.savefig(plot_file)
    
        else:
            def lik_fn(params):
                data = self.psf_b.encode_data(Ds, self.gamma)
                return self.psf_b.likelihood(params, data)
    
            def grad_fn(params):
                data = self.psf_b.encode_data(Ds, self.gamma)
                return self.psf_b.likelihood_grad(params, data)
            
            
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
                initial_a = np.array([])
                if tt is not None:
                    initial_a = np.zeros(((L+1), 2))
                    #initial_a = np.random.normal(size=((L+1), 2), scale=1./np.sqrt(tt.prior_prec + 1e-10))#np.zeros(2*self.L)
                #res = scipy.optimize.minimize(lik_fn, np.random.normal(size=jmax*2), method='BFGS', jac=grad_fn, options={'disp': True, 'gtol':1e-7})
                #initial_betas = np.random.normal(size=(L, jmax)) + 1.j*np.random.normal(size=(L, jmax))
                initial_betas = np.zeros((L, jmax), dtype='complex')
                params = self.psf_b.encode_params(initial_betas, initial_a)
                initial_lik = lik_fn(params)
                #res = scipy.optimize.fmin_cg(lik_fn, params, fprime=grad_fn, args=(), full_output=True)
                #res = scipy.optimize.fmin_bfgs(lik_fn, params, fprime=grad_fn, args=(), full_output=True)
                #lower_bounds = np.zeros(jmax*2)
                #upper_bounds = np.ones(jmax*2)*1e10
                res = scipy.optimize.minimize(lik_fn, params, method='CG', jac=grad_fn, options={'disp': True, 'gtol':initial_lik*1e-7})#, 'eps':.1})
                print(res)
                print("Optimization result:" + res["message"])
                print("Status", res['status'])
                print("Success", res['success'])
                loglik=res['fun']
                #assert(loglik == lik_fn(res['x']))
                if min_loglik is None or loglik < min_loglik:
                    min_loglik = loglik
                    min_res = res['x']
            for l in np.arange(0, L):
                for i in np.arange(0, jmax):
                    betas_est[l, i] = min_res[l*2*jmax + i] + 1.j*min_res[l*2*jmax + jmax + i]
            if tt is not None:
                a_est = min_res[L*2*jmax:L*2*jmax+(2*(L+1))].reshape((L+1, 2))
            
        print("betas_est", betas_est)
        if tt is not None:
            print("a_est", a_est)
        #betas_est = np.random.normal(size=psf.jmax) + np.random.normal(size=psf.jmax)*1.j
        if tt is not None:
            return (betas_est, a_est)
        else:
            return betas_est
