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

    def sample(self, Ds, plot_file = None):
        L = Ds.shape[0]
        jmax = self.psf.coh_trans_func.phase_aberr.jmax
        alphas_est = np.zeros((L, jmax))    
    
        tt = self.psf.tip_tilt
        if tt is not None:
            a_est = np.zeros(((L+1), 2))
    
        s = sampling.Sampling()
        if self.full_posterior:
            alphas = [None] * L * jmax
            with s.get_model():
                for i in np.arange(0, len(alphas)):
                    alphas[i] = pm.Normal('alpha' + str(i), sd=1.0)
    
            if tt is not None:
                a = [None] * 2*(L+1)
                for l in np.arange(0, len(a)):
                    a[l] = pm.Normal('a' + str(l), sd=1.0)
            else:
                a = None
    
            trace = s.sample(self.psf.likelihood, alphas, [Ds, self.gamma], self.num_samples, self.num_chains, self.psf.likelihood_grad)
            samples = []
            var_names = []
            for l in np.arange(0, L):
                for i in np.arange(0, jmax):
                    var_name = 'alpha' + str(l*jmax + i)
                    samples.append(trace[var_name])
                    var_names.append(r"$\alpha_" + str(i) + r"$")
                    alphas_est[l, i] = np.mean(trace[var_name])
            if tt is not None:
                for i in np.arange(0, 2*L):
                    var_name = 'a' + str(i)
                    samples.append(trace[var_name])
                    var_names.append(r"$a_" + str(i) + r"$")
                    a_est[i] = np.mean(trace[var_name])
            
            
            samples = np.asarray(samples).T
            if plot_file is not None:
                fig, ax = plt.subplots(nrows=jmax, ncols=jmax)
                fig.set_size_inches(3*jmax, 3*jmax)
                triangle.corner(samples[:,:], labels=var_names, fig=fig)
                fig.savefig(plot_file)
    
        else:
            def lik_fn(params):
                return self.psf.likelihood(params, [Ds, self.gamma])
    
            def grad_fn(params):
                return self.psf.likelihood_grad(params, [Ds, self.gamma])
            
            min_loglik = None
            min_res = None
            for trial_no in np.arange(0, self.num_samples):
                initial_a = np.array([])
                if tt is not None:
                    initial_a = np.zeros(((L+1), 2))
                    #initial_a = np.random.normal(size=((L+1), 2), scale=1./np.sqrt(tt.prior_prec + 1e-10))#np.zeros(2*self.L)
                #res = scipy.optimize.minimize(lik_fn, np.random.normal(size=jmax*2), method='BFGS', jac=grad_fn, options={'disp': True, 'gtol':1e-7})
                #initial_betas = np.random.normal(size=(L, jmax)) + 1.j*np.random.normal(size=(L, jmax))
                initial_alphas = np.zeros((L, jmax))
                params = self.psf.encode_params(initial_alphas, initial_a)
                initial_lik = lik_fn(params)
                #res = scipy.optimize.minimize(lik_fn, np.random.normal(size=jmax), method='BFGS', jac=grad_fn, options={'disp': True})
                res = scipy.optimize.minimize(lik_fn, params, method='CG', jac=grad_fn, options={'disp': True, 'gtol':initial_lik*1e-7})#, 'eps':.1})
                loglik = res['fun']
                #assert(loglik == lik_fn(res['x']))
                if min_loglik is None or loglik < min_loglik:
                    min_loglik = loglik
                    min_res = res['x']
            for l in np.arange(0, L):
                for i in np.arange(0, jmax):
                    alphas_est[l, i] = min_res[l*jmax + i]
            if tt is not None:
                a_est = min_res[L*jmax:L*jmax+(2*(L+1))].reshape((L+1, 2))
            
        print("alphas_est", alphas_est)
        if tt is not None:
            print("a_est", a_est)
        #betas_est = np.random.normal(size=psf.jmax) + np.random.normal(size=psf.jmax)*1.j
        if tt is not None:
            return (alphas_est, a_est)
        else:
            return alphas_est
