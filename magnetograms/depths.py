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


class depths():
    
    def __init__(self, xs, bx, by, bz):
        self.xs=xs
        self.bx=bx
        self.by=by
        self.bz=bz
        self.dx = xs[1:,:]-xs[:-1,:]
        self.dy = xs[:,1:]-xs[:,:-1]

        self.dbxx = bx[1:,:]-bx[:-1,:]
        self.dbxy = bx[:,1:]-bx[:,:-1]

        self.dbyx = by[1:,:]-by[:-1,:]
        self.dbyy = by[:,1:]-by[:,:-1]

        self.dbzx = bz[1:,:]-bz[:-1,:]
        self.dbzy = bz[:,1:]-bz[:,:-1]


        self.i = 0
        self.j = 0

    def loss_fn(self, params):
        
        b_derivs = params[:8]
        # 0: dB_x/d_x
        # 1: dB_x/d_y
        # 2: dB_x/d_z

        # 3: dB_y/d_x
        # 4: dB_y/d_y
        # 5: dB_y/d_z

        # 6: dB_z/d_x
        # 7: dB_z/d_y
        # dB_z/d_z is omitted, as expressed through others
        
        dz = params[8:12]
        # 0: dz^-x
        # 1: dz^x
        # 2: dz^-y
        # 3: dz^y

        i = self.i
        j = self.j

        b = np.zeros(12)
        b[0] = self.dbxx[i, j]
        b[1] = self.dbxx[i+1, j]
        
        b[2] = self.dbxy[i, j]
        b[3] = self.dbxy[i, j+1]

        b[4] = self.dbyx[i, j]
        b[5] = self.dbyx[i+1, j]
        
        b[6] = self.dbyy[i, j]
        b[7] = self.dbyy[i, j+1]
        
        b[8] = self.dbzx[i, j]
        b[9] = self.dbzx[i+1, j]
        
        b[10] = self.dbzy[i, j]
        b[11] = self.dbzy[i, j+1]
        
        d = np.zeros(12)
        d[0] = b_derivs[0]*self.dx[i, j]
        d[1] = -d[0]
        
        d[2] = b_derivs[1]*self.dy[i, j]
        d[3] = -d[2]

        d[4] = b_derivs[3]*self.dx[i, j]
        d[5] = -d[4]

        d[6] = b_derivs[4]*self.dy[i, j]
        d[7] = -d[6]

        d[8] = b_derivs[6]*self.dx[i, j]
        d[9] = -d[8]

        d[10] = b_derivs[7]*self.dy[i, j]
        d[11] = -d[10]
        
        dbx_dz = b_derivs[2]
        dby_dz = b_derivs[5]
        dbz_dz = -dbx_dz - dby_dz
        az = np.zeros(12)
        az[0] = dbx_dz*dz[0]
        az[1] = dbx_dz*dz[1]
        az[2] = dbx_dz*dz[2]
        az[3] = dbx_dz*dz[3]

        az[4] = dby_dz*dz[0]
        az[5] = dby_dz*dz[1]
        az[6] = dby_dz*dz[2]
        az[7] = dby_dz*dz[3]

        az[8] = dbz_dz*dz[0]
        az[9] = dbz_dz*dz[1]
        az[10] = dbz_dz*dz[2]
        az[11] = dbz_dz*dz[3]
        
        loss = np.sum(b - (d + az)**2)
        
        return loss


    def sample(self):

        nx = self.dx.shape[0] - 1
        ny = self.dy.shape[1] - 1
        depths = np.zeros((nx, ny))
        
        for i in np.arange(0, nx):
            self.i = i
            for j in np.arange(0, ny):
                self.j = j
            
        
        b_derivs_est = np.zeros(8)

        tt = self.psf_b.tip_tilt
        if tt is not None:
            a_est = np.zeros(((L+1), 2))


        def lik_fn(params):
            data = self.psf_b.encode_data(Ds, self.gamma)
            return self.loss_fn(params, data)

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
            #res = scipy.optimize.fmin_cg(lik_fn, params, fprime=grad_fn, args=(), full_output=True)
            #res = scipy.optimize.fmin_bfgs(lik_fn, params, fprime=grad_fn, args=(), full_output=True)
            #lower_bounds = np.zeros(jmax*2)
            #upper_bounds = np.ones(jmax*2)*1e10
            res = scipy.optimize.minimize(lik_fn, params, method='CG', jac=grad_fn, options={'disp': True, 'gtol':100})#, 'eps':.1})
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
        print("a_est", a_est)
        #betas_est = np.random.normal(size=psf.jmax) + np.random.normal(size=psf.jmax)*1.j
        if tt is not None:
            return (betas_est, a_est)
        else:
            return betas_est
