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
        
        dz = params[8:10]
        # 1: dz^x
        # 3: dz^y

        i = self.i
        j = self.j

        b = np.zeros(6)
        b[0] = self.dbxx[i, j]
        b[1] = self.dbxy[i, j]
        b[2] = self.dbyx[i, j]
        b[3] = self.dbyy[i, j]
        b[4] = self.dbzx[i, j]
        b[5] = self.dbzy[i, j]
        
        d = np.zeros(6)
        d[0] = b_derivs[0]*self.dx[i, j]
        d[1] = b_derivs[1]*self.dy[i, j]
        d[2] = b_derivs[3]*self.dx[i, j]
        d[3] = b_derivs[4]*self.dy[i, j]
        d[4] = b_derivs[6]*self.dx[i, j]
        d[5] = b_derivs[7]*self.dy[i, j]
        
        dbx_dz = b_derivs[2]
        dby_dz = b_derivs[5]
        dbz_dz = -dbx_dz - dby_dz
        az = np.zeros(12)
        az[0] = dbx_dz*dz[0]
        az[1] = dbx_dz*dz[1]

        az[2] = dby_dz*dz[0]
        az[3] = dby_dz*dz[1]

        az[4] = dbz_dz*dz[0]
        az[5] = dbz_dz*dz[1]
        
        loss = np.sum(b - (d + az)**2)
        
        return loss


    def sample(self):

        def lik_fn(params):
            return self.loss_fn(params)

        #def grad_fn(params):
        #    data = self.psf_b.encode_data(Ds, self.gamma)
        #    return self.psf_b.likelihood_grad(params, data)

        nx = self.dx.shape[0]
        ny = self.dy.shape[1]
        depths = np.zeros((nx, ny))
        
        for i in np.arange(0, nx-1, step=2):
            self.i = i
            for j in np.arange(0, ny-1, step=2):
                self.j = j
            
                b_derivs_est = np.zeros(8)
                dz_est = np.zeros(4)
        
                tt = self.psf_b.tip_tilt
                if tt is not None:
                    a_est = np.zeros(((L+1), 2))
        
        
                min_loss = None
                min_res = None
        
                for trial_no in np.arange(0, self.num_samples):
                    #initial_params = np.random.normal(size=12)
                    initial_params = np.zeros(12)
                    res = scipy.optimize.minimize(lik_fn, initial_params, method='CG', jac=None, options={'disp': True, 'gtol':100})#, 'eps':.1})
                    print(res)
                    print("Optimization result:" + res["message"])
                    print("Status", res['status'])
                    print("Success", res['success'])
                    loss=res['fun']
                    #assert(loglik == lik_fn(res['x']))
                    if min_loss is None or loss < min_loss:
                        min_loss = loss
                        min_res = res['x']
                dz = min_res[8:12]

                
                depths[i, j+1] -= dz[0]
                depths[i+2, j+1] += dz[1]

                depths[i+1, j] -= dz[2]
                depths[i+1, j+2] += dz[3]
                
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
