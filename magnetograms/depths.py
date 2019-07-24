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
import plot

import scipy.optimize


class depths():
    
    def __init__(self, xs, bx, by, bz, prior_prec=0.):
        self.xs=xs
        self.bx=bx
        self.by=by
        self.bz=bz
        self.prior_prec=prior_prec
        print("xs", xs.shape)
        self.dx = xs[1:,:,0]-xs[:-1,:,0]
        self.dy = xs[:,1:,1]-xs[:,:-1,1]

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
        dbz_dz = -b_derivs[0] - b_derivs[4]
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
        
        loss = np.sum((b - (d + az))**2) + np.sum(params**2*self.prior_prec/2)
        #print("params", params[8:12])
        #print("loss", loss)
        
        return loss


    def estimate(self):

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
            
                min_loss = None
                min_res = None
        
                for trial_no in np.arange(0, 1):
                    #initial_params = np.random.normal(size=10)
                    initial_params = np.zeros(12)
                    res = scipy.optimize.minimize(lik_fn, initial_params, method='CG', jac=None, options={'disp': True, 'gtol':1e-9, 'eps':1e-5})
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
                print(dz)
                print("grad Bx", min_res[:3])
                print("grad By", min_res[3:6])
                print("grad Bz", min_res[6:8])

                
                #depths[i+1, j] = depths[i, j] + dz[0]
                #depths[i, j+1] = depths[i, j] + dz[1]
                
        
        print("nx, ny", nx, ny)
        #print("depths", depths)
        
        my_plot = plot.plot(nrows=1, ncols=1)
        my_plot.colormap(depths)
        my_plot.save("depths.png")
        my_plot.close()

        return depths