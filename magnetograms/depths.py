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

        # Calculate horizontal, vertical and diagonal diffs
        self.dbxx = bx[1:,:]-bx[:-1,:]
        self.dbxy = bx[:,1:]-bx[:,:-1]

        self.dbx1 = bx[1:,1:]-bx[:-1,:-1]
        self.dbx2 = bx[:-1,1:]-bx[1:,:-1]

        self.dbyx = by[1:,:]-by[:-1,:]
        self.dbyy = by[:,1:]-by[:,:-1]

        self.dby1 = by[1:,1:]-by[:-1,:-1]
        self.dby2 = by[:-1,1:]-by[1:,:-1]

        self.dbzx = bz[1:,:]-bz[:-1,:]
        self.dbzy = bz[:,1:]-bz[:,:-1]

        self.dbz1 = bz[1:,1:]-bz[:-1,:-1]
        self.dbz2 = bz[:-1,1:]-bz[1:,:-1]

        self.i = 0
        self.j = 0


    def calc_b_d_az(self, b_derivs, dz):
        i = self.i
        j = self.j
        b = np.zeros(24)
        b[0] = self.dbxx[i-1, j]
        b[1] = self.dbxx[i, j]
        
        b[2] = self.dbxy[i, j-1]
        b[3] = self.dbxy[i, j]

        b[4] = self.dbyx[i-1, j]
        b[5] = self.dbyx[i, j]
        
        b[6] = self.dbyy[i, j-1]
        b[7] = self.dbyy[i, j]
        
        b[8] = self.dbzx[i-1, j]
        b[9] = self.dbzx[i, j]
        
        b[10] = self.dbzy[i, j-1]
        b[11] = self.dbzy[i, j]

        
        b[12] = self.dbx1[i-1, j-1]
        b[13] = self.dbx1[i, j]
        
        b[14] = self.dbx2[i, j-1]
        b[15] = self.dbx2[i-1, j]

        b[16] = self.dby1[i-1, j-1]
        b[17] = self.dby1[i, j]
        
        b[18] = self.dby2[i, j-1]
        b[19] = self.dby2[i-1, j]

        b[20] = self.dbz1[i-1, j-1]
        b[21] = self.dbz1[i, j]
        
        b[22] = self.dbz2[i, j-1]
        b[23] = self.dbz2[i-1, j]

        d = np.zeros(24)
        d[0] = -b_derivs[0]*self.dx[i, j]
        d[1] = -d[0]
        
        d[2] = -b_derivs[1]*self.dy[i, j]
        d[3] = -d[2]

        d[4] = -b_derivs[3]*self.dx[i, j]
        d[5] = -d[4]

        d[6] = -b_derivs[4]*self.dy[i, j]
        d[7] = -d[6]

        d[8] = -b_derivs[6]*self.dx[i, j]
        d[9] = -d[8]

        d[10] = -b_derivs[7]*self.dy[i, j]
        d[11] = -d[10]
        
        for di in np.arange(12, 14):
            d[di] = d[di-12] + d[di-10]
        for di in np.arange(14, 16):
            d[di] = -d[di-14] + d[di-12]

        for di in np.arange(16, 18):
            d[di] = d[di-12] + d[di-10]
        for di in np.arange(18, 20):
            d[di] = -d[di-14] + d[di-12]

        for di in np.arange(20, 22):
            d[di] = d[di-12] + d[di-10]
        for di in np.arange(22, 24):
            d[di] = -d[di-14] + d[di-12]
        
        dbx_dz = b_derivs[2]
        dby_dz = b_derivs[5]
        dbz_dz = -b_derivs[0] - b_derivs[4]
        az = np.zeros(24)

        #E-W and N-S
        for di in np.arange(0, 4):
            az[di] = dbx_dz*dz[di]
        do = 4            
        for di in np.arange(0, 4):
            az[di+do] = dby_dz*dz[di]
        do += 4
        for di in np.arange(0, 4):
            az[di+do] = dbz_dz*dz[di]
        do += 4
        #SE-NW and NE-SW
        for di in np.arange(0, 4):
            az[di+do] = dbx_dz*dz[di+4]
        do += 4
        for di in np.arange(0, 4):
            az[di+do] = dby_dz*dz[di+4]
        do += 4
        for di in np.arange(0, 4):
            az[di+do] = dbz_dz*dz[di+4]

        return b, d, az

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
        
        dz = params[8:16]
        # 0: dz^-x
        # 1: dz^x
        # 2: dz^-y
        # 3: dz^y
        
        # 0: dz^-x,-y
        # 1: dz^x,y
        # 2: dz^x,-y
        # 3: dz^y,-x
        
        b, d, az = self.calc_b_d_az(b_derivs, dz)
        
        
        loss = np.sum((b - (d + az))**2) + np.sum(params**2*self.prior_prec/2)
        #loss = np.sum(((b - (d + az))**2)[:12]) + np.sum(params[:12]**2*self.prior_prec/2)
        #print("params", params[8:12])
        #print("loss", loss)
        
        return loss


    def loss_fn_grad(self, params):
        b_derivs = params[:8]
        dz = params[8:16]

        i = self.i
        j = self.j
        

        b, d, az = self.calc_b_d_az(b_derivs, dz)
        
        grads = np.zeros(len(params))
        l = 2.*(b - (d + az))
        grads[0] = (l[0] - l[1] + l[12] - l[13] - l[14] + l[15])*self.dx[i, j]
        for di in np.arange(0, 4):
            grads[0] += l[di+8]*dz[di]+l[di+20]*dz[di+4]

        grads[1] = (l[2] - l[3] + l[12] - l[13] + l[14] - l[15])*self.dy[i, j]
        
        grads[3] = (l[4] - l[5] + l[16] - l[17] - l[18] + l[19])*self.dx[i, j]
        grads[4] = (l[6] - l[7] + l[16] - l[17] + l[18] - l[19])*self.dy[i, j]
        for di in np.arange(0, 4):
            grads[4] += l[di+8]*dz[di]+l[di+20]*dz[di+4]

        grads[6] = (l[8] - l[9] + l[20] - l[21] - l[22] + l[23])*self.dx[i, j]
        grads[7] = (l[10] - l[11] + l[20] - l[21] + l[22] - l[23])*self.dy[i, j]
        
        
        for di in np.arange(0, 4):
            grads[2] += -(l[di]*dz[di] + l[12+di]*dz[di+4])

        for di in np.arange(0, 4):
            grads[5] += -(l[di+4]*dz[di] + l[16+di]*dz[di+4])


        dbx_dz = b_derivs[2]
        dby_dz = b_derivs[5]
        dbz_dz = -b_derivs[0] - b_derivs[4]

        for di in np.arange(0, 4):
            grads[8+di] += -l[di]*dbx_dz
            grads[8+di] += -l[di+4]*dby_dz
            grads[8+di] += -l[di+8]*dbz_dz

        for di in np.arange(0, 4):
            grads[12+di] += -l[di+12]*dbx_dz
            grads[12+di] += -l[di+16]*dby_dz
            grads[12+di] += -l[di+20]*dbz_dz


        grads += params*self.prior_prec

        return grads

    def calc_depths(self, dzs):
        num_tries = 100
        nx = self.dx.shape[0]
        ny = self.dy.shape[1]
        depths1 = np.zeros((nx, ny))
        for trial in np.arange(0, num_tries):
            depths = np.zeros((nx, ny))
            counts = np.zeros((nx, ny))
            coords = np.random.choice(np.arange(0, nx*ny), nx*ny, replace=False)
            for coord in coords:
                i = coord// ny
                j = coord % nx
                
                dz = dzs[i, j]

                if i > 0:
                    depths[i-1, j] += depths[i, j] + dz[0]
                    counts[i-1, j] += 1
                    if j > 0:
                        depths[i-1, j-1] += depths[i, j] + dz[4]
                        counts[i-1, j-1] += 1
                if j > 0:
                    depths[i, j-1] += depths[i, j] + dz[2]
                    counts[i, j-1] += 1
                
                if i < nx - 1:
                    depths[i+1, j] += depths[i, j] + dz[1]
                    counts[i+1, j] += 1
                    if j < ny - 1:
                        depths[i+1, j+1] += depths[i, j] + dz[5]
                        counts[i+1, j+1] += 1
                        
                    if j > 0:
                        depths[i+1, j-1] += depths[i, j] + dz[7]
                        counts[i+1, j-1] += 1
                        
                if j < ny - 1:
                    depths[i, j+1] += depths[i, j] + dz[3]
                    counts[i, j+1] += 1
                    if i > 0:
                        depths[i-1, j+1] += depths[i, j] + dz[6]
                        counts[i-1, j+1] += 1

            indices = np.where(counts > 0)
            depths[indices] /= counts[indices]
            
            depths1 += depths
        
        depths1 /= num_tries
                
        return depths1 


    def estimate(self):

        def loss_fn(params):
            return self.loss_fn(params)

        def grad_fn(params):
            return self.loss_fn_grad(params)

        nx = self.dx.shape[0]
        ny = self.dy.shape[1]
        depths = np.zeros((nx, ny))
        dzs = np.zeros((nx, ny, 8))
        
        for i in np.arange(0, nx-1, step=1):
            self.i = i
            for j in np.arange(0, ny-1, step=1):
                self.j = j
            
                min_loss = None
                min_res = None
        
                for trial_no in np.arange(0, 1):
                    #initial_params = np.random.normal(size=16)
                    initial_params = np.zeros(16)
                    initial_loss = loss_fn(initial_params)
                    #res = scipy.optimize.minimize(lik_fn, initial_params, method='CG', jac=grad_fn, options={'disp': True, 'gtol':1e-9})#, 'eps':.1})
                    res = scipy.optimize.minimize(loss_fn, initial_params, method='CG', jac=grad_fn, options={'disp': True, 'gtol':initial_loss*1e-5, 'eps':.01})
                    print(res)
                    print("Optimization result:" + res["message"])
                    print("Status", res['status'])
                    print("Success", res['success'])
                    loss=res['fun']
                    #assert(loglik == lik_fn(res['x']))
                    if min_loss is None or loss < min_loss:
                        min_loss = loss
                        min_res = res['x']
                dz = min_res[8:16]
                print(dz)
                print("grad Bx", min_res[:3])
                print("grad By", min_res[3:6])
                print("grad Bz", min_res[6:8])
                dzs[i, j] = dz

                '''
                contrib = 0.
                count = 0.
                if i > 0:
                    contrib += depths[i-1, j] + dz[0]
                    count += 1
                    if j > 0:
                        contrib += depths[i-1, j-1] + dz[4]
                        count += 1
                if j > 0:
                    contrib += depths[i, j-1] + dz[2] + depths[i+1, j-1] + dz[7]
                    count += 2
                if count > 0:
                    depths[i, j] += contrib/count
                    depths[i, j] /=2
                
                depths[i+1, j+1] = depths[i, j] + dz[5]
                depths[i+1, j] = depths[i, j] + dz[1]
                depths[i, j+1] += depths[i, j] + dz[3]
                if i > 0:
                    depths[i-1, j+1] += depths[i, j] + dz[6]
                    depths[i-1, j+1] /= 2

                    depths[i, j+1] /= 2
                '''

                '''
                depths[i+1, j] = depths[i, j] + dz[1]
                if i > 0:
                    depths[i, j+1] += depths[i, j] + dz[3]
                    depths[i, j+1] /=2

                    depths[i-1, j] += depths[i, j] + dz[0]
                    depths[i-1, j] /= 2
                else:
                    depths[i, j+1] = depths[i, j] + dz[3]
                    
                
                if j > 0:
                    depths[i, j-1] += depths[i, j] + dz[2]
                    depths[i, j-1] /= 2
                '''
        
        depths = self.calc_depths(dzs)
        print("nx, ny", nx, ny)
        #print("depths", depths)
        
        my_plot = plot.plot(nrows=1, ncols=1)
        my_plot.colormap(depths)
        my_plot.save("depths.png")
        my_plot.close()

        for i in np.arange(0, dzs.shape[2]):
            my_plot = plot.plot(nrows=1, ncols=1)
            my_plot.colormap(dzs[:,:,i], colorbar=True)
            my_plot.save("dz" + str(i) + ".png")
            my_plot.close()


        return depths