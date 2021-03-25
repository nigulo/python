import sys
sys.path.append('../utils')
sys.path.append('..')
import os
import config
import matplotlib as mpl
from tqdm import tqdm

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import numpy as np
import cov_div_free as cov_div_free
import cov_sq_exp as cov_sq_exp

import scipy.misc
import numpy.random as random
import scipy.sparse.linalg as sparse
import scipy.stats as stats
import scipy.special as special
import utils
import plot
import misc
import kiss_gp

import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import simps

import time
import os.path
from astropy.io import fits
from scipy.io import readsav
import scipy.signal as signal
import pickle
import tables

sys.stdout = open(f'log{os.getpid()}', 'w')

state_file = 'data3d.pkl'
#state_file = 'data/IVM_AR9026.sav'
#state_file = 'pi-ambiguity-test/amb_turb.fits'

num_train = 2
num_test = 1#27*27


n1 = 27
n2 = 27
length_scale = 0.1
z_scale = 1

if len(sys.argv) > 1:
    state_file = sys.argv[1]
if len(sys.argv) > 2:
    n1 = int(sys.argv[2])
if len(sys.argv) > 3:
    n2 = int(sys.argv[3])
if len(sys.argv) > 4:
    length_scale = float(sys.argv[4])
if len(sys.argv) > 5:
    z_scale = float(sys.argv[5])

subsample = 1000000 # For D2 approximation
num_subsample_reps = 1


num_layers = 3
n3 = num_layers

m_kiss = 20

def load(file_name):
    
    if file_name[-4:] == '.sav':
        print("Use disambiguate.py for single layer data")
        sys.exit(0)
        idl_dict = readsav(file_name)
        #idl_dict = readsav('data/fan_simu_ts56.sav')
        
        lat = idl_dict['b'][0][1]
        long = idl_dict['b'][0][2]
        b_long = idl_dict['b'][0][3]
        b_trans = idl_dict['b'][0][4]
        b_azim = idl_dict['b'][0][5]
        
        #print(lat)
        #print(long)
        #print(b_long)
        #print(b_trans)
        #print(b_azim)
        
        b = np.sqrt(b_long**2 + b_trans**2)
        phi = b_azim*np.pi/180
        theta = np.arccos((b_long+1e-10)/(b+1e-10))
        
        print(b)
        print(phi)
        print(theta)
    
    elif file_name[-5:] == '.fits':
        hdul = fits.open(file_name)
        if file_name == 'hinode_data/inverted_atmos.fits':
            print("Data.shape", hdul[0].data.shape)
            #hdul = fits.open('pi-ambiguity-test/amb_spot.fits')
            # Actually x and y axis are swapped because of how the grid is defined later
            # Also swap y-axis to plot identically to XFITS Analyzer
            b = hdul[0].data[8:11, ::-1, :]
            theta = hdul[0].data[11:14, ::-1, :]*np.pi/180
            phi = hdul[0].data[14:17, ::-1, :]*np.pi/180
            
            b = np.transpose(b, (1, 2, 0))
            theta = np.transpose(theta, (1, 2, 0))
            phi = np.transpose(phi, (1, 2, 0))
        else:
            print("Use disambiguate.py for single layer data")
            sys.exit(0)
            dat = hdul[0].data[:,::4,::4]
            b = dat[0]
            theta = dat[1]
            phi = dat[2]
        hdul.close()
    elif file_name == 'MURaM':
        hdul = fits.open(file_name+'/result_5.100000.fits')
        print("Data.shape", hdul[0].data.shape)
        bx = hdul[0].data
        hdul.close()
        hdul = fits.open(file_name+'/result_6.100000.fits')
        bz = hdul[0].data
        hdul.close()
        hdul = fits.open(file_name+'/result_7.100000.fits')
        by = hdul[0].data
        hdul.close()

        bx = np.transpose(bx, (0, 2, 1))
        by = np.transpose(by, (0, 2, 1))
        bz = np.transpose(bz, (0, 2, 1))

        # Take only three surface layers
        bx = bx[:, :, -6::2]
        by = by[:, :, -6::2]
        bz = bz[:, :, -6::2]

        b = np.sqrt(bx**2 + by**2 + bz**2)
        phi = np.arctan2(by, bx)
        theta = np.arccos((bz+1e-10)/(b+1e-10))


    elif file_name[-4:] == '.pkl':
        if os.path.isfile(file_name):
            y = pickle.load(open(file_name, 'rb'))
    #    if os.path.isfile('data3d50x50x10.pkl'):
    #        y = pickle.load(open('data3d50x50x10.pkl', 'rb'))
        else: 
            # Data not present, generate new
            n1, n2, n3 = 100, 100, 3
            n = n1 * n2 * n3
            x1_range, x2_range, x3_range = 1., 1., .03
            
            x1 = np.linspace(0, x1_range, n1)
            x2 = np.linspace(0, x2_range, n2)
            x3 = np.linspace(0, x3_range, n3)
            x1_mesh, x2_mesh, x3_mesh = np.meshgrid(x1, x2, x3, indexing='ij')
            assert np.all(x1_mesh[:,0,0] == x1)
            assert np.all(x2_mesh[0,:,0] == x2)
            assert np.all(x3_mesh[0,0,:] == x3)
            x = np.stack((x1_mesh, x2_mesh, x3_mesh), axis=3)
            print("x1_mesh", x1_mesh)
            print("x2_mesh", x2_mesh)
            print("x3_mesh", x3_mesh)
            print("x", x)
            x = x.reshape(-1, 3)
            print("x", x)
            
            sig_var_train = 1.0
            length_scale_train = .03
            noise_var_train = 0.01
            mean_train = 0.
    
            gp_train = cov_div_free.cov_div_free(sig_var_train, length_scale_train, noise_var_train)
            K = gp_train.calc_cov(x, x, True)
    
            print("SIIN")
            for i in np.arange(0, n1):
                for j in np.arange(0, n2):
                    assert(K[i, j]==K[j, i])
            
            L = la.cholesky(K)
            s = np.random.normal(0.0, 1.0, 3*n)
            
            y = np.repeat(mean_train, 3*n) + np.dot(L, s)
            
            y = np.reshape(y, (n1, n2, n3, 3))
            
            with open(file_name, 'wb') as f:
                pickle.dump(y, f)    
        
        ###########################################################################
        # Plotting the whole qube
        x1_range, x2_range = 1., 1.
        x1 = np.linspace(0, x1_range, y.shape[0])
        x2 = np.linspace(0, x2_range, y.shape[1])
        x_mesh = np.meshgrid(x2, x1)
    
        for i in np.arange(0, y.shape[2]):
            test_plot = plot.plot(nrows=1, ncols=1)
            
            test_plot.colormap(y[:, :, i, 2], cmap_name="bwr")
            test_plot.vectors(x_mesh[0], x_mesh[1], y[:, :, i, 0], y[:, :, i, 1], [], units='width', color = 'k')
            test_plot.save("test_field" + str(i) +".png")
            test_plot.close()
        ###########################################################################
    
        bx = y[:, :, :, 0]
        by = y[:, :, :, 1]
        bz = y[:, :, :, 2]
        
        ###########################################################################
        # Overwrite some of the vector for depth testing purposes
        #for i in np.arange(0, y.shape[0]):
        #    for j in np.arange(0, y.shape[1]):
        #        #if i == y.shape[0]//2 or j == y.shape[1]//2:
        #        if i == j or y.shape[0] - i == j:
        #            bx[i, j] = y[i, j, 1, 0]
        #            by[i, j] = y[i, j, 1, 1]
        #            bz[i, j] = y[i, j, 1, 2]
        ###########################################################################
        b = np.sqrt(bx**2 + by**2 + bz**2)
        phi = np.arctan2(by, bx)
        theta = np.arccos((bz+1e-10)/(b+1e-10))
    
    
    else:
        print("Unknown input file type")
        sys.exit(1)
    return b, phi, theta

def get_patches(b, phi, theta, num_patches, rnd=True):
    ys = []
    
    x_start = 0
    y_start = 0
    for i in np.arange(num_patches):
        
        if rnd:

            x_start = random.randint(0, b.shape[0] + 1 - n1)
            y_start = random.randint(0, b.shape[1] + 1 - n2)
            
    
        #x_start = n1*x_no
        x_end = min(x_start + n1, b.shape[0])
        #y_start = n2*y_no
        y_end = min(y_start + n1, b.shape[1])
        
        if x_start < x_end and y_start < y_end:
        
            b1 = b[x_start:x_end, y_start:y_end, :num_layers]
            phi1 = phi[x_start:x_end, y_start:y_end, :num_layers]
            theta1 = theta[x_start:x_end, y_start:y_end, :num_layers]
    ###########################################################################
            
            
            
            bz = b1*np.cos(theta1)
            bxy = b1*np.sin(theta1)
            bx = bxy*np.cos(phi1)
            by = bxy*np.sin(phi1)
            
            #bx = np.reshape(bx, n)
            #by = np.reshape(by, n)
            #bz = np.reshape(bz, n)
            
            y = np.array([bx, by, bz])
            ys.append(y)
        if not rnd:
            x_start += n1
            if x_start > b.shape[0] - n1:
                x_start = 0
                y_start += n2
                if y_start > b.shape[0] - n2:
                    break
            
    return ys
    ###########################################################################

b, phi, theta = load(state_file)

n1_orig = b.shape[0]
n2_orig = b.shape[1]
n3_orig = num_layers

n = n1*n2*n3

x1_range = 1.0
x2_range = 1.0
x3_range = x1_range*n3/n1


x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x3 = np.linspace(0, x3_range, n3)

x1_mesh, x2_mesh, x3_mesh = np.meshgrid(x1, x2, x3, indexing='ij')
print("x1_mesh", x1_mesh[:,0,0])
print("x2_mesh", x2_mesh[0,:,0])
print("x3_mesh", x3_mesh[0,0,:])
x_grid = np.stack((x1_mesh, x2_mesh, x3_mesh), axis=3)
x = x_grid.reshape(-1, 3)
x_flat = np.reshape(x, (3*n, -1))



m1 = max(m_kiss, n1//10)
m2 = max(m_kiss, n2//10)
m3 = 2
m = m1 * m2 * m3
u1_range = np.max(x[:,0])
u2_range = np.max(x[:,1])
u3_range = np.max(x[:,2])

u1 = np.linspace(0, u1_range, m1)
u2 = np.linspace(0, u2_range, m2)
u3 = np.linspace(0, u3_range, m3)
u_mesh = np.meshgrid(u1, u2, u3, indexing='ij')



class data_generator():
    
    def __init__(self, data_array, loglik_array, x, ys, sig_var=None, length_scale=None, noise_var=None, approx_type='kiss-gp', u_mesh=None):
        self.x = x
        self.ys = ys
        self.n = len(x)
        assert(self.n == n1*n2*n3)
        
        self.y_sign = np.ones(n)
        thetas = np.ones(self.n)/2.
        self.thetas = np.log(thetas)

        self.num_positive = np.zeros(self.n)
        self.num_negative = np.zeros(self.n)
        
        self.approx_type = approx_type
        
        if approx_type == 'kiss-gp':
            assert(u_mesh is not None)
            self.u_mesh = u_mesh
            self.u = np.stack(u_mesh, axis=3).reshape(-1, 3)
        
        if sig_var is not None and length_scale is not None and noise_var is not None:
            self.sig_var = sig_var
            self.length_scale = length_scale
            self.noise_var = noise_var
            self.init()
            
        self.data_array = data_array
        self.loglik_array = loglik_array
        
    def init(self):    
        gp = cov_div_free.cov_div_free(self.sig_var, self.length_scale, self.noise_var)

    def loglik(self, y):
        gp = cov_div_free.cov_div_free(self.sig_var, self.length_scale, self.noise_var)
        if (self.approx_type == 'd2'):
            loglik = 0.
            for i in np.arange(0, num_subsample_reps):
                loglik += gp.calc_loglik_approx(self.x, np.reshape(y, (3*self.n, -1)), subsample=subsample)
                #if (best_loglik is None or loglik > best_loglik):
                #    best_loglik = loglik
            return loglik/num_subsample_reps
        elif self.approx_type == 'kiss-gp':
            U = gp.calc_cov(self.u, self.u, data_or_test=True)
            W = utils.calc_W(self.u_mesh, self.x, us=self.u, indexing_type=False)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
            (x, istop, itn, normr) = sparse.lsqr(W, np.reshape(y, (3*self.n, -1)))[:4]#, x0=None, tol=1e-05, maxiter=None, M=None, callback=None)
            L = la.cholesky(U)
            v = la.solve(L, x)
            return -0.5 * np.dot(v.T, v) - sum(np.log(np.diag(L))) - 0.5 * self.n * np.log(2.0 * np.pi)
        else:
            return gp.calc_loglik(self.x, np.reshape(y, (3*self.n, -1)))
    


    def generate(self, train=False):#, best_loglik = sys.float_info.min):

        num_data = len(self.ys)
        #ret_data = np.empty((num_data, np.shape(self.ys[0])[0], np.shape(self.ys[0])[1], np.shape(self.ys[0])[2], np.shape(self.ys[0])[3]))
        #ret_loglik = np.empty(num_data)
        
        for i in tqdm(np.arange(num_data)):
            if train:
                index = np.random.randint(num_data)
            else:
                index = i
            y = np.array(self.ys[index])
            if train:
                r = np.random.uniform(size=y[0].shape)
                r1 = np.random.uniform()
                y[:2, r < r1] *= -1
                
                if i < 10:
                    test_plot = plot.plot(nrows=3, ncols=3)
                    
                    test_plot.colormap(y[0, :, :, 0], [0, 0])
                    test_plot.colormap(y[1, :, :, 0], [0, 1])
                    test_plot.colormap(y[2, :, :, 0], [0, 2])

                    test_plot.colormap(y[0, :, :, 1], [1, 0])
                    test_plot.colormap(y[1, :, :, 1], [1, 1])
                    test_plot.colormap(y[2, :, :, 1], [1, 2])

                    test_plot.colormap(y[0, :, :, 2], [2, 0])
                    test_plot.colormap(y[1, :, :, 2], [2, 1])
                    test_plot.colormap(y[2, :, :, 2], [2, 2])
                    test_plot.save("random_field" + str(i) +".png")
                    test_plot.close()
                    
            
            loglik = self.loglik(y)
            self.data_array.append(np.array(y)[None,])
            self.loglik_array.append(np.array([[loglik]]))
        #return ret_data, ret_loglik
    

sig_var=1.
noise_var=0.01

x_orig = np.array(x)

x = np.array(x_orig)
x[:, 2] *= z_scale

print("length_scale, z_scale", length_scale, z_scale)
print("x", np.min(x), np.max(x))

ys = get_patches(b, phi, theta, num_train)
ys = np.asarray(ys)
print("mean, std", np.mean(ys, axis=(0,1,2,3)), np.std(ys, axis=(0,1,2,3)))
ys -= np.mean(ys, axis = 0)
ys /= np.std(ys, axis = 0)

output_file = tables.open_file(f"data_nn_out_{length_scale}_{z_scale}.h5", mode='a')
atom = tables.Float64Atom()

suffix = 0
for node in output_file.walk_nodes(output_file.root):
    node = node._v_name
    if node[:10] == "data_train":
        suf = int(node[10:])
        print(suf)
        suffix = max(suf, suffix)
suffix += 1

data_train = output_file.create_earray(output_file.root, f'data_train{suffix}', atom, (0,) + ys.shape[1:])
loglik_train = output_file.create_earray(output_file.root, f'loglik_train{suffix}', atom, (0, 1))
data_test = output_file.create_earray(output_file.root, f'data_test{suffix}', atom, (0,) + ys.shape[1:])
loglik_test = output_file.create_earray(output_file.root, f'loglik_test{suffix}', atom, (0, 1))

print("Num train patches", len(ys))
generator = data_generator(data_train, loglik_train, x, ys, sig_var, length_scale, noise_var, approx_type='kiss-gp', u_mesh=u_mesh)
generator.generate(train=True)


ys = get_patches(b, phi, theta, num_test, rnd=False)
ys -= np.mean(ys, axis = 0)
ys /= np.std(ys, axis = 0)

print("Num test patches", len(ys))
generator = data_generator(data_test, loglik_test, x, ys, sig_var, length_scale, noise_var, approx_type='kiss-gp', u_mesh=u_mesh)
generator.generate()

output_file.close()

#np.savez_compressed(f'data_nn_out_{length_scale}_{z_scale}', data_train=dat, loglik_train=loglik, data_test=dat_test, loglik_test=loglik_test)

print("Done")
