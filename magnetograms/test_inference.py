import sys
sys.path.append('../')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import pickle
import numpy as np
#import pylab as plt
#import pandas as pd
import GPR_div_free as GPR_div_free
import scipy.misc
import numpy.random as random
import scipy.interpolate as interp
import scipy.sparse.linalg as sparse
from matplotlib import cm


#import os
#import os.path
#from scipy.stats import gaussian_kde
#from sklearn.cluster import KMeans
import numpy.linalg as la
import matplotlib.pyplot as plt

num_iters = 50
num_chains = 1
inference = False

eps = 0.001
learning_rate = 1.0
max_num_tries = 20


if len(sys.argv) > 3:
    num_iters = int(sys.argv[3])
if len(sys.argv) > 4:
    num_chains = int(sys.argv[4])

n_jobs = num_chains

model = pickle.load(open('model.pkl', 'rb'))

n1 = 10
n2 = 10
n = n1*n2
x1_range = 1.0
x2_range = 1.0
x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x_mesh = np.meshgrid(x1, x2)
x = np.dstack(x_mesh).reshape(-1, 2)
x_flat = np.reshape(x, (2*n, -1))

m1 = 5
m2 = 5
m = m1 * m2
u1 = np.linspace(0, x1_range, m1)
u2 = np.linspace(0, x2_range, m2)
u_mesh = np.meshgrid(u1, u2)
u = np.dstack(u_mesh).reshape(-1, 2)

print "u_mesh=", u_mesh
print "u=", u

def bilinear_interp(xs, ys, x, y):
    coefs = np.zeros(len(xs)*len(ys))
    h = 0
    for i in np.arange(0, len(xs)):
        num = 1.0
        denom = 1.0
        for j in np.arange(0, len(xs)):
            if i != j:
                for k in np.arange(0, len(ys)):
                    for l in np.arange(0, len(ys)):
                        if k != l:
                            num *= (x - xs[j])*(y - ys[l])
                            denom *= (xs[i] - xs[j])*(ys[k] - ys[l])
        coefs[h] = num/denom
        h += 1
    return coefs

def get_closest(xs, ys, x, y, count_x=2, count_y=2):
    dists_x = np.abs(xs - x)
    dists_y = np.abs(ys - y)
    indices_x = np.argsort(dists_x)
    indices_y = np.argsort(dists_y)
    xs_c = np.zeros(count_x)
    ys_c = np.zeros(count_y)
    for i in np.arange(0, count_x):
        xs_c[i] = xs[indices_x[i]]
    for i in np.arange(0, count_y):
        ys_c[i] = ys[indices_y[i]]
    return (xs_c, ys_c), (indices_x[:count_x], indices_y[:count_y])

#Optimeerida
#def get_W(u_mesh, us, xys):
#    W = np.zeros((np.shape(xys)[0], np.shape(us)[0]))
#    i = 0
#    for (x, y) in xys:
#        (u1s, u2s), (indices_x, indices_y) = get_closest(u_mesh[0][0,:], u_mesh[1][:,0], x, y)
#        coefs = bilinear_interp(u1s, u2s, x, y)
#        for j in np.arange(0, len(us)):
#            found = False
#            coef_ind = 0
#            for u1 in u1s:
#                if us[j][0] == u1:
#                    for u2 in u2s:
#                        if us[j][1] == u2:
#                            found = True
#                            break
#                        coef_ind += 1
#                    if found:
#                        break
#                else:
#                    coef_ind += len(u2s)
#            if found:
#                W[i, j] = coefs[coef_ind]
#                #print "W=", W
#        i += 1
#    return W

#def calc_W(u_mesh, u, x):
#    W = np.zeros((np.shape(x)[0]*2, np.shape(u)[0]*2))
#    for i in np.arange(0, np.shape(x)[0]):
#        W1 = get_W(u_mesh, u, x)#, np.reshape(U[i,0::2], (len(u1), len(u2))))
#        #print np.shape(W), np.shape(W1), np.shape(W1)
#        for j in np.arange(0, np.shape(W1)[1]):
#            W[2*i,2*j] = W1[i, j]
#            W[2*i,2*j+1] = W1[i, j]
#            W[2*i+1,2*j] = W1[i, j]
#            W[2*i+1,2*j+1] = W1[i, j]
#    return W

def calc_W(u_mesh, us, xys):
    W = np.zeros((np.shape(xys)[0]*2, np.shape(us)[0]*2))
    i = 0
    for (x, y) in xys:
        (u1s, u2s), (indices_u1, indices_u2) = get_closest(u_mesh[0][0,:], u_mesh[1][:,0], x, y)
        coefs = bilinear_interp(u1s, u2s, x, y)
        coef_ind = 0
        for u1_index in indices_u1:
            for u2_index in indices_u2:
                j = u2_index * m1 + u1_index
                W[2*i,2*j] = coefs[coef_ind]
                W[2*i,2*j+1] = coefs[coef_ind]
                W[2*i+1,2*j] = coefs[coef_ind]
                W[2*i+1,2*j+1] = coefs[coef_ind]
                coef_ind += 1
        i += 1
    return W

print x_mesh

sig_var_train = 0.2
length_scale_train = 0.1
noise_var_train = 0.000001
mean_train = 0.0

gp_train = GPR_div_free.GPR_div_free(sig_var_train, length_scale_train, noise_var_train)
K = gp_train.calc_cov(x, x, True)
#U = gp_train.calc_cov(u, u, data_or_test=True)
#W = calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
#K = np.dot(W.T, np.dot(U, W))

print "SIIN"
for i in np.arange(0, n1):
    for j in np.arange(0, n2):
        assert(K[i, j]==K[j, i])

L = la.cholesky(K)
s = np.random.normal(0.0, 1.0, 2*n)

y = np.repeat(mean_train, 2*n) + np.dot(L, s)

y = np.reshape(y, (n, 2))
y_orig = np.array(y)
print y_orig

fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2, ncols=2, sharex=True)
fig.set_size_inches(12, 12)

extent=[x1.min(), x1.max(), x2.min(), x2.max()]
plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])*2/3 

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

my_cmap = reverse_colourmap(plt.get_cmap('gnuplot'))

im11 = ax11.imshow(np.reshape(K[0,0::2], (len(x1), len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K), vmax=np.max(K))
im12 = ax12.imshow(np.reshape(K[0,1::2], (len(x1), len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K), vmax=np.max(K))
im21 = ax21.imshow(np.reshape(K[1,0::2], (len(x1), len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K), vmax=np.max(K))
im22 = ax22.imshow(np.reshape(K[1,1::2], (len(x1), len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K), vmax=np.max(K))

fig.savefig("cov_true.png")

fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2, ncols=2, sharex=True)
fig.set_size_inches(12, 12)

U = gp_train.calc_cov(u, u, data_or_test=True)
W = calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))

K1 = np.dot(W, np.dot(U, W.T))

im11 = ax11.imshow(np.reshape(K1[0,0::2], (len(x1), len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K1), vmax=np.max(K1))
im12 = ax12.imshow(np.reshape(K1[0,1::2], (len(x1), len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K1), vmax=np.max(K1))
im21 = ax21.imshow(np.reshape(K1[1,0::2], (len(x1), len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K1), vmax=np.max(K1))
im22 = ax22.imshow(np.reshape(K1[1,1::2], (len(x1), len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K1), vmax=np.max(K1))

fig.savefig("cov_approx.png")

y_flat = np.reshape(y, (2*n, -1))

mean = 0.0
length_scale = 1.0
noise_var = 0.0001

print np.shape(x)
print np.shape(y)
print n

def calc_loglik_approx(U, W, y):
    #print np.shape(W), np.shape(y)
    (x, istop, itn, normr) = sparse.lsqr(W, y)[:4]#, x0=None, tol=1e-05, maxiter=None, M=None, callback=None)
    #print x
    L = la.cholesky(U)
    #print L
    v = la.solve(L, x)
    return -0.5 * np.dot(v.T, v) - sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)
    
def calc_loglik(K, y):
    L = la.cholesky(K)
    v = la.solve(L, y)
    return -0.5 * np.dot(v.T, v) - sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)

#def get_w(x_mesh, u, K):
#    #print K
#    #print np.shape(K), np.shape(x_mesh[0][0,:]), np.shape(x_mesh[1][:,0])
#    rbs = interp.RectBivariateSpline(x_mesh[0][0,:], x_mesh[1][:,0], K, kx=3, ky=3)
#    #w = rbs.ev(u[:,0], u[:,1])
#    w = rbs.get_coeffs()
#    r = rbs.get_residual()
#    k = rbs.get_knots()
#    print "w=", w
#    print "r=", r
#    print "k=", k
#    print rbs.ev(u[:,0], u[:,1])
#    for i in np.arange(0, np.shape(u)[0]):
#        val = 0
#        for j in np.arange(0, len(w)):
#            val += (u[i,0] - x[j,0])*w[j]*(u[i,1] - x[j,1])
#        print val
#    #W = np.reshape(w, (len(x1)*len(x2), np.shape(u)[0]))
#    return w

def test():
    test_fig, ax_test = plt.subplots(nrows=1, ncols=1, sharex=True)
    test_fig.set_size_inches(4, 6)
    
    length_scales = np.linspace(length_scale_train/100, length_scale_train*2, 20)
    logliks_approx = np.zeros(len(length_scales))
    logliks_true = np.zeros(len(length_scales))
    for test_no in np.arange(0, len(length_scales)):
        length_scale = length_scales[test_no]
        gp = GPR_div_free.GPR_div_free(sig_var_train, length_scale, noise_var_train)
        K = gp.calc_cov(x, x, data_or_test=True)
        U = gp.calc_cov(u, u, data_or_test=True)
        #print K
        #print "x=", x
        #print "u=", u
        
        W = calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
        #for i in np.arange(0, len(x1)*len(x2)):
        #    W1 = get_W(u_mesh, u, x)#, np.reshape(U[i,0::2], (len(u1), len(u2))))
        #    #print np.shape(W), np.shape(W1), np.shape(W1)
        #    for j in np.arange(0, np.shape(W1)[1]):
        #        W[2*i,2*j] = W1[i, j]
        #        W[2*i,2*j+1] = W1[i, j]
        #        W[2*i+1,2*j] = W1[i, j]
        #        W[2*i+1,2*j+1] = W1[i, j]
        
        #for i in np.arange(0, np.shape(W)[0]):
        #    for j in np.arange(0, np.shape(W)[1]):
        #        print "W=", W[i,j]
        
        logliks_approx[test_no] = calc_loglik_approx(U, W, y_flat)
        logliks_true[test_no] = calc_loglik(K, y_flat)
    
    logliks_approx = (logliks_approx - np.min(logliks_approx))/(np.max(logliks_approx) - np.min(logliks_approx))
    logliks_true = (logliks_true - np.min(logliks_true))/(np.max(logliks_true) - np.min(logliks_true))
    
    ax_test.plot(length_scales, logliks_approx, "b-")
    ax_test.plot(length_scales, logliks_true, "r-")
    test_fig.savefig("test.png")
    #U1 = np.zeros((np.shape(K)[0], len(u1)*len(u2)*2))
    #for i in np.arange(0, np.shape(K)[0]):
    #    w1 = get_w(x_mesh, u, np.reshape(K[i,0::2], (len(x1), len(x2))))
    #    w2 = get_w(x_mesh, u, np.reshape(K[i,1::2], (len(x1), len(x2))))
    #    print np.shape(U1), np.shape(w1)
    #    for j in np.arange(0, np.shape(w1)[0]):
    #        U1[i,2*j] = w1[j]
    #        U1[i,2*j+1] = w2[j]
    
    #U1 = U1.T
    #U = np.zeros((len(u1)*len(u2)*2, len(u1)*len(u2)*2))
    #for i in np.arange(0, np.shape(U1)[0]):
    #    w1 = get_w(x_mesh, u, np.reshape(U1[i,0::2], (len(x1), len(x2))))
    #    w2 = get_w(x_mesh, u, np.reshape(U1[i,1::2], (len(x1), len(x2))))
    #    print np.shape(U1), np.shape(w1)
    #    for j in np.arange(0, np.shape(w1)[0]):
    #        U[i,2*j] = w1[j]
    #        U[i,2*j+1] = w2[j]
    
    #print U

#test()




