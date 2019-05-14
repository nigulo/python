import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=6

import sys
sys.path.append('../utils')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import numpy as np
import cov_div_free as cov_div_free
import cov_sq_exp as cov_sq_exp
import plot
import numpy.random as random

'''
xs = []

x = random.uniform(-1, 1., 20)
y = random.uniform(-1, 1., 20)
z = random.uniform(-1, 1., 20)

xs.append(np.column_stack((x, y, z)))

x = random.uniform(-1, 1., 20)
y = random.uniform(-1, 1., 20)
z = random.uniform(-1, 1., 20)

xs.append(np.column_stack((x, y, z)))

n = xs[0].shape[0]

ys = []

bx = random.uniform(-1, 1., 20)
by = random.uniform(-1, 1., 20)
bz = random.uniform(-1, 1., 20)
ys.append(np.column_stack((bx, by, bz)))
'''

xs = []

xs.append(np.array([
    [1., 0., 0.], 
    [np.cos(np.pi*2/3), np.sin(np.pi*2/3), 0.],
    [np.cos(np.pi*4/3), np.sin(np.pi*4/3), 0.],
#    [.5, 0., 0.], 
#    [.5*np.cos(np.pi*2/3), .5*np.sin(np.pi*2/3), -2.],
#    [.5*np.cos(np.pi*4/3), .5*np.sin(np.pi*4/3), 1.],
#    [-1, .5, -1.],
#    [-1, -.5, 1.],
#    [.5, .5, 1.],
#    [.5, -.5, 0.],
    [.25, 0., 0.],
    ]))

xs.append(np.array([
    [1., 0., 0.], 
    [np.cos(np.pi*2/3), np.sin(np.pi*2/3), 0.],
    [np.cos(np.pi*4/3), np.sin(np.pi*4/3), 0.],
#    [.5, 0., 0.], 
#    [.5*np.cos(np.pi*2/3), .5*np.sin(np.pi*2/3), 0.],
#    [.5*np.cos(np.pi*4/3), .5*np.sin(np.pi*4/3), 0.],
#    [-1, .5, -1.],
#    [-1, -.5, 1.],
#    [.5, .5, 1.],
#    [.5, -.5, 0.],
    [.25, 0., 1.],
    ]))



n_test1 = 5
n_test2 = 5
n_test = n_test1*n_test2
x_test1 = np.linspace(-1., 1., n_test1)
x_test2 = np.linspace(-1., 1., n_test2)
x_test_mesh = np.meshgrid(x_test2, x_test1)
x_test = np.dstack(x_test_mesh).reshape(-1, 2)
x_test = np.column_stack((x_test, np.zeros(n_test)))
#print(x_test.shape)


ys = []
ys.append(np.array([
    [-1., 0., 0.],
    [-np.cos(np.pi*2/3), -np.sin(np.pi*2/3), 0.],
    [-np.cos(np.pi*4/3), -np.sin(np.pi*4/3), 0.],
#    [-.5, 0., 0.],
#    [-.5*np.cos(np.pi*2/3), -.5*np.sin(np.pi*2/3), 0.],
#    [-.5*np.cos(np.pi*4/3), -.5*np.sin(np.pi*4/3), 0.],
#    [-1, .5, -1.],
#    [-1, -.5, 1.],
#    [.5, .5, 1.],
#    [.5, -.5, 0.],
    [0., 0., 0.]
    ]))

#ys.append(np.array([
#    [1., 0., 0.],
#    [-np.cos(np.pi*2/3), -np.sin(np.pi*2/3), 0.],
#    [-np.cos(np.pi*4/3), -np.sin(np.pi*4/3), 0.],
#    [0., 0., 1.],
#    ]))

#ys.append(np.array([
#    [1., 0., 0.],
#    [np.cos(np.pi*2/3), -np.sin(np.pi*2/3), 0.],
#    [-np.cos(np.pi*4/3), -np.sin(np.pi*4/3), 0.],
#    [0., 0., 1.]
#    ]))

sig_var = 1.
length_scale = 2.
noise_var = 0.01
gp = cov_div_free.cov_div_free(sig_var, length_scale, noise_var)
gp1 = cov_sq_exp.cov_sq_exp(sig_var, length_scale, noise_var)

max_loglik = None
max_loglik1 = None
i_x = 0
for x in xs:
    print("x:", x)
    i_y = 0
    for y in ys:
        y_flat = np.reshape(y, (3*len(y), -1))
        loglik = gp.init(x, y_flat)
        y_test_mean = gp.fit(x_test, calc_var = False)
        y_test_mean = np.reshape(y_test_mean, (n_test, -1))

        loglik1 = gp1.init(x, y_flat)
        y_test_mean1 = gp1.fit(x_test, calc_var = False)
        y_test_mean1 = np.reshape(y_test_mean1, (n_test, -1))
        
        print("y:", y)
        print("Log-likelihood:", loglik, loglik1)
        if max_loglik is None or loglik > max_loglik:
            max_loglik = loglik
            max_y = np.array(y)
            max_i_x = i_x
            max_i_y = i_y
        myplot = plot.plot(extent=None)
        #print(y_test_mean)
        #myplot.set_color_map('bwr')
        myplot.vectors(x[:,0], x[:,1], y[:,0], y[:,1], ax_index = [])
        myplot.vectors(x_test[:,0], x_test[:,1], y_test_mean[:,0], y_test_mean[:,1], ax_index = [], color = 'b')
        myplot.vectors(x_test[:,0], x_test[:,1], y_test_mean1[:,0], y_test_mean1[:,1], ax_index = [], color = 'g')
        myplot.colormap(y_test_mean[:, 2].reshape((n_test1, n_test2)))
        myplot.save("simple_test" + str(i_x) + "_" + str(i_y) + ".png")
        myplot.close()
        i_y += 1
    i_x += 1

myplot = plot.plot()
myplot.vectors(x[:,0], x[:,1], max_y[:,0], max_y[:,1], ax_index = [])
myplot.save("simple_test.png")
myplot.close()
print("------------------------------------")
print(max_y)
print(max_i_x, max_i_y)
