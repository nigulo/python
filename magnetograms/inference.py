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

n1 = 5
n2 = 6
n = n1*n2
x1_range = 1.0
x2_range = 1.0
x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x_mesh = np.meshgrid(x1, x2)
x = np.dstack(x_mesh).reshape(-1, 2)

nu1 = 4
nu2 = 4
u1 = np.linspace(0, x1_range, nu1)
u2 = np.linspace(0, x2_range, nu2)
u_mesh = np.meshgrid(u1, u2)
u = np.dstack(u_mesh).reshape(-1, 2)

print x_mesh

sig_var_train = 0.2
length_scale_train = 0.2
noise_var_train = 0.000001
m_train = 0.0

gp_train = GPR_div_free.GPR_div_free(sig_var_train, length_scale_train, noise_var_train)
K = gp_train.calc_cov(x, x, True)

for i in np.arange(0, n1):
    for j in np.arange(0, n2):
        assert(K[i, j]==K[j, i])

L = la.cholesky(K)
s = np.random.normal(0.0, 1.0, 2*n)

y = np.repeat(m_train, 2*n) + np.dot(L, s)

y = np.reshape(y, (n, 2))
y_orig = np.array(y)
print y_orig

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
fig.set_size_inches(6, 12)

ax1.set_title('True field')
ax2.set_title('Probabilistic')
ax3.set_title('Non-probabilistic')

y1 = np.cos(x_mesh[0])
y2 = np.sin(x_mesh[1])

ax1.quiver(x_mesh[0], x_mesh[1], y_orig[:,0], y_orig[:,1], units='width', color = 'k')
#ax2.quiver(x_mesh[0], x_mesh[1], y[:,0], y[:,1], units='width', color = 'k')
#qk = ax.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')

perf_null = 0.0
for i in np.arange(0, n):
    if np.random.uniform() < 0.5:
        y[i] = y[i]*-1
        perf_null += 1.0
perf_null /=n

m = 0.0
length_scale = 1.0
noise_var = 0.0001

print np.shape(x)
print np.shape(y)
print n


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


def get_W(u_mesh, us, xys):
    W = np.zeros((np.shape(xys)[0], np.shape(us)[0]))
    i = 0
    for (x, y) in xys:
        (u1s, u2s), (indices_x, indices_y) = get_closest(u_mesh[0][0,:], u_mesh[1][:,0], x, y)
        coefs = bilinear_interp(u1s, u2s, x, y)
        for j in np.arange(0, len(us)):
            found = True
            coef_ind = 0
            for u1 in u1s:
                if us[j][0] != u1:
                    found = False
                    break
                for u2 in u2s:
                    if us[j][1] != u2:
                        found = False
                        break
                    coef_ind += 1
            if found:
                W[i, j] = coefs[coef_ind]
        i += 1
    return W

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
    
gp = GPR_div_free.GPR_div_free(sig_var_train, length_scale_train, noise_var_train)
K = gp.calc_cov(x, x, data_or_test=True)
U = gp.calc_cov(u, u, data_or_test=True)
print K
print "x=", x
print "u=", u

W = np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
for i in np.arange(0, len(x1)*len(x2)):
    W1 = get_W(u_mesh, u, x)#, np.reshape(U[i,0::2], (len(u1), len(u2))))
    W2 = get_W(u_mesh, u, x)#, np.reshape(U[i,1::2], (len(u1), len(u2))))
    print np.shape(W), np.shape(W1), np.shape(W1)
    for j in np.arange(0, np.shape(W1)[1]):
        W[2*i,2*j] = W1[i, j]
        W[2*i,2*j+1] = W2[i, j]
        W[2*i+1,2*j] = W1[i, j]
        W[2*i+1,2*j+1] = W2[i, j]

def calc_loglik_approx(U, W, y):
    x, info = sparse.cg(W, y, x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None)
    L = la.cholesky(U)
    v = la.solve(L.T, x)
    return -0.5 * np.dot(v.T, v) - sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)
    
def calc_loglik(K, y):
    L = la.cholesky(K)
    v = la.solve(L.T, y)
    return -0.5 * np.dot(v.T, v) - sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)
    

print "Approx: ", calc_loglik_approx(U, W, x)
print "True:", calc_loglik(K, x)

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

def algorithm_a(x, y, y_orig):
    y_in = np.array(y)
    loglik = None
    max_loglik = None

    #thetas = random.uniform(size=n)
    thetas = np.ones(n)/2
    thetas = np.log(thetas)
    num_tries = 0
    
    while (max_loglik is None or num_tries % max_num_tries != 0 or (loglik < max_loglik) or (loglik > max_loglik + eps)):
        print "num_tries", num_tries
    
        num_tries += 1
    
        initial_param_values = []
        
        if inference:
            #for i in np.arange(0, num_chains):
            #    initial_m = m
            #    initial_length_scale = length_scale
            #    initial_param_values.append(dict(m=initial_m))
            
            fit = model.sampling(data=dict(x=x,N=n,y=y,noise_var=noise_var), init=initial_param_values, iter=num_iters, chains=num_chains, n_jobs=n_jobs)
            
            results = fit.extract()
            loglik_samples = results['lp__']
            print loglik_samples 
            loglik = np.mean(loglik_samples)
            
            length_scale_samples = results['length_scale'];
            length_scale = np.mean(length_scale_samples)
            
            sig_var_samples = results['sig_var']
            sig_var = np.mean(sig_var_samples)
            
            m_samples = results['m'];
            m = np.mean(m_samples)
        else:
            sig_var=sig_var_train
            m=m_train
            length_scale=length_scale_train
            gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
            loglik = gp.init(x, y)
        
        print "sig_var=", sig_var
        print "length_scale", length_scale
        print "m", m
        print "loglik=", loglik, "max_loglik=", max_loglik
        
        if max_loglik is None or loglik > max_loglik:
            num_tries = 1
            max_loglik = loglik
        
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)

        y_last = np.array(y)
        for i in np.random.choice(n, size=1, replace=False):
            js = []
            for j in np.arange(0, n):
                x_diff = x[j] - x[i]
                if (np.dot(x_diff, x_diff) < length_scale**2/4):
                    js.append(j)
            for j in js:
                y[j] = np.array(y_in[j])
            loglik1 = gp.init(x, y)
            for j in js:
                y[j] = np.array(y_in[j])*-1
            loglik2 = gp.init(x, y)

            if (loglik1 > loglik or loglik2 > loglik):    
                thetas_i = thetas[i]
                exp_theta = np.exp(thetas_i)
                #new_theta = loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]))
                #thetas[i] = new_theta
                new_theta = thetas_i + loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]), b=np.array([exp_theta, 1.0-exp_theta]))
                thetas[i] += learning_rate * (new_theta - thetas_i)
    
                thetas_i = thetas[i]
        
                print thetas_i, np.exp(thetas_i), loglik1, loglik2
                #r = np.log(random.uniform())
                for j in js:
                    thetas[j] = thetas_i
                    if thetas_i > np.log(0.5):#r < thetas_i
                        #assert(loglik1 >= loglik2)
                        y[j] = np.array(y_in[j])
                    else:
                        #assert(loglik2 >= loglik1)
                        y[j] = np.array(y_in[j])*-1
                        
                #if thetas_i > np.log(0.5):#r < thetas_i
                #    for j in js:
                #        thetas[j] = thetas_i
                #        y[j][0] = -y[j][0]
                #        y[j][1] = -y[j][1]
                #else:
                #    for j in js:
                #        thetas[j] = np.log(1.0 - np.exp(thetas_i))
            else:
                y = y_last
            
    
    num_guessed = 0.0
    for i in np.arange(0, n):
        if np.array_equal(y[i], y_orig[i]):
            num_guessed += 1.0

    exp_thetas = np.exp(thetas)
    if num_guessed < n/2:
        num_guessed = n - num_guessed
        y *= -1
        exp_thetas = np.ones(n) - exp_thetas

    return num_guessed/n, exp_thetas, y
    
def algorithm_b(x, y, y_orig):
    loglik = None
    max_loglik = None

    num_tries = 0
    
    while (max_loglik is None or num_tries % max_num_tries != 0 or (loglik > max_loglik + eps)):
        print "num_tries", num_tries
    
        num_tries += 1
    
        initial_param_values = []
        
        if inference:
            for i in np.arange(0, num_chains):
                initial_m = m
                initial_length_scale = length_scale
                initial_param_values.append(dict(m=initial_m))
            
            fit = model.sampling(data=dict(x=x,N=n,y=y,noise_var=noise_var), init=initial_param_values, iter=num_iters, chains=num_chains, n_jobs=n_jobs)
            
            results = fit.extract()
            loglik_samples = results['lp__']
            print loglik_samples 
            loglik = np.mean(loglik_samples)
            
            length_scale_samples = results['length_scale'];
            length_scale = np.mean(length_scale_samples)
            
            sig_var_samples = results['sig_var']
            sig_var = np.mean(sig_var_samples)
            
            m_samples = results['m'];
            m = np.mean(m_samples)
        else:
            sig_var=sig_var_train
            m=m_train
            length_scale=length_scale_train
            gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
            loglik = gp.init(x, y)
        
        print "sig_var=", sig_var
        print "length_scale", length_scale
        print "m", m
        print "loglik=", loglik, "max_loglik=", max_loglik
        
        if max_loglik is None or loglik > max_loglik:
            num_tries = 1
            max_loglik = loglik
        
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        for i in np.random.choice(n, size=1, replace=False):
            loglik1 = loglik#gp.init(x, y)
            js = []
            for j in np.arange(0, n):
                x_diff = x[j] - x[i]
                if (np.dot(x_diff, x_diff) < length_scale**2/4):
                    y[j] = y[j]*-1
                    js.append(j)
            loglik2 = gp.init(x, y)
            for j in js:
                if loglik1 > loglik2:        
                    y[j] = y[j]*-1
    
        
    
    num_guessed = 0.0
    for i in np.arange(0, n):
        if np.array_equal(y[i], y_orig[i]):
            num_guessed += 1.0

    if num_guessed < n/2:
        num_guessed = n - num_guessed
        y *= -1
   
    return num_guessed/n, y


print "******************** Algorithm a ********************"
perf_a, prob_a, field_a = algorithm_a(x, np.array(y), y_orig)
print "******************** Algorithm b ********************"
perf_b, field_b = algorithm_b(x, np.array(y), y_orig)

print "Results:"
print perf_null
print perf_a, prob_a#np.mean(np.abs(np.ones(n)/2 - prob_a)*2)
print perf_b

indices = np.unique(np.where(field_a - y_orig == np.array([0, 0]))[0])
x_a_correct = x[indices]
field_a_correct = field_a[indices]
indices = np.unique(np.where(field_a - y_orig != np.array([0, 0]))[0])
x_a_incorrect = x[indices]
field_a_incorrect = field_a[indices]

Q_a_1 = ax2.quiver(x_a_correct[:,0], x_a_correct[:,1], field_a_correct[:,0], field_a_correct[:,1], units='width', color='g', linestyle=':')
Q_a_2 = ax2.quiver(x_a_incorrect[:,0], x_a_incorrect[:,1], field_a_incorrect[:,0], field_a_incorrect[:,1], units='width', color='r', linestyle=':')

indices = np.unique(np.where(field_b - y_orig == np.array([0, 0]))[0])
x_b_correct = x[indices]
field_b_correct = field_b[indices]
indices = np.unique(np.where(field_b - y_orig != np.array([0, 0]))[0])
x_b_incorrect = x[indices]
field_b_incorrect = field_b[indices]

Q_a_1 = ax3.quiver(x_b_correct[:,0], x_b_correct[:,1], field_b_correct[:,0], field_b_correct[:,1], units='width', color='g', linestyle=':')
Q_a_2 = ax3.quiver(x_b_incorrect[:,0], x_b_incorrect[:,1], field_b_incorrect[:,0], field_b_incorrect[:,1], units='width', color='r', linestyle=':')

fig.savefig("field.png")
