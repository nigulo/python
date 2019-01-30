import sys
sys.path.append('../')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
#import pickle
import numpy as np
#import pylab as plt
#import pandas as pd
import GPR_div_free as GPR_div_free
import scipy.misc
import numpy.random as random
import scipy.interpolate as interp
import scipy.sparse.linalg as sparse
import utils
import pymc3 as pm
import theano.tensor as tt
#import os
#import os.path
#from scipy.stats import gaussian_kde
#from sklearn.cluster import KMeans
import numpy.linalg as la
import matplotlib.pyplot as plt
import warnings

num_samples = 1
num_chains = 3
inference = False

eps = 0.001
learning_rate = 1.0
max_num_tries = 20
import theano
import theano.tensor as tt

if len(sys.argv) > 3:
    num_iters = int(sys.argv[3])
if len(sys.argv) > 4:
    num_chains = int(sys.argv[4])

n_jobs = num_chains

#model = pickle.load(open('model.pkl', 'rb'))

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

m1 = 10
m2 = 10
m = m1 * m2
u1 = np.linspace(0, x1_range, m1)
u2 = np.linspace(0, x2_range, m2)
u_mesh = np.meshgrid(u1, u2)
u = np.dstack(u_mesh).reshape(-1, 2)

print("u_mesh=", u_mesh)
print("u=", u)

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
#            if found:print
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

print(x_mesh)

sig_var_train = 0.2
length_scale_train = 0.2
noise_var_train = 0.000001
mean_train = 0.0

gp_train = GPR_div_free.GPR_div_free(sig_var_train, length_scale_train, noise_var_train)
K = gp_train.calc_cov(x, x, True)
#U = gp_train.calc_cov(u, u, data_or_test=True)
#W = calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
#K = np.dot(W.T, np.dot(U, W))

print("SIIN")
for i in np.arange(0, n1):
    for j in np.arange(0, n2):
        assert(K[i, j]==K[j, i])

L = la.cholesky(K)
s = np.random.normal(0.0, 1.0, 2*n)

y = np.repeat(mean_train, 2*n) + np.dot(L, s)

y = np.reshape(y, (n, 2))
y_orig = np.array(y)
print(y_orig)

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
y_flat = np.reshape(y, (2*n, -1))

mean = 0.0
length_scale = 1.0
noise_var = 0.0001

print(np.shape(x))
print(np.shape(y))
print(n)

# Using KISS-GP
# https://arxiv.org/pdf/1503.01057.pdf

# define the parameters with their associated priors


"""
# The codef for user-defined likelihood function and 
# it's derivateives taken from:
# https://docs.pymc.io/notebooks/blackbox_external_likelihood.html
"""

def gradients(vals, func, releps=1e-3, abseps=None, mineps=1e-9, reltol=1e-3,
              epsscale=0.5):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.
    releps: float, array_like, 1e-3
        The initial relative step size for calculating the derivative.
    abseps: float, array_like, None
        The initial absolute step size for calculating the derivative.
        This overrides `releps` if set.
        `releps` is set then that is used.
    mineps: float, 1e-9
        The minimum relative step size at which to stop iterations if no
        convergence is achieved.
    epsscale: float, 0.5
        The factor by which releps if scaled in each iteration.

    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    grads = np.zeros(len(vals))

    # maximum number of times the gradient can change sign
    flipflopmax = 10.

    # set steps
    if abseps is None:
        if isinstance(releps, float):
            eps = np.abs(vals)*releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
            teps = releps*np.ones(len(vals))
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
            teps = releps
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")
    else:
        if isinstance(abseps, float):
            eps = abseps*np.ones(len(vals))
        elif isinstance(abseps, (list, np.ndarray)):
            if len(abseps) != len(vals):
                raise ValueError("Problem with input absolute step sizes")
            eps = np.array(abseps)
        else:
            raise RuntimeError("Absolute step sizes are not a recognised type!")
        teps = eps

    # for each value in vals calculate the gradient
    count = 0
    for i in range(len(vals)):
        # initial parameter diffs
        leps = eps[i]
        cureps = teps[i]

        flipflop = 0

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5*leps  # change forwards distance to half eps
        bvals[i] -= 0.5*leps  # change backwards distance to half eps
        cdiff = (func(fvals)-func(bvals))/leps

        while 1:
            fvals[i] -= 0.5*leps  # remove old step
            bvals[i] += 0.5*leps

            # change the difference by a factor of two
            cureps *= epsscale
            if cureps < mineps or flipflop > flipflopmax:
                # if no convergence set flat derivative (TODO: check if there is a better thing to do instead)
                warnings.warn("Derivative calculation did not converge: setting flat derivative.")
                grads[count] = 0.
                break
            leps *= epsscale

            # central difference
            fvals[i] += 0.5*leps  # change forwards distance to half eps
            bvals[i] -= 0.5*leps  # change backwards distance to half eps
            cdiffnew = (func(fvals)-func(bvals))/leps

            if cdiffnew == cdiff:
                grads[count] = cdiff
                break

            # check whether previous diff and current diff are the same within reltol
            rat = (cdiff/cdiffnew)
            if np.isfinite(rat) and rat > 0.:
                # gradient has not changed sign
                if np.abs(1.-rat) < reltol:
                    grads[count] = cdiffnew
                    break
                else:
                    cdiff = cdiffnew
                    continue
            else:
                cdiff = cdiffnew
                flipflop += 1
                continue

        count += 1

    return grads

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, noise_var, y):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.noise_var = noise_var
        self.y = y
        self.logpgrad = LogLikeGrad(self.likelihood, self.noise_var, self.y)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.noise_var, self.y)

        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        theta, = inputs  # our parameters
        return [g[0]*self.logpgrad(theta)]

class LogLikeGrad(tt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, noise_var, y):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.noise_var = noise_var
        self.y = y

    def perform(self, node, inputs, outputs):
        theta, = inputs

        # define version of likelihood function to pass to derivative function
        def lnlike(values):
            return self.likelihood(values, self.noise_var, self.y)

        # calculate gradients
        grads = gradients(theta, lnlike)

        outputs[0][0] = grads
        
def sample(x, y):
    W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
    #(x, istop, itn, normr) = sparse.lsqr(W, y)[:4]#, x0=None, tol=1e-05, maxiter=None, M=None, callback=None)
    
    def likelihood(theta, noise_var, y):
        ell = theta[0]
        sig_var = theta[1]
        gp = GPR_div_free.GPR_div_free(sig_var, ell, noise_var)
        #loglik = gp.init(x, y)
        
        U = gp.calc_cov(u, u, data_or_test=True)
        (x, istop, itn, normr) = sparse.lsqr(W, y)[:4]#, x0=None, tol=1e-05, maxiter=None, M=None, callback=None)
        #print x
        L = la.cholesky(U)
        #print L
        v = la.solve(L, x)
        return -0.5 * np.dot(v.T, v) - sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)

    with pm.Model() as model:
        ell = pm.Uniform('ell', 0.0, 1.0)
        sig_var = pm.Uniform('sig_var', 0.0, 1.0)

        logl = LogLike(likelihood, noise_var_train, y)

        theta = tt.as_tensor_variable([ell, sig_var])
        #data = pymc.MvNormal('data', mu=np.zeros(N), tau=tau, value=y, observed=True)
        like = pm.DensityDist('like', lambda v: logl(v), observed={'v': theta})
        
        #sampler = pymc.MCMC([ell, sig_var, noise_var, x_train, y_train])
        #sampler.use_step_method(pymc.AdaptiveMetropolis, [ell, sig_var, noise_var],
        #                        scales={ell:1.0, sig_var:1.0, noise_var:1.0})
        
        step = pm.NUTS()
        trace = pm.sample(num_samples, tune=int(num_samples/2), init=None, step=step, cores=num_chains)

        #print(trace['model_logp'])
        m_ell = np.mean(trace['ell'])
        m_sig_var = np.mean(trace['sig_var'])
        return m_ell, m_sig_var

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


def algorithm_a(x, y, y_orig):
    y_in = np.array(y)
    loglik = None
    max_loglik = None

    #thetas = random.uniform(size=n)
    thetas = np.ones(n)/2
    thetas = np.log(thetas)
    num_tries = 0
    
    while (max_loglik is None or num_tries % max_num_tries != 0 or (loglik < max_loglik) or (loglik > max_loglik + eps)):
        print("num_tries", num_tries)
    
        num_tries += 1
    
        initial_param_values = []
        
        if inference:
            #for i in np.arange(0, num_chains):
            #    #initial_m = m
            #    #initial_length_scale = length_scale
            #    #initial_param_values.append(dict(m=initial_m))
            #    initial_param_values.append(dict())
            #
            #fit = model.sampling(data=dict(x=x,N=n,y=y,noise_var=noise_var), init=initial_param_values, iter=num_iters, chains=num_chains, n_jobs=n_jobs)
            #
            #results = fit.extract()
            #loglik_samples = results['lp__']
            #print loglik_samples 
            #loglik = np.mean(loglik_samples)
            #
            #length_scale_samples = results['length_scale'];
            #length_scale = np.mean(length_scale_samples)
            #
            #sig_var_samples = results['sig_var']
            #sig_var = np.mean(sig_var_samples)
            #
            #mean_samples = results['m'];
            #mean = np.mean(mean_samples)
            
            length_scale, sig_var = sample(x, np.reshape(y, (2*n, -1)))
        else:
            sig_var=sig_var_train
            #mean=mean_train
            length_scale=length_scale_train

        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        #loglik = gp.init(x, y)
        U = gp.calc_cov(u, u, data_or_test=True)
        W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
        loglik = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
            
        
        print("sig_var=", sig_var)
        print("length_scale", length_scale)
        #print("mean", mean)
        print("loglik=", loglik, "max_loglik=", max_loglik)
        
        if max_loglik is None or loglik > max_loglik:
            num_tries = 1
            max_loglik = loglik
        
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        U = gp.calc_cov(u, u, data_or_test=True)
        W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))

        y_last = np.array(y)
        for i in np.random.choice(n, size=1, replace=False):
            js = []
            for j in np.arange(0, n):
                x_diff = x[j] - x[i]
                if (np.dot(x_diff, x_diff) < length_scale**2/4):
                    js.append(j)
            for j in js:
                y[j] = np.array(y_in[j])
            #loglik1 = gp.init(x, y)
            loglik1 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
            for j in js:
                y[j] = np.array(y_in[j])*-1
            #loglik2 = gp.init(x, y)
            loglik2 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))

            if (loglik1 > loglik or loglik2 > loglik):    
                thetas_i = thetas[i]
                exp_theta = np.exp(thetas_i)
                #new_theta = loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]))
                #thetas[i] = new_theta
                new_theta = thetas_i + loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]), b=np.array([exp_theta, 1.0-exp_theta]))
                thetas[i] += learning_rate * (new_theta - thetas_i)
    
                thetas_i = thetas[i]
        
                print(thetas_i, np.exp(thetas_i), loglik1, loglik2)
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
        print("num_tries", num_tries)
    
        num_tries += 1
    
        #initial_param_values = []
        
        if inference:
            #for i in np.arange(0, num_chains):
            #    initial_m = mean
            #    initial_length_scale = length_scale
            #    initial_param_values.append(dict(m=initial_m))
            #
            #fit = model.sampling(data=dict(x=x,N=n,y=y,noise_var=noise_var), init=initial_param_values, iter=num_iters, chains=num_chains, n_jobs=n_jobs)
            #
            #results = fit.extract()
            #loglik_samples = results['lp__']
            #print loglik_samples 
            #loglik = np.mean(loglik_samples)
            #
            #length_scale_samples = results['length_scale'];
            #length_scale = np.mean(length_scale_samples)
            #
            #sig_var_samples = results['sig_var']
            #sig_var = np.mean(sig_var_samples)
            #
            #mean_samples = results['m'];
            #mean = np.mean(mean_samples)
            
            length_scale, sig_var = sample(x, np.reshape(y, (2*n, -1)))
            
        else:
            sig_var=sig_var_train
            #mean=mean_train
            length_scale=length_scale_train

        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        #loglik = gp.init(x, y)
        U = gp.calc_cov(u, u, data_or_test=True)
        W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
        loglik = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
        
        print("sig_var=", sig_var)
        print("length_scale", length_scale)
        #print("mean", mean)
        print("loglik=", loglik, "max_loglik=", max_loglik)
        
        if max_loglik is None or loglik > max_loglik:
            num_tries = 1
            max_loglik = loglik
        
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        U = gp.calc_cov(u, u, data_or_test=True)
        W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
        
        
        if False:
            # Select n/2 random indices and filter out those too close to each other
            random_indices = np.random.choice(n, size=n/2, replace=False)
            i = 0
            while i < len(random_indices):
                random_index_filter = np.ones_as(random_indices, dtype=bool)
                ri = random_indices[i]
                for j in np.arange(i + 1, len(random_indices)):
                    rj = random_indices[j]
                    x_diff = x[rj] - x[ri]
                    if (np.dot(x_diff, x_diff) < length_scale**2/4):
                        random_index_filter[j] = False
                random_indices = random_indices[random_index_filter]
                i += 1
    
            loglik1 = loglik
            y_saved = np.array(y)
            for ri in random_indices:
                y[ri] = y[ri]*-1
            loglik2 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
            if loglik1 > loglik2:
                y = y_saved
        else:
            for i in np.random.choice(n, size=1, replace=False):
                loglik1 = loglik#gp.init(x, y)
                js = []
                for j in np.arange(0, n):
                    x_diff = x[j] - x[i]
                    if (np.dot(x_diff, x_diff) < length_scale**2/4):
                        y[j] = y[j]*-1
                        js.append(j)
                #loglik2 = gp.init(x, y)
                loglik2 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
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


print("******************** Algorithm a ********************")
perf_a, prob_a, field_a = algorithm_a(x, np.array(y), y_orig)
print("******************** Algorithm b ********************")
perf_b, field_b = algorithm_b(x, np.array(y), y_orig)

print("Results:")
print(perf_null)
print(perf_a, prob_a)#np.mean(np.abs(np.ones(n)/2 - prob_a)*2)
print(perf_b)

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
