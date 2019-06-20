import numpy as np
import scipy.optimize as optimize
import numpy.fft as fft

class tip_tilt:
    
    '''
        D and S are tensors with index l, k and x1, x2, where
        l is number od frames, k, number of diversities and
        x1 and x2 are width and height of the sensor in pixels
    '''
    def __init__(self, D, S, F, x, initial_a = None, prior_prec=.01, num_rounds=1):
        self.D_in = D
        self.S = S
        self.C = np.absolute(S)*np.absolute(D)*np.absolute(F)
        self.D = np.angle(D)-np.angle(S)-np.angle(F)
        self.C_T = np.transpose(self.C, axes=(1, 0, 2, 3)) # swap k and l
        self.D_T = np.transpose(self.D, axes=(1, 0, 2, 3)) # swap k and l
        self.C1 = np.sum(self.C, axis = (0, 1))
        self.non_zero_inds = np.where(self.C1 != 0.)
        self.zero_inds = np.where(self.C1 == 0.)
        self.C1 = 1./self.C1[self.non_zero_inds]
        self.C2 = np.sum(self.C, axis = 1)
        self.CD1 = np.sum(self.C*self.D, axis = (0, 1))
        self.CD2 = np.sum(self.C*self.D, axis = 1)
        self.x_in = x
        self.x = np.roll(np.roll(x, -int(x.shape[0]/2), axis=0), -int(x.shape[1]/2), axis=1)
        self.L = D.shape[0]
        self.K = D.shape[1]
        self.initial_a = initial_a
        
        if self.initial_a is not None:
            self.initial_a = self.initial_a.reshape(self.L*2)
        #self.initial_a = np.zeros(self.L, 2)
    
        self.prior_prec = prior_prec
        self.num_rounds = num_rounds

    def lik(self, theta, data):
        a = theta[0:2*self.L].reshape((self.L, 2))
        a0 = theta[2*self.L:2*self.L+2]
        #f = theta[self.L:self.L+self.x.shape[0]*self.x.shape[1]].reshape((self.x.shape[0], self.x.shape[1]))
        au = np.tensordot(a, self.x, axes=(1, 2)) + np.tensordot(a0, self.x, axes=(0, 2))
        #f = self.get_f(a)
        phi = self.D_T - au# - f
        val = np.sum(self.C_T*np.cos(phi))
        val += np.sum(a*a)*self.prior_prec/2
        #print(val)
        return val
    
    '''
    def lik_grad(self, theta, data):
        a = theta[0:2*self.L].reshape((self.L, 2))
        #f = theta[self.L:self.L+self.x.shape[0]*self.x.shape[1]].reshape((self.x.shape[0], self.x.shape[1]))
        au = np.tensordot(a, self.x, axes=(1, 2))

        phi = self.D_T - au
        sin_phi = np.sin(phi)

        val = np.zeros_like(self.C_T)
        val = self.C_T*(sin_phi)
        val1 = np.sum(np.tensordot(val, self.x[:,:,0], axes=([2,3], [0,1])), axis=0)
        val2 = np.sum(np.tensordot(val, self.x[:,:,1], axes=([2,3], [0,1])), axis=0)

        retval = np.concatenate((val1, val2))
        
        return retval
    '''

    def lik_grad(self, theta, data):
        a = theta[0:2*self.L].reshape((self.L, 2))
        a0 = theta[2*self.L:2*self.L+2]

        au = np.tensordot(a, self.x, axes=(1, 2)) + np.tensordot(a0, self.x, axes=(0, 2))
        phi = self.D_T - au
        sin_phi = np.sin(phi)

        #val = np.zeros(num.shape[0]*num.shape[1])
        val = np.zeros_like(self.C_T)
        #print("AAAA", self.C_T.shape, den.shape, cos_f.shape, sin_phi[:,:,indices_not_null].shape, tan_f.shape, cos_phi[:,:,indices_not_null].shape)
        val = self.C_T*sin_phi
        #val = np.reshape(val, (num.shape[0], num.shape[1]))
        #print(val.shape)
        val1 = np.sum(np.tensordot(val, self.x[:,:,0], axes=([2,3], [0,1])), axis=0)
        val2 = np.sum(np.tensordot(val, self.x[:,:,1], axes=([2,3], [0,1])), axis=0)

        
        val01 = np.sum(val1)
        val02 = np.sum(val2)


        retval = np.concatenate((np.column_stack((val1, val2)).reshape(self.L*2), np.array([val01, val02])))
        
        #print(retval)
        return retval
    
    '''
    # Functions related to finding the solution directly (not used)
    def get_f(self, a):
        eps = 1e-10

        au = np.tensordot(a, self.x, axes=(1, 2))
        phi = self.D_T - au
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        num = np.sum(self.C_T*sin_phi, axis=(0, 1))
        den = (np.sum(self.C_T*cos_phi, axis=(0, 1)))
        indices_not_null = den != 0.

        f = np.ones((self.x.shape[0], self.x.shape[1]))*np.pi/2
        f[indices_not_null] = np.arctan(num[indices_not_null]/den[indices_not_null])
        return f

        #tan_f = (np.sum(self.C_T*sin_phi, axis=(0, 1)) + eps)/((np.sum(self.C_T*cos_phi, axis=(0, 1))) + eps)
        #return np.arctan(tan_f)        
        #return np.zeros((self.x.shape[0], self.x.shape[0]))
        #return np.sum(self.C_T, axis=(0,1))/(self.L*self.K)

    def fun(self, a_in, a_old_in=None):
        a = a_in.reshape((self.L, 2))
        if a_old_in is None:
            a_old = a
        else:
            a_old = a_old_in.reshape((self.L, 2))
        #eps = 1e-10

        au_old = np.tensordot(a_old, self.x, axes=(1, 2))
        phi_old = self.D_T - au_old
        sin_phi_old = np.sin(phi_old)
        cos_phi_old = np.cos(phi_old)

        den = (np.sum(self.C_T*cos_phi_old, axis=(0, 1)))
        indices_not_null = den != 0.
        num = np.sum(self.C_T*sin_phi_old, axis=(0, 1))
        tan_f = num[indices_not_null]/den[indices_not_null]
        cos_f = 1./np.sqrt(1.+tan_f*tan_f)
        #print("tan_f:", tan_f)
        #print("cos_f:", cos_f)

        au = np.tensordot(a, self.x, axes=(1, 2))
        phi = self.D_T - au
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        #val = np.zeros(num.shape[0]*num.shape[1])
        val = np.zeros_like(self.C_T)
        #print("AAAA", self.C_T.shape, den.shape, cos_f.shape, sin_phi[:,:,indices_not_null].shape, tan_f.shape, cos_phi[:,:,indices_not_null].shape)
        val[:,:,indices_not_null] = self.C_T[:,:,indices_not_null]*cos_f*(sin_phi[:,:,indices_not_null] - tan_f*cos_phi[:,:,indices_not_null])
        #val = np.reshape(val, (num.shape[0], num.shape[1]))
        #print(val.shape)
        val1 = np.sum(np.tensordot(val, self.x[:,:,0], axes=([2,3], [0,1])), axis=0)
        val2 = np.sum(np.tensordot(val, self.x[:,:,1], axes=([2,3], [0,1])), axis=0)

        retval = np.concatenate((val1, val2))
        
        #print(retval)
        return retval
    
    def second_deriv(self, a_in, a_old_in):
        a = a_in.reshape((self.L, 2))
        a_old = a_old_in.reshape((self.L, 2))
        eps = 1e-10

        au_old = np.tensordot(a_old, self.x, axes=(1, 2))
        phi_old = self.D_T - au_old
        sin_phi_old = np.sin(phi_old)
        cos_phi_old = np.cos(phi_old)

        tan_f = (np.sum(self.C_T*sin_phi_old, axis=(0, 1)) + eps)/((np.sum(self.C_T*cos_phi_old, axis=(0, 1))) + eps)
        cos_f = 1./np.sqrt(1.+tan_f*tan_f)
        
        au = np.tensordot(a, self.x, axes=(1, 2))
        phi = self.D_T - au
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        val = self.C_T*cos_f*(cos_phi + tan_f*sin_phi)
        #print(val.shape)
        val1 = np.sum(np.tensordot(val, self.x[:,:,0]**2, axes=([2,3], [0,1])), axis=0)
        val2 = np.sum(np.tensordot(val, self.x[:,:,1]**2, axes=([2,3], [0,1])), axis=0)

        retval = np.concatenate((val1, val2))
        
        #print(retval)
        return retval
    
    def grid_search(self, a0, a1, a_opt = None, min_val = None, precision = 1000):
        if a_opt is None:
            a_opt = a0
        if min_val is None:
            min_val = np.ones_like(a0)*np.finfo('d').max
        #val = np.abs(self.fun(a_opt, a_opt))
        val = np.abs(self.lik_grad(a_opt, []))
        print("VALLLLLLLLLLL", val)
        for a in np.linspace(a0, a1, precision):
            val = np.abs(self.fun(a, a_opt))
            #print("val, min_val", val, min_val)

            indices = np.where(val < min_val)[0]
            #val2 = self.second_deriv(a, a_opt)
            a_opt_new = np.array(a_opt)
            for i in indices:
                #print(val2)
                #if (val2[i] < 0.):
                min_val[i] = val[i]
                a_opt_new[i] = a[i]
                #else:
                #    print("BLAAA")
            #val = np.abs(self.fun3(a, a_opt_new))
            #indices1 = np.where(val < min_val)[0]
            #if np.all(indices == indices1):
                #print("Juhuuu")
            a_opt = a_opt_new
                
            
                
            #val2 = self.second_deriv(a_opt_new)
            #if (np.all(val2 < 0.)):
            #a_opt = a_opt_new
                
            #min_val[indices] = val[indices]
            #a_opt[indices] = a[indices]
            #print("min_val", min_val, a_opt)
        return a_opt, min_val
                
    def calc(self):
        #for l in np.arange(0, self.D.shape[0]):
        #res = optimize.root(self.fun, x0=np.zeros((self.L, 2)), args=(), method='lm', jac=None, tol=None, callback=None, options=None)
        #res = optimize.root(self.fun2, x0=np.zeros((self.L*2+self.x.shape[0]*self.x.shape[1])), args=(), method='hybr', jac=None, tol=None, callback=None, options=None)
        #res = optimize.fsolve(self.fun2, x0=np.zeros((self.L*2+self.x.shape[0]*self.x.shape[1])), args=(), fprime=None)
        #
        #res = optimize.fsolve(self.fun3, x0=np.zeros(self.L*2), args=(), fprime=None)
        #print("RES", res)
        res = self.initial_a
        min_val = None
        precision = 10000
        eps = 1e-10
        d = 20.
        res, min_val = self.grid_search(np.ones(self.L*2)*d*-1., np.ones(self.L*2)*d, res, min_val, precision)
        while np.any(min_val > eps):#1./precision):
            res_old = res
            #res, min_val = self.grid_search(np.ones(self.L*2)*-20, np.ones(self.L*2)*20, res, min_val, precision)
            res, min_val = self.grid_search(res - d, res + d, res, min_val, precision)
            res_diff = np.sqrt(np.sum((res-res_old)**2))
            print("MIN_VAL", np.sum(min_val), res, d, res_diff)
            if res_diff == 0:
                d *= .9
        val2 = self.second_deriv(res, res)
        #np.testing.assert_array_less(val2, np.zeros_like(val2))
        a = res.reshape((self.L, 2))
        f = self.get_f(a)
        print("a, f", a, f)
        
        #test_val = self.fun3(res)
        #print("test_val", test_val)
        
        return (a, f)
    '''
    
    def optimize(self):
        def lik_fn(params):
            return self.lik(params, [])

        def grad_fn(params):
            return self.lik_grad(params, [])
        
        min_loglik = None
        min_res = None
        for trial_no in np.arange(0, self.num_rounds):
            initial_a = np.random.normal(size=2*(self.L+1), scale=1./np.sqrt(self.prior_prec + 1e-10))#np.zeros(2*self.L)
            if self.initial_a is not None:
                initial_a = self.initial_a
            from timeit import default_timer as timer
            #start = timer()
            #res1 = optimize.fmin_cg(lik_fn, initial_a, fprime=None, args=(), full_output=True, gtol=1e-05, norm=np.Inf, epsilon=1.5e-08)
            #end = timer()
            #print("Without grad:", end - start) # Time in seconds, e.g. 5.38091952400282
            start = timer()
            res = optimize.fmin_cg(lik_fn, initial_a, fprime=grad_fn, args=(), full_output=True, gtol=1e-05, norm=np.Inf, epsilon=1.5e-08)
            end = timer()
            print("With grad:", end - start) # Time in seconds, e.g. 5.38091952400282
            #np.testing.assert_array_almost_equal(res1, res, 1)
            #res2 = optimize.fmin_cg(lik_fn, initial_a, fprime=None, args=(), full_output=True)
            #res2 = optimize.fmin_cg(lik_fn, initial_a, fprime=grad_fn, args=(), full_output=True)
            #res = optimize.fmin_l_bfgs_b(lik_fn, initial_a, fprime=None, args=(), approx_grad=True, bounds=[(-20, 20)]*2*self.L)
            #print("res1, res2", res1, res2)
            #lower_bounds = np.zeros(jmax*2)
            #upper_bounds = np.ones(jmax*2)*1e10
            #res = scipy.optimize.minimize(lik_fn, np.random.normal(size=jmax*2), method='L-BFGS-B', jac=grad_fn, bounds = zip(lower_bounds, upper_bounds), options={'disp': True, 'gtol':1e-7})
            loglik = res[1]
            #assert(loglik == lik_fn(res['x']))
            if min_loglik is None or loglik < min_loglik:
                min_loglik = loglik
                min_res = res
        a_est = min_res[0].reshape((self.L+1, 2))
        print("a_est", a_est, min_loglik)
        print("a_est_mean", np.mean(a_est, axis=0))
        return a_est[:self.L,:], a_est[self.L,:]#self.get_f(a_est)
    
    def calc(self):
        a, a0 = self.optimize()
        
        image_F = np.zeros((self.L, self.x.shape[0], self.x.shape[1]), dtype = 'complex')
        image = np.zeros((self.L, self.x.shape[0], self.x.shape[1]))
        S = np.zeros((self.L, self.K, self.x.shape[0], self.x.shape[1]), dtype = 'complex')
        for trial in np.arange(0, self.L):
            #tt = tip_tilt.tip_tilt(np.array([Ds[trial]]), np.array([Ps[trial]]), np.array([Fs[trial]]), psf_.coords1)
            #a, f = tt.calc()
            #tt_phase = np.exp(1.j*np.tensordot(psf_.coords1, a[0], axes=(2, 0)))
        
            #tt_phase = np.exp(1.j*np.tensordot(psf_.coords1, a[trial], axes=(2, 0)))
            tt_phase = np.exp(1.j*(np.tensordot(self.x, -a[trial], axes=(2, 0))))
            #tt_phase = np.exp(1.j*np.tensordot(np.ones_like(psf_.coords1), np.array([100., 0.]), axes=(2, 0)))
            
            #tt_phase1 = np.zeros((psf_.coords1.shape[0], psf_.coords1.shape[1]), dtype='complex')
            #tt_phase1=np.exp(1.j*tt_phase1)
            #np.testing.assert_array_equal(tt_phase, tt_phase1)
            S[trial] = self.S[trial]
            S[trial] *= tt_phase
            
            image_F[trial] = self.D_in[trial, 0]*tt_phase
            image[trial] = fft.ifft2(image_F[trial]).real
        return image, image_F, S
    
def main():
    '''
    D = np.loadtxt("D.txt", dtype="complex")
    D_d = np.loadtxt("D_d.txt", dtype="complex")
    D = np.array([np.stack((D, D_d))])
    
    S = np.loadtxt("P.txt", dtype="complex")
    S_d = np.loadtxt("P_d.txt", dtype="complex")
    S = np.array([np.stack((S, S_d))])
    
    F = np.array([[np.loadtxt("F.txt", dtype="complex")]])
    '''
    
    D = np.random.normal(size=(10, 2, 20, 20)) + np.random.normal(size=(10, 2, 20, 20))*1.j
    S = np.random.normal(size=(10, 2, 20, 20)) + np.random.normal(size=(10, 2, 20, 20))*1.j
    F = np.random.normal(size=(10, 1, 20, 20)) + np.random.normal(size=(10, 1, 20, 20))*1.j
    xs = np.linspace(-1., 1., D.shape[2])
    coords = np.dstack(np.meshgrid(xs, xs)[::-1])
    tt = tip_tilt(D, S, F, coords)
    #tt.calc()
    tt.optimize()

if __name__== "__main__":
    main()