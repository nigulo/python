import numpy as np
import scipy.optimize as optimize

class tip_tilt:
    
    '''
        D and S are tensors with index l, k and x1, x2, where
        l is number od frames, k, number of diversities and
        x1 and x2 are width and height of the sensor in pixels
    '''
    def __init__(self, D, S, F, x, initial_a = None):
        #self.D = D
        #self.S = S
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
        self.x = x
        self.L = D.shape[0]
        self.initial_a = initial_a
        
        if self.initial_a is not None:
            self.initial_a = self.initial_a.reshape(self.L*2)
        #self.initial_a = np.zeros(self.L, 2)
    
    '''
    def fun(self, a_in):
        a = a_in.reshape((self.L, 2))
        au = np.tensordot(a, self.x, axes=(1, 2))
        f = self.CD1 - np.sum(self.C2*au, axis=0)
        f[self.zero_inds] = 0.
        f[self.non_zero_inds] *= self.C1
        val = self.CD2 - self.C2*f - au
        val1 = np.tensordot(val, self.x[:,:,0], axes=([1,2], [0,1]))
        val2 = np.tensordot(val, self.x[:,:,1], axes=([1,2], [0,1]))
        #retval = np.concatenate((val1**2, val2**2))
        retval = np.concatenate((val1, val2))
        #print(retval)
        return retval

    def fun2(self, a_in):
        a = a_in[:self.L*2].reshape((self.L, 2))
        fu = a_in[self.L*2:].reshape((self.x.shape[0], self.x.shape[1]))
        au = np.tensordot(a, self.x, axes=(1, 2))
        val = np.sin(self.D_T - fu - au)
        val *= self.C_T
        #print(val.shape)
        val1 = np.sum(np.tensordot(val, self.x[:,:,0], axes=([2,3], [0,1])), axis=0)
        val2 = np.sum(np.tensordot(val, self.x[:,:,1], axes=([2,3], [0,1])), axis=0)

        val3 = np.sum(val, axis=(0,1))
        retval = np.concatenate((val1, val2, np.reshape(val3, -1)))
        
        print(retval)
        return retval
    '''

    def fun3(self, a_in, a_old_in=None):
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
        val = np.abs(self.fun3(a_opt, a_opt))
        print("VALLLLLLLLLLL", val)
        for a in np.linspace(a0, a1, precision):
            val = np.abs(self.fun3(a, a_opt))
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
    
    def get_f(self, a):
        eps = 1e-10

        au = np.tensordot(a, self.x, axes=(1, 2))
        phi = self.D_T - au
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        tan_f = (np.sum(self.C_T*sin_phi, axis=(0, 1)) + eps)/((np.sum(self.C_T*cos_phi, axis=(0, 1))) + eps)
        return np.arctan(tan_f)        
                
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
    tt.calc()

if __name__== "__main__":
    main()