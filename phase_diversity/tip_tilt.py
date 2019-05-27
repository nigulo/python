import numpy as np
import scipy.optimize as optimize

class tip_tilt:
    
    '''
        D and S are tensors with index l, k and x1, x2, where
        l is number od frames, k, number of diversities and
        x1 and x2 are width and height of the sensor in pixels
    '''
    def __init__(self, D, S, F, x):
        #self.D = D
        #self.S = S
        self.C = np.absolute(S)*np.absolute(D)*np.absolute(F)
        self.D = np.angle(D)-np.angle(S)-np.angle(F)
        self.C1 = np.sum(self.C, axis = (0, 1))
        self.non_zero_inds = np.where(self.C1 != 0.)
        self.zero_inds = np.where(self.C1 == 0.)
        self.C1 = 1./self.C1[self.non_zero_inds]
        self.C2 = np.sum(self.C, axis = 1)
        self.CD1 = np.sum(self.C*self.D, axis = (0, 1))
        self.CD2 = np.sum(self.C*self.D, axis = 1)
        self.x = x
        self.L = D.shape[0]
        #self.a = np.zeros(self.L, 2)
        
    def fun(self, a_in):
        a = a_in.reshape((self.L, 2))
        au = np.tensordot(a, self.x, axes=(1, 2))
        f = self.CD1 - np.sum(self.C2*au, axis=0)
        f[self.zero_inds] = 0.
        f[self.non_zero_inds] *= self.C1
        val = self.CD2 - self.C2*f - au
        val1 = np.tensordot(val, self.x[:,:,0], axes=([1,2], [0,1]))
        val2 = np.tensordot(val, self.x[:,:,1], axes=([1,2], [0,1]))
        retval = np.concatenate((val1**2, val2**2))
        print(retval)
        return retval
                
    def calc(self):
        #for l in np.arange(0, self.D.shape[0]):
        res = optimize.root(self.fun, x0=np.zeros((self.L, 2)), args=(), method='lm', jac=None, tol=None, callback=None, options=None)
        print(res)
        

D = np.loadtxt("D.txt", dtype="complex")
D_d = np.loadtxt("D_d.txt", dtype="complex")
D = np.array([np.stack((D, D_d))])

S = np.loadtxt("P.txt", dtype="complex")
S_d = np.loadtxt("P_d.txt", dtype="complex")
S = np.array([np.stack((S, S_d))])

F = np.array([[np.loadtxt("F.txt", dtype="complex")]])

#D = np.random.normal(size=(10, 2, 3, 3)) + np.random.normal(size=(10, 2, 3, 3))*1.j
#S = np.random.normal(size=(10, 2, 3, 3)) + np.random.normal(size=(10, 2, 3, 3))*1.j
#F = np.random.normal(size=(10, 1, 3, 3)) + np.random.normal(size=(10, 1, 3, 3))*1.j
xs = np.linspace(-1., 1., D.shape[2])
coords = np.dstack(np.meshgrid(xs, xs)[::-1])
tt = tip_tilt(D, S, F, coords)
tt.calc()
