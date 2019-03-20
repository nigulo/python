import numpy as np
import scipy.misc as misc

class zernike():
    
    def __init__(self, n, m):
        assert(n >= 0)
        assert(n >= abs(m))
        self.m = m
        self.abs_m = abs(m)
        self.n = n

    def get_value(self, rhos_phis):
        scalar = False
        if len(np.shape(rhos_phis)) == 1:
            scalar = True
            rhos_phis = np.array([rhos_phis])
        orig_shape = rhos_phis.shape
        if len(orig_shape) > 2:
            rhos_phis = rhos_phis.reshape(-1, 2)

        rhos = rhos_phis[:,0]
        phis = rhos_phis[:,1]
        #assert(all(rhos >= 0) and all(rhos <= 1))
        if (self.n - self.abs_m) % 2 != 0:
            return 0.0
        R = np.zeros(np.shape(rhos)[0])
        nmm2 = (self.n-self.abs_m)/2
        npm2 = (self.n+self.abs_m)/2
        k_fac = 1.0
        nmk_fac = misc.factorial(self.n)
        npm2_fac = misc.factorial(npm2)
        nmm2_fac = misc.factorial(nmm2)
        for k in np.arange(0, nmm2+1):
            R1 = np.power(rhos, self.n-2*k) * nmk_fac / k_fac / npm2_fac / nmm2_fac
            if k % 2 != 0:
                R1 = -R1
            R += R1
            k_fac *= k + 1
            if self.n - k  > 0:
                nmk_fac /= self.n - k
            if npm2 - k > 0:
                npm2_fac /=  npm2 - k
            if nmm2 - k > 0:
                nmm2_fac /=  nmm2 - k
        if self.m < 0:
            Z = R*np.sin(phis*self.abs_m)
        else:
            Z = R*np.cos(phis*self.abs_m)
        if len(orig_shape) > 2:
            Z = Z.reshape(orig_shape[:-1])
        if scalar:
            Z = Z[0]
        return Z


def get_noll(n, m):
    j = n*(n+1)/2 + abs(m)
    nmod4 = n % 4
    if (m >= 0 and nmod4 >= 2) or (m <= 0 and nmod4 <= 1):
        j += 1
    return j

def get_nm(noll_index):
    assert(noll_index  > 0)
    for n in np.arange(0, noll_index):
        for m in np.arange(-n, n+1, step = 2):
            j = get_noll(n, m)
            if j == noll_index:
                return (n, m)
    raise Exception("Should never happen")
    
