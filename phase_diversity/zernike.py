import numpy as np
import scipy.misc as misc

class zernike():
    
    def __init__(self, m, n):
        assert(n >= 0)
        assert(n >= abs(m))
        self.m = m
        self.abs_m = abs(m)
        self.n = n

    def get_value(self, rho, phi):
        assert(rho >= 0 and rho <= 1)
        if (self.n - self.abs_m) % 2 != 0:
            return 0.0
        R = 0.0
        nmm2 = (self.n-self.abs_m)/2
        npm2 = (self.n+self.abs_m)/2
        k_fac = 1.0
        nmk_fac = misc.factorial(self.n)
        npm2_fac = misc.factorial(npm2)
        nmm2_fac = misc.factorial(nmm2)
        for k in np.arange(0, nmm2+1):
            R1 = np.power(rho, self.n-2*k) * nmk_fac / k_fac / npm2_fac / nmm2_fac
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
            Z = R*np.sin(phi*self.abs_m)
        else:
            Z = R*np.cos(phi*self.abs_m)
        return Z

def get_mn(index):
    assert(index  > 0)
    i = 1
    for n in np.arange(0, index):
        if  i + n + 1 > index:
            break
        i += n + 1
    m = -n + 2*(index - i)
    return m, n