import numpy as np
import scipy.special as special
import numpy.fft as fft
import zernike

'''
 Return a binomial coefficient
'''
def binomial_coef(n, k):

    if k > n:
        return 0.0
    kk = k
        
    if k > n/2:
        kk = n-k
            
    binomial_coef = 1.0
    for i in np.arange(1, kk + 1):
        binomial_coef *= (n-kk+i) / i

    return binomial_coef

'''
 Return if a number is odd
'''
def odd(m):
    return (m % 2) != 0


'''
def bessj0(x):

    p1=1.0
    p2=-.1098628627e-2
    p3=.2734510407e-4
    p4=-.2073370639e-5
    p5=.2093887211e-6
    q1=-.1562499995e-1
    q2=.1430488765e-3,
    q3=-.6911147651e-5,
    q4=.7621095161e-6,
    q5=-.934945152e-7

    r1=57568490574.0
    r2=-13362590354.0
    r3=651619640.7
    r4=-11214424.18
    r5=77392.33017
    r6=-184.9052456

    s1=57568490411.0
    s2=1029532985.0
    s3=9494680.718
    s4=59272.64853
    s5=267.8532712
    s6=1.0
    if(abs(x)<8.):
        y=x**2
        bessj0=(r1+y*(r2+y*(r3+y*(r4+y*(r5+y*r6)))))/(s1+y*(s2+y*(s3+y*(s4+y*(s5+y*s6)))))
    else:
        ax=abs(x)
        z=8./ax
        y=z**2
        xx=ax-.785398164
        bessj0=np.sqrt(.636619772/ax)*(np.cos(xx)*(p1+y*(p2+y*(p3+y*(p4+y*p5))))-z*np.sin(xx)*(q1+y*(q2+y*(q3+y*(q4+y*q5)))))
    return bessj0

def bessj1(x):

    r1=72362614232.0
    r2=-7895059235.0
    r3=242396853.1
    r4=-2972611.439
    r5=15704.4826
    r6=-30.16036606

    s1=144725228442.0
    s2=2300535178.0
    s3=18583304.74
    s4=99447.43394
    s5=376.9991397
    s6=1.0


    p1=1.0
    p2=.183105e-2
    p3=-.3516396496e-4
    p4=.2457520174e-5
    p5=-.240337019e-6

    q1=.04687499995
    q2=-.2002690873e-3,
    q3=.8449199096e-5
    q4=-.88228987e-6
    q5=.105787412e-6
    
    if(abs(x)<8.):
        y=x**2
        bessj1=x*(r1+y*(r2+y*(r3+y*(r4+y*(r5+y*r6)))))/(s1+y*(s2+y*(s3+y*(s4+y*(s5+y*s6)))))
    else:
        ax=abs(x)
        z=8./ax
        y=z**2
        xx=ax-2.356194491
        bessj1=np.sqrt(.636619772/ax)*(np.cos(xx)*(p1+y*(p2+y*(p3+y*(p4+y*p5))))-z*np.sin(xx)*(q1+y*(q2+y*(q3+y*(q4+y*q5)))))*np.sign(x)

    return bessj1


def bessj(n,x):
      
    iacc = 40
    bigno = 1.e10
    bigni = 1.e-10
      

    if n == 0:
        return bessj0(x)

    if n == 1:
        return bessj1(x)


    ax=abs(x)
    if(ax == 0.):
        bessj=0.
    elif ax > float(n):
        tox=2./ax
        bjm=bessj0(ax)
        bj=bessj1(ax)
        for j in np.arange(1, n):#1,n-1
            bjp=j*tox*bj-bjm
            bjm=bj
            bj=bjp

        bessj=bj
    else:
        tox=2./ax
        m=int(2*((n+int(np.sqrt(float(iacc*n))))/2))
        bessj=0.
        jsum=0
        sum_=0.
        bjp=0.
        bj=1.
        for j in np.arange(m, 0, -1):#m,1,-1
            bjm=j*tox*bj-bjp
            bjp=bj
            bj=bjm
            if abs(bj) > bigno:
                bj=bj*bigni
                bjp=bjp*bigni
                bessj=bessj*bigni
                sum_=sum_*bigni

            if jsum != 0:
                sum_=sum_+bj
            jsum=1-jsum
            if j==n:
                bessj=bjp

        sum_=2.*sum_-bj
        bessj=bessj/sum_
    if x<0 and (n % 2) == 1:
        bessj = -bessj
    return bessj
'''   

'''   
 Return the Vnm(r,f) function of the theory
'''
def Vnmf(radio, f, n, m):
    #real(kind=8) :: radio(:,:), f
    #integer :: n, m, l, j, lmax, p, q, ix, iy
    #real(kind=8) :: t1, t2, t3, t4, ulj, sum(size(radio(:,1)),size(radio(1,:))), epsm
    #complex(kind=8) :: ii, Vnm(size(radio(:,1)),size(radio(1,:)))

    p = int(0.5*(n-abs(m)))
    q = int(0.5*(n+abs(m)))
    
    epsm = 1.0
    if m < 0.0 and odd(m):
        epsm = -1.0
    
    lmax = 3.5*abs(f)+1
    #if(lmax != int(lmax)):
    #    print(lmax)
    #assert(lmax == int(lmax))
    lmax = int(lmax)
    
    ii = 1.j
    
    Vnm = np.zeros_like(radio, dtype='complex')
    
    for l in np.arange(1, lmax + 1):

        sum_ = np.zeros_like(radio)
        for j in np.arange(0, p + 1):
            if p-j < l:
                t1 = binomial_coef(abs(m)+j+l-1,l-1)
                t2 = binomial_coef(j+l-1,l-1)
                t3 = binomial_coef(l-1,p-j)
                t4 = binomial_coef(q+l+j,l)
                ulj = (-1.0)**p * (abs(m)+l+2*j) * t1 * t2 * t3 / t4
                for ix in np.arange(0, np.shape(radio)[0]):#size(radio(:,1)))
                    for iy in np.arange(0, np.shape(radio)[1]):#size(radio(1,:))
                        #x = 2.0*np.pi*(radio[ix,iy]+1.e-10)
                        #b1 = bessj(abs(m)+l+2*j, 2.0*np.pi*(radio[ix,iy]+1.e-10)) / (l*(2.0*np.pi*(radio[ix,iy]+1.e-10))**l)
                        #b2 = special.jv(abs(m)+l+2*j, 2.0*np.pi*(radio[ix,iy]+1.e-10)) / (l*(2.0*np.pi*(radio[ix,iy]+1.e-10))**l)
                        #b3 = special.iv(abs(m)+l+2*j, 2.0*np.pi*(radio[ix,iy]+1.e-10)) / (l*(2.0*np.pi*(radio[ix,iy]+1.e-10))**l)
                        
                        #print(b1, b2, b3, abs(m)+l+2*j, x, m, l, j)
                        #if (3.0 and x != 1.1279621214707376
                        #np.testing.assert_approx_equal(b1, b2, 7)

                        #np.testing.assert_approx_equal(bessj0(x), special.j0(x), 7)
                        #np.testing.assert_approx_equal(bessj1(x), special.j1(x), 7)


                        #sum_[ix,iy] += ulj * bessj(abs(m)+l+2*j, 2.0*np.pi*(radio[ix,iy]+1.e-10)) / (l*(2.0*np.pi*(radio[ix,iy]+1.e-10))**l)
                        sum_[ix,iy] += ulj * special.jv(abs(m)+l+2*j, 2.0*np.pi*(radio[ix,iy]+1.e-10)) / (l*(2.0*np.pi*(radio[ix,iy]+1.e-10))**l)
                        #sum_[ix,iy] += ulj * special.jv(abs(m)+l+2*j, 2.0*np.pi*(radio[ix,iy])) / (l*(2.0*np.pi*(radio[ix,iy]))**l)
        Vnm += sum_ * (-2.0*ii*f)**(l-1)
        
    Vnm *= epsm * np.exp(ii*f)
    return Vnm


class psf_basis:
    
    def __init__(self, jmax = 10, arcsec_per_px = 0.055, diameter = 20.0, wavelength = 5250.0, nx = 100, F_D = 1.0):
        
        self.jmax = jmax
        self.arcsec_per_px = arcsec_per_px
        self.diameter = diameter
        self.wavelength = wavelength
        self.nx = nx
        self.F_D = F_D
        self.betas = None

    def create_basis(self, do_fft=False, do_defocus=False):
        print("do_defocus", do_defocus)
        print("do_fft", do_fft)
        jmax = self.jmax
        arcsec_per_px = self.arcsec_per_px
        diameter = self.diameter
        wavelength = self.wavelength
        nx = self.nx
        F_D = self.F_D
        
        if self.betas is not None:
            dtype = 'complex'
        else:
            dtype = 'float'
        
        self.Xs = np.zeros((jmax, jmax, nx, nx), dtype=dtype)
        self.Ys = np.zeros((jmax, jmax, nx, nx), dtype=dtype)

        if do_fft:
            self.FXs = np.zeros((jmax, jmax, nx, nx), dtype=dtype)
            self.FYs = np.zeros((jmax, jmax, nx, nx), dtype=dtype)

        if do_defocus:
            self.Xs_d = np.zeros((jmax, jmax, nx, nx), dtype=dtype)
            self.Ys_d = np.zeros((jmax, jmax, nx, nx), dtype=dtype)

            if do_fft:
                self.FXs_d = np.zeros((jmax, jmax, nx, nx), dtype=dtype)
                self.FYs_d = np.zeros((jmax, jmax, nx, nx), dtype=dtype)


        diffraction = 1.22 * wavelength * 1e-8 / diameter
        diffraction = 206265.0 * diffraction
        
        print('Diffraction limit ["]=', diffraction)
        
        # Generate pupil plane
        x_diff = np.zeros(nx)
        y_diff = np.zeros(nx)
        for i in np.arange(0, nx):
            x_diff[i] = arcsec_per_px*i
            y_diff[i] = arcsec_per_px*i
        
        x_diff = x_diff - x_diff[int(nx/2)]
        y_diff = y_diff - y_diff[int(nx/2)]
        
        x_diff = x_diff / diffraction
        y_diff = y_diff / diffraction
        
        radio = np.zeros((nx,nx))
        phi = np.zeros((nx,nx))
        
        for i in np.arange(0, nx):
            for j in np.arange(0, nx):
                radio[i,j] = np.sqrt(x_diff[i]**2 + y_diff[j]**2)
                phi[i,j] = np.arctan2(y_diff[j], x_diff[i])
        
        radio = radio * 3.8317 / (2.0 * np.pi)
        # Generate the two focus+defocused PSFs
        
        defocus_array = [False]
        if do_defocus:
            defocus_array.append(True)
            
        for defocus in defocus_array:
            f = 0.0
            defocus_mm = 0.0
            
            if defocus:
                d_lambda = 8.0 * F_D**2
                f = np.pi * d_lambda / (4 * (F_D)**2)
                defocus_mm = d_lambda * wavelength*1.e-7
        
                print('Defocus in mm = ', d_lambda * wavelength*1.e-7)
                print('Defocus f = ', f)
        
        
        
            # Generate all the basis functions
            for j in np.arange(1, jmax + 1):
                n, m = zernike.get_nm(j)
        
                V_n_m =  Vnmf(radio, f, n, m)
                
                for k in np.arange(1, j + 1):
                    n_p, m_p = zernike.get_nm(k)
        
                    print(n, m, n_p, m_p)
        
                    V_np_mp = Vnmf(radio, f, n_p, m_p)
        
                    ca = np.cos(0.5*(m+m_p)*np.pi)
                    sa = np.sin(0.5*(m+m_p)*np.pi)
                    if abs(ca) < 1.0e-14:
                        ca = 0.0
                    if abs(sa) < 1.0e-14:
                        sa = 0.0
        
                    c_sum = ca*np.cos((m-m_p)*phi) - sa*np.sin((m-m_p)*phi)
                    s_sum = sa*np.cos((m-m_p)*phi) + ca*np.sin((m-m_p)*phi)
        
                    xi = V_n_m.imag * V_np_mp.imag + V_n_m.real * V_np_mp.real
                    psi = V_n_m.real * V_np_mp.imag - V_n_m.imag * V_np_mp.real
                
                    X = 8.0 * (-1.0)**m_p * (c_sum * xi + s_sum * psi)
                    Y = 8.0 * (-1.0)**m_p * (c_sum * psi - s_sum * xi)
                    
                    if j == k:
                        eps = 0.5
                    else:
                        eps = 1.0
                    
                    if self.betas is not None:
                        X *= eps*(self.betas[j-1]*self.betas[k-1].conjugate()).real
                        Y *= eps*(self.betas[j-1]*self.betas[k-1].conjugate()).imag
                        
                    if defocus:
                        self.Xs_d[j-1, k-1] = X
                        self.Ys_d[j-1, k-1]  = Y
                    else:
                        self.Xs[j-1, k-1] = X
                        self.Ys[j-1, k-1]  = Y
                        
                    
                    ###################################################################
                    if do_fft:
                    # Do the FFT and save the results
                    
                        FX = fft.fft2(X)
                        #FX = np.roll(np.roll(FX, int(nx/2), axis=0), int(nx/2), axis=1)
                        FY = fft.fft2(Y)
                        #FY = np.roll(np.roll(FY, int(nx/2), axis=0), int(nx/2), axis=1)
                        if defocus:
                            self.FXs_d[j-1, k-1] = FX
                            self.FYs_d[j-1, k-1] = FY
                        else:
                            self.FXs[j-1, k-1] = FX
                            self.FYs[j-1, k-1] = FY

    def getXY(self, j, k, defocus = False):
        if defocus:
            return self.Xs_d[j, k], self.Ys_d[j, k]
        else:
            return self.Xs[j, k], self.Ys[j, k]

    def getFXFY(self, j, k, defocus = False):
        if defocus:
            return self.FXs_d[j, k], self.FYs_d[j, k]
        else:
            return self.FXs[j, k], self.FYs[j, k]
            
    def set_betas(self, betas):
        self.betas=betas


    def likelihood(self, theta, data):
        D = data[0]
        D_d = data[1]
        gamma = data[2]
        betas = theta
        
        L = 0.
        for j in np.arange(0, self.jmax):
            for k in np.arange(0, j):
                if j == k:
                    eps = 0.5
                else:
                    eps = 1.0
                FX, FY = self.getFXFY(j, k)
                FX_d, FY_d = self.getFXFY(j, k, defocus=True)

                coef = eps*(betas[j-1]*betas[k-1].conjugate())
                coef_x = coef.real
                coef_y = coef.imag
                    
                FX *= coef_x
                FY *= coef_y

                FX_d *= coef_x
                FY_d *= coef_y

                P = FX + FY
                P_d = FX_d + FY_d
                num = D_d*P - D*P_d
                L += num*num.conjugate()/(P*P.conjugate()+gamma*P_d*P_d.conjugate())

        return L
        

    def likelihood_grad(self, theta, data):
        D = data[0]
        D_d = data[1]
        gamma = data[2]
        betas = theta

        grads = np.zeros(len(theta))


        for j1 in np.arange(0, len(grads)):

            dP_dbeta = 0.
            dP_d_dbeta = 0.
            for k1 in np.arange(0, self.jmax):
                FX1, FY1 = self.getFXFY(j1, k1)
                FX1_d, FY1_d = self.getFXFY(j1, k1, defocus=True)

                dP_dbeta += betas[k1].real*FX1-betas[k1].imag*FY1 + 1.j*(betas[k1].real*FY1+betas[k1].imag*FX1)
                dP_d_dbeta += betas[k1].real*FX1_d-betas[k1].imag*FY1_d + 1.j*(betas[k1].real*FY1_d+betas[k1].imag*FX1_d)
            dP_dbeta *= 8.
            dP_d_dbeta *= 8.

            for j in np.arange(0, self.jmax):
                for k in np.arange(0, j):
                    if j == k:
                        eps = 0.5
                    else:
                        eps = 1.0
                    FX, FY = self.getFXFY(j, k)
                    FX_d, FY_d = self.getFXFY(j, k, defocus=True)
    
                    coef = eps*(betas[j-1]*betas[k-1].conjugate())
                    coef_x = coef.real
                    coef_y = coef.imag
                        
                    P = coef_x*FX + coef_y*FY
                    P_d = coef_x*FX_d + coef_y*FY_d
    
                    num = D_d*P - D*P_d


                    Q = 1./(P*P.conjugate()+gamma*P_d*P_d.conjugate())
                
                    real_part = (Q * num.conjugate()*(D_d*dP_dbeta.real - D*dP_d_dbeta.real) -
                        Q**2*num*num.conjugate()*(P.conjugate()*dP_dbeta.real+gamma*P_d.conjugate()*dP_d_dbeta.real)).real
                    imag_part = (Q * num.conjugate()*(D_d*dP_dbeta.imag - D*dP_d_dbeta.imag) -
                        Q**2*num*num.conjugate()*(P.conjugate()*dP_dbeta.imag+gamma*P_d.conjugate()*dP_d_dbeta.imag)).real
                    grads[j1] += real_part + 1.j*imag_part
            grads[j1] *= 2.
                    

        return grads


