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
        
    def create_basis(self, do_fft=True, do_defocus=True):
        print("do_defocus", do_defocus)
        print("do_fft", do_fft)
        jmax = self.jmax
        arcsec_per_px = self.arcsec_per_px
        diameter = self.diameter
        wavelength = self.wavelength
        nx = self.nx
        F_D = self.F_D
        
        self.Xs = np.zeros((jmax, jmax, nx, nx))
        self.Ys = np.zeros((jmax, jmax, nx, nx))

        if do_fft:
            self.FXs = np.zeros((jmax, jmax, nx, nx), dtype='complex')
            self.FYs = np.zeros((jmax, jmax, nx, nx), dtype='complex')

        if do_defocus:
            self.Xs_d = np.zeros((jmax, jmax, nx, nx))
            self.Ys_d = np.zeros((jmax, jmax, nx, nx))

            if do_fft:
                self.FXs_d = np.zeros((jmax, jmax, nx, nx), dtype='complex')
                self.FYs_d = np.zeros((jmax, jmax, nx, nx), dtype='complex')


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

    def multiply(self, dat_F, betas, defocus = True):
        ret_val = np.zeros((self.nx, self.nx), dtype='complex')
        if defocus:
            ret_val_d = np.zeros((self.nx, self.nx), dtype='complex')
        for j in np.arange(0, self.jmax):
            for k in np.arange(0, j + 1):
                if defocus:
                    defocus_array = [False, True]
                else:
                    defocus_array = [False]
                for defocus1 in defocus_array:
                    FX, FY = self.get_FXFY(j, k, defocus = defocus1)
                    
                    FX1 = FX * (betas[j]*betas[k].conjugate()).real
                    FY1 = FY * (betas[j]*betas[k].conjugate()).imag
                    
                    if k == j:
                        FX1 *= 0.5
                        FY1 *= 0.5
                    
                    if defocus1:
                        ret_val_d += dat_F*(FX1 + FY1)
                    else:
                        ret_val += dat_F*(FX1 + FY1)
        if defocus:
            return (ret_val, ret_val_d)
        else:
            return ret_val
        
    def convolve(self, dat, betas, defocus = True):
        dat_F = fft.fft2(dat)
        ret_val = []
        for m_F in self.multiply(dat_F, betas, defocus):
            m = fft.fftshift(fft.ifft2(m_F).real)
            ret_val.append(m) 
        if defocus:
            return (ret_val[0], ret_val[1])
        else:
            return ret_val[0]

    def deconvolve(self, D, D_d, betas, gamma, do_fft = True):
        P, P_d = self.get_FP(betas)
        #P = np.roll(np.roll(P, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
        #P_d = np.roll(np.roll(P_d, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
        
        P_conj = P.conjugate()
        P_d_conj = P_d.conjugate()
        
        F_image = D * P_conj + gamma * D_d * P_d_conj
        F_image /= P*P_conj + gamma * P_d * P_d_conj
        
        if not do_fft:
            return F_image

        image = fft.ifft2(F_image).real
        #image = np.roll(np.roll(image, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
        return image

    def get_XY(self, j, k, defocus = False):
        if defocus:
            return self.Xs_d[j, k], self.Ys_d[j, k]
        else:
            return self.Xs[j, k], self.Ys[j, k]

    def get_FXFY(self, j, k, defocus = False):
        if defocus:
            return self.FXs_d[j, k], self.FYs_d[j, k]
        else:
            return self.FXs[j, k], self.FYs[j, k]
            
    def get_FP(self, betas, defocus = True):
        ret_val = np.zeros((self.nx, self.nx), dtype='complex')
        if defocus:
            ret_val_d = np.zeros((self.nx, self.nx), dtype='complex')
        for j in np.arange(0, self.jmax):
            for k in np.arange(0, j + 1):
                if defocus:
                    defocus_array = [False, True]
                else:
                    defocus_array = [False]
                for defocus1 in defocus_array:
                    FX, FY = self.get_FXFY(j, k, defocus = defocus1)
                    
                    FX1 = FX*(betas[j]*betas[k].conjugate()).real
                    FY1 = FY*(betas[j]*betas[k].conjugate()).imag
                    
                    if k == j:
                        FX1 *= 0.5
                        FY1 *= 0.5
                        

                    if defocus1:
                        ret_val_d += (FX1 + FY1)
                    else:
                        ret_val += (FX1 + FY1)
        if defocus:
            return (ret_val, ret_val_d)
        else:
            return ret_val


    '''
        Actually this is negative log likelihood
    '''
    def likelihood(self, theta, data):
        regularizer_eps = 1e-10

        D = data[0]
        D_d = data[1]
        gamma = data[2]
        betas_real = theta[:self.jmax]
        betas_imag = theta[self.jmax:]
        betas = betas_real + betas_imag*1.j
        
        P = np.zeros_like(D, dtype = 'complex')
        P_d = np.zeros_like(D_d, dtype = 'complex')

        for j in np.arange(0, self.jmax):
            for k in np.arange(0, j + 1):
                FX, FY = self.get_FXFY(j, k)
                FX_d, FY_d = self.get_FXFY(j, k, defocus=True)

                coef = betas[j]*betas[k].conjugate()
                if j == k:
                    coef *= 0.5
                coef_x = coef.real
                coef_y = coef.imag

                P += FX*coef_x + FY*coef_y
                P_d += FX_d*coef_x + FY_d*coef_y
        num = D_d*P - D*P_d
        num *= num.conjugate()
        den = P*P.conjugate()+gamma*P_d*P_d.conjugate() + regularizer_eps
        #if num == 0 and den == 0:
        #    L = np.ones((self.nx, self.nx))
        #else:
        L = num/den

        retval = np.sum(L.real)
        #print("likelihood", theta, retval)

        return retval
        

    def likelihood_grad(self, theta, data):
        regularizer_eps = 1e-10

        D = data[0]
        D_d = data[1]
        gamma = data[2]
        betas_real = theta[:self.jmax]
        betas_imag = theta[self.jmax:]
        betas = betas_real + betas_imag*1.j

        grads = np.zeros(len(theta))#, np.shape(D)[0], np.shape(D)[1]), dtype='complex')

        P, P_d = self.get_FP(betas, defocus = True)
        Q = 1./(P*P.conjugate()+gamma*P_d*P_d.conjugate() + regularizer_eps)

        for j1 in np.arange(0, len(betas)):

            dP_dbeta_real = np.zeros((np.shape(D)[0], np.shape(D)[1]), dtype = 'complex')
            dP_dbeta_imag = np.zeros((np.shape(D)[0], np.shape(D)[1]), dtype = 'complex')
            dP_d_dbeta_real = np.zeros((np.shape(D)[0], np.shape(D)[1]), dtype = 'complex')
            dP_d_dbeta_imag = np.zeros((np.shape(D)[0], np.shape(D)[1]), dtype = 'complex')
            for k1 in np.arange(0, self.jmax):#self.jmax):
                eps = 1.0
                if j1 == k1:
                    eps = 0.5
                if k1 > j1:
                    FX, FY = self.get_FXFY(k1, j1)
                    FX_d, FY_d = self.get_FXFY(k1, j1, defocus=True)
                    eps *= -1.
                else:
                    FX, FY = self.get_FXFY(j1, k1)
                    FX_d, FY_d = self.get_FXFY(j1, k1, defocus=True)

                dP_dbeta_real += betas[k1].real*FX - eps*betas[k1].imag*FY
                dP_dbeta_imag += eps*betas[k1].real*FY + betas[k1].imag*FX

                dP_d_dbeta_real += betas[k1].real*FX_d - eps*betas[k1].imag*FY_d
                dP_d_dbeta_imag += eps*betas[k1].real*FY_d + betas[k1].imag*FX_d
                
            num = D_d*P - D*P_d
            num_conj = num.conjugate()
            num_sq = num*num_conj

            real_part = (Q * num_conj*(D_d*dP_dbeta_real - D*dP_d_dbeta_real) -
                Q**2*num_sq*(P.conjugate()*dP_dbeta_real + gamma*P_d.conjugate()*dP_d_dbeta_real).real).real
                         
            imag_part = (Q * num_conj*(D_d*dP_dbeta_imag - D*dP_d_dbeta_imag) -
                Q**2*num_sq*(P.conjugate()*dP_dbeta_imag + gamma*P_d.conjugate()*dP_d_dbeta_imag).real).real

            grads[j1] = 2.*np.sum(real_part)
            grads[j1 + self.jmax] = 2.*np.sum(imag_part)

        #print("likelihood_grad", theta, grads)
        return grads


