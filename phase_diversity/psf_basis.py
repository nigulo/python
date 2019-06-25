import numpy as np
import scipy.special as special
import numpy.fft as fft
import zernike
import utils

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
def Vnmf(radius, f, n, m):
    epsilon = 1.e-10
    #real(kind=8) :: radius(:,:), f
    #integer :: n, m, l, j, lmax, p, q, ix, iy
    #real(kind=8) :: t1, t2, t3, t4, ulj, sum(size(radius(:,1)),size(radius(1,:))), epsm
    #complex(kind=8) :: ii, Vnm(size(radius(:,1)),size(radius(1,:)))

    abs_m = abs(m)

    p = int(0.5*(n-abs_m))
    q = int(0.5*(n+abs_m))
    
    epsm = 1.0
    if m < 0.0 and odd(m):
        epsm = -1.0
    
    lmax = 3.5*abs(f)+1
    #if(lmax != int(lmax)):
    #    print(lmax)
    #assert(lmax == int(lmax))
    lmax = int(lmax)
    #lmax = min(20, lmax)
    
    Vnm = np.zeros_like(radius, dtype='complex')
    
    for l in np.arange(1, lmax + 1):

        v = 2.0*np.pi*(radius+epsilon)
        if f > 0.:
            inds = np.where(np.log(f/v)*l < 690) #Overflow of exponentiation
            v = v[inds]

        #inv_l_v_pow_l = 1./(l*v**l)
        sum_ = np.zeros_like(v)
        for j in np.arange(0, p + 1):
            if p-j < l:
                t1 = special.binom(abs_m+j+l-1,l-1)
                t2 = special.binom(j+l-1,l-1)
                t3 = special.binom(l-1,p-j)
                t4 = special.binom(q+l+j,l)
                
                #t1a = binomial_coef(abs(m)+j+l-1,l-1)
                #t2a = binomial_coef(j+l-1,l-1)
                #t3a = binomial_coef(l-1,p-j)
                #t4a = binomial_coef(q+l+j,l)

                #np.testing.assert_almost_equal(t1a, t1)
                #np.testing.assert_almost_equal(t2a, t2)
                #np.testing.assert_almost_equal(t3a, t3)
                #np.testing.assert_almost_equal(t4a, t4)
                ulj = (-1.0)**p * (abs_m+l+2*j) * t1 * t2 * t3 / t4
                
                #sum_ += ulj * special.jv(abs_m+l+2.*j, v) * inv_l_v_pow_l
                sum_ += ulj * special.jv(abs_m+l+2.*j, v)
        #inv_l_v_pow_l = 1./(l*v**l)
        #sum_ *= inv_l_v_pow_l
        #Vnm += sum_ * (-2.j*f)**(l-1)
        if f > 0:
            Vnm[inds] += sum_ * (-2.j*f/v)**l/(-2.j*f*l)
        else:
            if l == 1:
                Vnm += sum_ / (l*v**l)
            
    
    Vnm *= epsm * np.exp(1.j*f)
    #if n == 0 and m == 0:
    #    if f == 0.:
    #        res = special.jv(1, 2*np.pi*radius+epsilon)/(2.*np.pi*radius+epsilon)
    #        np.testing.assert_almost_equal(Vnm, res, 8)
    #    else:
    #        res = np.zeros_like(Vnm)
    #        for l in np.arange(1, 250):
    #            res += (-2.j*f)**(l-1)*special.jv(l, 2*np.pi*radius)/(2.*np.pi*radius)**l
    #        res *= np.exp(1.j*f)
    #        np.testing.assert_almost_equal(Vnm, res)
                
                
    return Vnm

def deconvolve_(Ds, Ps, gamma, do_fft = True, ret_all=False, tip_tilt = None, a_est=None):
    D = Ds[:,0,:,:]
    D_d = Ds[:,1,:,:]

    P = Ps[:,0,:,:]
    P_d = Ps[:,1,:,:]

    P_conj = P.conjugate()
    P_d_conj = P_d.conjugate()

    F_image = D * P_conj + gamma * D_d * P_d_conj
    F_image /= P*P_conj + gamma * P_d * P_d_conj
    
    #np.savetxt("F.txt", F_image, fmt='%f')
    
    if not do_fft and not ret_all:
        return F_image

    #image = fft.ifft2(fft.ifftshift(F_image)).real
    image = fft.ifft2(F_image).real
    #image = np.roll(np.roll(image, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
    
    if tip_tilt is not None and a_est is not None:
        image, image_F, Ps = tip_tilt.deconvolve(F_image, Ps, a_est)
        
    if ret_all:
        return image, F_image, Ps
    else:
        return image


class psf_basis:
    '''
        diameter is in centimeters
        wavelength is in Angstroms
    '''
    def __init__(self, jmax, nx, arcsec_per_px, diameter, wavelength, defocus, tip_tilt=None):
        
        self.jmax = jmax
        self.arcsec_per_px = arcsec_per_px
        self.diameter = diameter
        self.wavelength = wavelength
        self.nx = nx
        self.defocus = defocus
        self.tip_tilt = tip_tilt
    
    def get_state(self):
        return [self.FXs, self.FYs, self.FXs_d, self.FYs_d]
    
    def set_state(self, state):
        self.FXs = state[0]
        self.FYs = state[1]
        self.FXs_d = state[2]
        self.FYs_d = state[3]
        
        arcsec_per_px = self.arcsec_per_px/3600*np.pi/180
        diameter = self.diameter
        wavelength = self.wavelength
        nx = self.nx
        
        # Diffraction limit or the resolution of the telescope (in arcesonds)
        #diffraction = 1.22 * wavelength * 1e-8 / diameter
        diffraction = wavelength * 1e-8 / diameter
        
        print('Diffraction limit, arcsec_per_px', diffraction, arcsec_per_px)
        
        # Generate pupil plane
        #x_diff = np.linspace(-2., 2., nx)*arcsec_per_px
        x_diff = np.linspace(-float(nx)/2, float(nx)/2, nx)*arcsec_per_px
        print(len(x_diff))
        
        # Angular separation of pixels measured from the center (in arceconds)
        #x_diff = x_diff - x_diff[int(nx/2)]
        
        # Angular separation of resolved pixels measured from the center (in arceconds)
        x_diff /= diffraction
        
        coords = np.dstack(np.meshgrid(x_diff, x_diff)[::-1])
        self.coords = coords
        #np.testing.assert_array_almost_equal(coords, utils.get_coords(self.nx, self.arcsec_per_px, self.diameter, self.wavelength))
        
    
    def create_basis(self, do_fft=True, do_defocus=True):
        print("do_defocus", do_defocus)
        print("do_fft", do_fft)
        jmax = self.jmax
        arcsec_per_px = self.arcsec_per_px/3600*np.pi/180
        diameter = self.diameter
        wavelength = self.wavelength
        nx = self.nx
        
        self.Xs = np.zeros((jmax+1, jmax+1, nx, nx))
        self.Ys = np.zeros((jmax+1, jmax+1, nx, nx))

        if do_fft:
            self.FXs = np.zeros((jmax+1, jmax+1, nx, nx), dtype='complex')
            self.FYs = np.zeros((jmax+1, jmax+1, nx, nx), dtype='complex')

        if do_defocus:
            self.Xs_d = np.zeros((jmax+1, jmax+1, nx, nx))
            self.Ys_d = np.zeros((jmax+1, jmax+1, nx, nx))

            if do_fft:
                self.FXs_d = np.zeros((jmax+1, jmax+1, nx, nx), dtype='complex')
                self.FYs_d = np.zeros((jmax+1, jmax+1, nx, nx), dtype='complex')

        # Diffraction limit or the resolution of the telescope (in arcesonds)
        #diffraction = 1.22 * wavelength * 1e-8 / diameter
        diffraction = wavelength * 1e-8 / diameter
        
        print('Diffraction limit, arcsec_per_px', diffraction, arcsec_per_px)
        
        # Generate pupil plane
        #x_diff = np.linspace(-2., 2., nx)*arcsec_per_px
        x_diff = np.linspace(-float(nx)/2, float(nx)/2, nx)*arcsec_per_px
        print(len(x_diff))
        
        # Angular separation of pixels measured from the center (in arceconds)
        #x_diff = x_diff - x_diff[int(nx/2)]
        
        # Angular separation of resolved pixels measured from the center (in arceconds)
        x_diff /= diffraction
        
        print("scale_factor", arcsec_per_px/diffraction)
        
        radius = np.zeros((nx,nx))
        phi = np.zeros((nx,nx))
        
        print("PSF_BASIS x_limit", x_diff[0], x_diff[-1])
        
        coords = np.dstack(np.meshgrid(x_diff, x_diff)[::-1])
        self.coords = coords
        #np.testing.assert_array_almost_equal(coords, utils.get_coords(self.nx, self.arcsec_per_px, self.diameter, self.wavelength))
        print("psf_basis_coords", np.min(coords, axis=(0,1)), np.max(coords, axis=(0,1)), np.shape(coords))
        radiuses_phis = utils.cart_to_polar(coords)
        
        radius = radiuses_phis[:,:,0]
        phi = radiuses_phis[:,:,1]
        
        # What is this factor?
        #radius = radius * 3.8317 / (2.0 * np.pi)
        # Generate the two focus+defocused PSFs
        
        defocus_array = [False]
        if do_defocus:
            defocus_array.append(True)
            
        for is_defocus in defocus_array:
            f = 0.0
            
            if is_defocus:
                #d_lambda = 8.0 * F_D**2
                #f = np.pi * d_lambda / (4 * (F_D)**2)
                #defocus_mm = d_lambda * wavelength*1.e-7
        
                #print('Defocus in mm = ', defocus_mm)
                
                f = self.defocus #scale it with r
                print('Defocus f = ', f)
        
        
            # Generate all the basis functions
            for j in np.arange(0, jmax + 1):
                if j == 0:
                    n, m = 0, 0
                else:
                    n, m = zernike.get_nm(j)
        
                V_n_m =  Vnmf(radius, f, n, m)
                
                for k in np.arange(0, j + 1):
                    if k == 0:
                        n_p, m_p = 0, 0
                    else:
                        n_p, m_p = zernike.get_nm(k)
        
                    print(n, m, n_p, m_p)
        
                    V_np_mp = Vnmf(radius, f, n_p, m_p)
        
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
                    
                    if is_defocus:
                        self.Xs_d[j, k] = X
                        self.Ys_d[j, k]  = Y
                    else:
                        self.Xs[j, k] = X
                        self.Ys[j, k]  = Y
                        
                    
                    ###################################################################
                    if do_fft:
                    # Do the FFT and save the results
                    
                        #FX = fft.fft2(X)
                        FX = fft.fft2(fft.fftshift(X))
                        #FY = fft.fft2(Y)
                        FY = fft.fft2(fft.fftshift(Y))
                        if is_defocus:
                            self.FXs_d[j, k] = FX
                            self.FYs_d[j, k] = FY
                        else:
                            self.FXs[j, k] = FX
                            self.FYs[j, k] = FY

    '''
    dat_F.shape = [l, 2, nx, nx]
    betas.shape = [l, jmax]
    '''
    def multiply(self, dat_F, betas):
        ret_val = np.zeros_like(dat_F)
        for l in np.arange(0, dat_F.shape[0]):
            for j in np.arange(0, self.jmax+1):
                if j == 0:
                    betas_j = 1.
                else:
                    betas_j = betas[l, j-1]
                for k in np.arange(0, j + 1):
                    if k == 0:
                        betas_k = 1.
                    else:
                        betas_k = betas[l, k-1]
                        
                    FX, FY = self.get_FXFY(j, k, defocus=False)
                    FX_d, FY_d = self.get_FXFY(j, k, defocus=True)

                    coef = betas_j*betas_k.conjugate()
                    if j == k:
                        coef *= 0.5
                    coef_x = coef.real
                    coef_y = coef.imag
    
                    ret_val[l, 0] += FX*coef_x + FY*coef_y # focus
                    ret_val[l, 1] += FX_d*coef_x + FY_d*coef_y # defocus
        ret_val *= dat_F
        return ret_val
    
    '''
    dat.shape = [l, 2, nx, nx]
    betas.shape = [l, jmax]
    '''
    def convolve(self, dat, betas):
        #dat_F = fft.fftshift(fft.fft2(dat))
        dat_F = fft.fft2(fft.fftshift(dat))
        #dat_F = fft.fft2(dat)
        m_F = self.multiply(dat_F, betas) # m_F.shape is [l, 2, nx, nx]
        m = fft.ifftshift(fft.ifft2(m_F))
        threshold = np.ones_like(m.imag)*1e-12
        np.testing.assert_array_less(abs(m.imag), threshold)
        return m.real

    def deconvolve(self, Ds, betas, gamma, do_fft = True, ret_all=False, a_est=None):
        Ps = self.get_FP(betas)
        return deconvolve_(Ds, Ps, gamma, do_fft = do_fft, ret_all=ret_all, tip_tilt = self.tip_tilt, a_est=a_est)
        #P = np.roll(np.roll(P, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
        #P_d = np.roll(np.roll(P_d, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
        #print(D)
        #print(D_d)
        #print(P)
        #print(P_d)
        #np.savetxt("D.txt", D, fmt='%f')
        #np.savetxt("D_d.txt", D_d, fmt='%f')
        #np.savetxt("P.txt", P, fmt='%f')
        #np.savetxt("P_d.txt", P_d, fmt='%f')



    def get_XY(self, j, k, defocus):
        if defocus:
            return self.Xs_d[j, k], self.Ys_d[j, k]
        else:
            return self.Xs[j, k], self.Ys[j, k]

    def get_FXFY(self, j, k, defocus):
        if defocus:
            return self.FXs_d[j, k], self.FYs_d[j, k]
        else:
            return self.FXs[j, k], self.FYs[j, k]
            
    def get_FP(self, betas):
        return self.multiply(np.ones((betas.shape[0], 2, self.nx, self.nx), dtype='complex'), betas)


    def encode_params(self, betas, a = None):
        theta = np.stack((betas.real, betas.imag), axis=1).flatten()
        if self.tip_tilt is not None:
            theta = np.concatenate((theta, self.tip_tilt.encode(a)))
        return theta

    def encode_data(self, Ds, gamma):
        return [Ds, gamma]
    
    def encode(self, betas, Ds, gamma, a = []):
        return self.encode_params(betas, a), self.encode_data(Ds, gamma)

    def decode(self, theta, data):
        Ds = data[0]
        gamma = data[1]
        L = Ds.shape[0]
        betas = np.zeros((L, self.jmax), dtype = 'complex')
        #print("theta.shape", theta.shape, L, self.jmax, theta)
        for l in np.arange(0, L):
            begin_index = l*2*self.jmax
            betas_real = theta[begin_index:begin_index+self.jmax]
            betas_imag = theta[begin_index+self.jmax:begin_index+2*self.jmax]
            betas[l] = betas_real + betas_imag*1.j
        return betas, Ds, gamma, theta[L*2*self.jmax:]

    '''
        Actually this is negative log likelihood
    '''
    def likelihood(self, theta, data):
        betas, Ds, gamma, other = self.decode(theta, data)
        regularizer_eps = 1e-10

        D = Ds[:,0,:,:]
        D_d = Ds[:,1,:,:]
        
        Ps = np.zeros_like(Ds, dtype = 'complex')
        #P = np.zeros_like(D, dtype = 'complex')
        #P_d = np.zeros_like(D_d, dtype = 'complex')

        L = Ds.shape[0]

        for l in np.arange(0, L):
            for j in np.arange(0, self.jmax + 1):
                for k in np.arange(0, j + 1):
                    FX, FY = self.get_FXFY(j, k, defocus=False)
                    FX_d, FY_d = self.get_FXFY(j, k, defocus=True)
    
                    if j == 0:
                        betas_j = 1.
                    else:
                        betas_j = betas[l, j-1]
                    if k == 0:
                        betas_k = 1.
                    else:
                        betas_k = betas[l, k-1]
                    coef = betas_j*betas_k.conjugate()
                    if j == k:
                        coef *= 0.5
                    coef_x = coef.real
                    coef_y = coef.imag
    
                    Ps[l, 0, :, :] += FX*coef_x + FY*coef_y # focus
                    Ps[l, 1, :, :] += FX_d*coef_x + FY_d*coef_y # defocus
        
        P = Ps[:, 0, :, :]
        P_d = Ps[:, 1, :, :]
        num = D_d*P - D*P_d
        num *= num.conjugate()
        den = P*P.conjugate()+gamma*P_d*P_d.conjugate()
        
        eps_indices = np.where(abs(den) < regularizer_eps)
        sign_indices = np.where(den[eps_indices] < 0.)
        den[eps_indices] = regularizer_eps
        den[eps_indices][sign_indices] *= -1.

        #den += regularizer_eps
        lik = num/den

        retval = np.sum(lik.real)
        #print("likelihood", theta, retval)
        
        #######################################################################
        # Tip-tilt estimation
        #######################################################################
        if self.tip_tilt is not None:
            self.tip_tilt.set_data(Ds, Ps)#, F)
            retval += self.tip_tilt.lik(other)

        return retval
        
    def likelihood_grad(self, theta, data):
        betas, Ds, gamma, other = self.decode(theta, data)
        L = Ds.shape[0]
        regularizer_eps = 1e-10
        
        D = Ds[:,0,:,:]
        D_d = Ds[:,1,:,:]
        
        grads = np.zeros(L*2*self.jmax)#, np.shape(D)[0], np.shape(D)[1]), dtype='complex')

        Ps = self.get_FP(betas)
        P = Ps[:, 0, :, :]
        P_d = Ps[:, 1, :, :]
        
        Q = P*P.conjugate()+gamma*P_d*P_d.conjugate()
        
        eps_indices = np.where(abs(Q) < regularizer_eps)
        sign_indices = np.where(Q[eps_indices] < 0.)
        Q[eps_indices] = regularizer_eps
        Q[eps_indices][sign_indices] *= -1.
        
        #Q += regularizer_eps
        Q = 1./Q

        for l in np.arange(0, L):
            for j1 in np.arange(1, self.jmax+1):
                dP_dbeta_real = np.zeros((self.nx, self.nx), dtype = 'complex')
                dP_dbeta_imag = np.zeros((self.nx, self.nx), dtype = 'complex')
                dP_d_dbeta_real = np.zeros((self.nx, self.nx), dtype = 'complex')
                dP_d_dbeta_imag = np.zeros((self.nx, self.nx), dtype = 'complex')
                for k1 in np.arange(0, self.jmax+1):
                    eps = 1.0
                    if j1 == k1:
                        eps = 0.5
                    if k1 > j1:
                        FX, FY = self.get_FXFY(k1, j1, defocus=False)
                        FX_d, FY_d = self.get_FXFY(k1, j1, defocus=True)
                        eps *= -1.
                    else:
                        FX, FY = self.get_FXFY(j1, k1, defocus=False)
                        FX_d, FY_d = self.get_FXFY(j1, k1, defocus=True)
                    if k1 == 0:
                        betas_k1 = 1.
                    else:
                        betas_k1 = betas[l, k1-1]
                    dP_dbeta_real += betas_k1.real*FX - eps*betas_k1.imag*FY
                    dP_dbeta_imag += eps*betas_k1.real*FY + betas_k1.imag*FX
    
                    dP_d_dbeta_real += betas_k1.real*FX_d - eps*betas_k1.imag*FY_d
                    dP_d_dbeta_imag += eps*betas_k1.real*FY_d + betas_k1.imag*FX_d
                    
                num = D_d*P - D*P_d
                num_conj = num.conjugate()
                num_sq = num*num_conj
    
                real_part = (Q * num_conj*(D_d*dP_dbeta_real - D*dP_d_dbeta_real) -
                    Q**2*num_sq*(P.conjugate()*dP_dbeta_real + gamma*P_d.conjugate()*dP_d_dbeta_real).real).real
                             
                imag_part = (Q * num_conj*(D_d*dP_dbeta_imag - D*dP_d_dbeta_imag) -
                    Q**2*num_sq*(P.conjugate()*dP_dbeta_imag + gamma*P_d.conjugate()*dP_d_dbeta_imag).real).real
    
                l_index = l*2*self.jmax
                grads[l_index + j1-1] = 2.*np.sum(real_part)
                grads[l_index + j1-1 + self.jmax] = 2.*np.sum(imag_part)

        #eps_indices = np.where(abs(grads) < regularizer_eps)
        #grads[eps_indices] = np.random.normal()*regularizer_eps
        #print("likelihood_grad", theta, grads)
        
        #######################################################################
        # Tip-tilt estimation
        #######################################################################
        if self.tip_tilt is not None:
            self.tip_tilt.set_data(Ds, Ps)#, F)
            grads = np.concatenate((grads, self.tip_tilt.lik_grad(other)))
        
        return grads

def maybe_invert(image_est, image):
    mse = np.sum((image_est-image)**2)
    mse_neg = np.sum((-image_est-image)**2)
    if mse_neg < mse:
        return -image_est
    else:
        return image_est
