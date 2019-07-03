import sys
sys.path.append('..')
import numpy as np
import psf_basis
import numpy.fft as fft
import unittest
sys.path.append('../../utils')
import plot
import misc
import zernike
import utils
import scipy.special as special

image = [[0.41960785, 0.38039216, 0.36862746, 0.38039216, 0.40784314, 0.40392157,
  0.38431373, 0.4509804,  0.45882353, 0.5137255 ],
 [0.4117647,  0.38039216, 0.39607844, 0.39215687, 0.34117648, 0.3529412,
  0.35686275, 0.37254903, 0.36862746, 0.38039216],
 [0.36862746, 0.34901962, 0.30980393, 0.3254902,  0.31764707, 0.3372549,
  0.3019608,  0.3254902,  0.33333334, 0.34901962],
 [0.3254902,  0.34509805, 0.3647059,  0.37254903, 0.41568628, 0.36078432,
  0.33333334, 0.32156864, 0.28235295, 0.30980393],
 [0.4392157,  0.4509804,  0.5019608,  0.4627451,  0.4745098,  0.43529412,
  0.36078432, 0.3254902,  0.2901961,  0.2627451 ],
 [0.54901963, 0.5058824,  0.56078434, 0.56078434, 0.5803922,  0.49803922,
  0.3882353,  0.34117648, 0.28627452, 0.30588236],
 [0.6039216,  0.61960787, 0.64705884, 0.61960787, 0.627451,   0.5568628,
  0.42745098, 0.3647059,  0.32941177, 0.32156864],
 [0.6666667,  0.7294118,  0.69803923, 0.7176471,  0.62352943, 0.5803922,
  0.45882353, 0.37254903, 0.3372549,  0.3372549 ],
 [0.6745098,  0.654902,   0.7019608,  0.6862745,  0.6431373,  0.5529412,
  0.42352942, 0.40392157, 0.37254903, 0.39215687],
 [0.6509804,  0.6901961,  0.6509804,  0.6392157,  0.58431375, 0.5294118,
  0.45490196, 0.39607844, 0.36862746, 0.37254903]]

D = [[ 2.69347342e+02+0.00000000e+00j, -2.89554084e+02-3.58791959e+01j,
   7.78166322e+00+2.19016473e+01j,  4.13088439e-01+1.05472906e+00j,
  -1.89394228e-01+2.79339914e-01j, -8.79842874e-02-1.07611518e-15j,
  -1.89394228e-01-2.79339914e-01j,  4.13088439e-01-1.05472906e+00j,
   7.78166322e+00-2.19016473e+01j, -2.89554084e+02+3.58791959e+01j],
 [ 2.20569539e+02-3.82197177e+01j, -1.48964906e+02+1.66741643e+02j,
   1.08886177e+01+1.22060601e-01j,  9.82446036e-01-1.51733843e-01j,
   3.49968053e-01+5.40157833e-02j,  1.51165360e-01+3.39417635e-01j,
   1.65372141e-01+3.62126579e-01j, -1.38871618e-01+1.96330637e-01j,
  -6.88177685e-01-1.91618306e+00j, -3.24473709e+01+4.34690238e+00j],
 [-6.73694394e+00+4.86597987e-01j,  6.16744458e+00-3.98384510e+00j,
   1.44632161e-01+1.19208047e-01j,  5.50547259e-02+3.18707980e-02j,
  -2.99519595e-02+7.62596249e-02j,  3.24758725e-02+6.11444406e-02j,
   3.56973988e-02-6.56128166e-02j, -4.98874045e-02-3.47566097e-02j,
  -3.02237881e-02-1.23435566e-01j,  4.04257162e-01+8.33998525e-01j],
 [-1.25136395e-01-2.17017423e-01j,  6.94810919e-01+1.36620924e-01j,
   1.94309712e-02+1.48505054e-01j, -3.76371622e-04+5.50484602e-03j,
  -2.36229089e-02+5.77454916e-03j, -1.77015648e-02+5.57688088e-03j,
  -2.54265704e-02-7.72744643e-03j, -3.22203605e-02-7.66114178e-03j,
  -3.37811918e-02+4.86935904e-02j,  1.75678264e-01-9.45620427e-02j],
 [-1.12912483e-01-9.81225464e-02j,  6.47330468e-01-1.87891423e-01j,
   2.14891325e-01+1.95371552e-01j,  1.01616806e-02+1.55925934e-03j,
   1.65165827e-02+1.14798844e-02j, -1.74619556e-02-4.75322954e-03j,
  -3.05675761e-02-8.28579109e-03j, -5.42700655e-03-1.01090272e-02j,
  -1.11165683e-01-1.18874428e-01j,  2.25328571e-01-1.28035175e-01j],
 [-5.45039411e-02+8.70453321e-16j,  4.45028507e-01-6.14295323e-02j,
  -8.70485302e-03+1.50061235e-01j, -1.94549622e-02+7.13177320e-03j,
   5.11292546e-03+1.59197675e-02j, -2.20514119e-03-1.38831297e-18j,
   5.11292546e-03-1.59197675e-02j, -1.94549622e-02-7.13177320e-03j,
  -8.70485302e-03-1.50061235e-01j,  4.45028507e-01+6.14295323e-02j],
 [-1.12912483e-01+9.81225464e-02j,  2.25328571e-01+1.28035175e-01j,
  -1.11165683e-01+1.18874428e-01j, -5.42700655e-03+1.01090272e-02j,
  -3.05675761e-02+8.28579109e-03j, -1.74619556e-02+4.75322954e-03j,
   1.65165827e-02-1.14798844e-02j,  1.01616806e-02-1.55925934e-03j,
   2.14891325e-01-1.95371552e-01j,  6.47330468e-01+1.87891423e-01j],
 [-1.25136395e-01+2.17017423e-01j,  1.75678264e-01+9.45620427e-02j,
  -3.37811918e-02-4.86935904e-02j, -3.22203605e-02+7.66114178e-03j,
  -2.54265704e-02+7.72744643e-03j, -1.77015648e-02-5.57688088e-03j,
  -2.36229089e-02-5.77454916e-03j, -3.76371622e-04-5.50484602e-03j,
   1.94309712e-02-1.48505054e-01j,  6.94810919e-01-1.36620924e-01j],
 [-6.73694394e+00-4.86597987e-01j,  4.04257162e-01-8.33998525e-01j,
  -3.02237881e-02+1.23435566e-01j, -4.98874045e-02+3.47566097e-02j,
   3.56973988e-02+6.56128166e-02j,  3.24758725e-02-6.11444406e-02j,
  -2.99519595e-02-7.62596249e-02j,  5.50547259e-02-3.18707980e-02j,
   1.44632161e-01-1.19208047e-01j,  6.16744458e+00+3.98384510e+00j],
 [ 2.20569539e+02+3.82197177e+01j, -3.24473709e+01-4.34690238e+00j,
  -6.88177685e-01+1.91618306e+00j, -1.38871618e-01-1.96330637e-01j,
   1.65372141e-01-3.62126579e-01j,  1.51165360e-01-3.39417635e-01j,
   3.49968053e-01-5.40157833e-02j,  9.82446036e-01+1.51733843e-01j,
   1.08886177e+01-1.22060601e-01j, -1.48964906e+02-1.66741643e+02j]]
D_d = [[ 5.41635689e+02+0.00000000e+00j,  3.68881920e+01-1.19828784e+01j,
  -8.64816943e-01+9.42576414e-01j, -6.51504955e-01+3.83253339e-01j,
  -5.82391244e-01+1.30527504e-02j, -3.39103812e-01+2.01535668e-16j,
  -5.82391244e-01-1.30527504e-02j, -6.51504955e-01-3.83253339e-01j,
  -8.64816943e-01-9.42576414e-01j,  3.68881920e+01+1.19828784e+01j],
 [-2.48032613e+01-1.19056748e+00j, -1.23030366e+00-2.76166909e+01j,
   1.57626952e+00-6.29222777e-01j,  3.51148463e-01-2.20709147e-01j,
   2.47066490e-01+4.02749354e-02j,  5.72245770e-02+3.28316064e-01j,
   3.44870772e-02+4.24151735e-01j, -2.39094529e-01+2.36572506e-01j,
  -3.13439404e-01+1.10982069e-01j, -2.07496554e+00-5.17236807e+00j],
 [ 4.29771322e-01-1.68117924e-01j,  6.05283278e-01-9.56106590e-01j,
  -1.04342481e-01-7.43989854e-02j,  9.71253275e-05-2.50628086e-02j,
   2.74439735e-02+1.12812870e-02j,  2.68168765e-02-6.01139665e-03j,
  -1.98936907e-02-2.72274040e-02j, -3.07655639e-02+1.12510275e-02j,
  -4.43348031e-02-6.39044814e-02j,  1.22610705e-01-3.34345891e-01j],
 [ 2.45062025e-02+1.64002469e-01j,  3.47128889e-01-9.60586510e-02j,
   4.81920513e-02-2.51848755e-02j,  5.08895195e-03+6.83627324e-03j,
  -2.79413077e-02+2.41980710e-02j, -2.29088805e-02+1.73621054e-02j,
  -4.18062764e-02-2.78279698e-03j, -5.22953409e-02-7.83359563e-03j,
  -3.38981350e-02+1.21449388e-02j,  2.12244825e-01-3.40816681e-01j],
 [-1.83730874e-02+1.09223189e-01j,  5.17828324e-01-1.78269393e-01j,
   8.41795118e-02-6.35746257e-02j,  1.49055064e-02-4.47683245e-03j,
   2.91527923e-02+1.08062146e-02j, -2.81956731e-02-3.71090267e-03j,
  -4.86324423e-02-1.19463514e-02j, -6.38500493e-03-1.68782268e-02j,
   1.12760821e-02-7.88018940e-02j,  2.23862951e-01-2.57480896e-01j],
 [-6.41254011e-02+1.10782857e-16j,  4.68265919e-01+2.20161661e-02j,
   5.42975043e-02+3.07582688e-02j, -2.57332203e-02+1.89737827e-02j,
   1.08008137e-02+2.39326841e-02j, -3.48344309e-03+3.13577405e-18j,
   1.08008137e-02-2.39326841e-02j, -2.57332203e-02-1.89737827e-02j,
   5.42975043e-02-3.07582688e-02j,  4.68265919e-01-2.20161661e-02j],
 [-1.83730874e-02-1.09223189e-01j,  2.23862951e-01+2.57480896e-01j,
   1.12760821e-02+7.88018940e-02j, -6.38500493e-03+1.68782268e-02j,
  -4.86324423e-02+1.19463514e-02j, -2.81956731e-02+3.71090267e-03j,
   2.91527923e-02-1.08062146e-02j,  1.49055064e-02+4.47683245e-03j,
   8.41795118e-02+6.35746257e-02j,  5.17828324e-01+1.78269393e-01j],
 [ 2.45062025e-02-1.64002469e-01j,  2.12244825e-01+3.40816681e-01j,
  -3.38981350e-02-1.21449388e-02j, -5.22953409e-02+7.83359563e-03j,
  -4.18062764e-02+2.78279698e-03j, -2.29088805e-02-1.73621054e-02j,
  -2.79413077e-02-2.41980710e-02j,  5.08895195e-03-6.83627324e-03j,
   4.81920513e-02+2.51848755e-02j,  3.47128889e-01+9.60586510e-02j],
 [ 4.29771322e-01+1.68117924e-01j,  1.22610705e-01+3.34345891e-01j,
  -4.43348031e-02+6.39044814e-02j, -3.07655639e-02-1.12510275e-02j,
  -1.98936907e-02+2.72274040e-02j,  2.68168765e-02+6.01139665e-03j,
   2.74439735e-02-1.12812870e-02j,  9.71253275e-05+2.50628086e-02j,
  -1.04342481e-01+7.43989854e-02j,  6.05283278e-01+9.56106590e-01j],
 [-2.48032613e+01+1.19056748e+00j, -2.07496554e+00+5.17236807e+00j,
  -3.13439404e-01-1.10982069e-01j, -2.39094529e-01-2.36572506e-01j,
   3.44870772e-02-4.24151735e-01j,  5.72245770e-02-3.28316064e-01j,
   2.47066490e-01-4.02749354e-02j,  3.51148463e-01+2.20709147e-01j,
   1.57626952e+00+6.29222777e-01j, -1.23030366e+00+2.76166909e+01j]]


# Comparison method without optomization for large f-s
def Vnmf1(radius, f, n, m):
    epsilon = 1.e-10

    abs_m = abs(m)

    p = int(0.5*(n-abs_m))
    q = int(0.5*(n+abs_m))
    
    epsm = 1.0
    if m < 0.0 and (m % 2) == 1:
        epsm = -1.0
    
    lmax = 3.5*abs(f)+1
    lmax = int(lmax)
    
    Vnm = np.zeros_like(radius, dtype='complex')
    
    for l in np.arange(1, lmax + 1):

        v = 2.0*np.pi*(radius+epsilon)
        inv_l_v_pow_l = 1./(l*v**l)
        sum_ = np.zeros_like(v)
        for j in np.arange(0, p + 1):
            if p-j < l:
                t1 = special.binom(abs_m+j+l-1,l-1)
                t2 = special.binom(j+l-1,l-1)
                t3 = special.binom(l-1,p-j)
                t4 = special.binom(q+l+j,l)
                ulj = (-1.0)**p * (abs_m+l+2*j) * t1 * t2 * t3 / t4
                
                sum_ += ulj * special.jv(abs_m+l+2.*j, v) * inv_l_v_pow_l
        Vnm += sum_ * (-2.j*f)**(l-1)
    Vnm *= epsm * np.exp(1.j*f)
    return Vnm

class test_psf_basis(unittest.TestCase):

    
    def test_Vnmf(self):

        nx = 20
        x_diff = np.linspace(-float(nx)/2, float(nx)/2, nx)*.00001
        
        radius = np.zeros((nx,nx))
        coords = np.dstack(np.meshgrid(x_diff, x_diff)[::-1])
        radiuses_phis = utils.cart_to_polar(coords)
        
        radius = radiuses_phis[:,:,0]

        for f in [1., 3., 5.]:
            for n in np.arange(0, 10):
                for m in np.arange(0, 10):
                    result = psf_basis.Vnmf(radius, f, n, m)
                    expected = Vnmf1(radius, f, n, m)
                    #for i in np.arange(0, nx):
                    #    for j in np.arange(0, nx):
                    #        if (result[i, j] - expected[i, j] != 0):
                    #            print(f, n, m, radius[i, j], result[i, j], expected[i, j], result[i, j] - expected[i, j])
                    #        np.testing.assert_almost_equal(result[i, j], expected[i, j])
                    np.testing.assert_array_almost_equal(result, expected)

    def test_convolve1(self):
        #First test if the same image is returned if betas are zero
        jmax = 10
        arcsec_per_px = 0.055
        diameter = 20.0
        wavelength = 5250.0
        nx = 10

        L = 5
        
        #######################################################################
        # Use flat field to test the normalization

        psf = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = 3.)
        psf.create_basis(do_fft=True, do_defocus=True)

        betas = np.random.normal(size=(L, jmax)) + np.random.normal(size=(L, jmax))*1.j
        #betas = np.ones((L, jmax), dtype='complex')*1.j
        ns = np.zeros(jmax)
        for j in np.arange(0, jmax):
            n, m = zernike.get_nm(j + 2)
            ns[j] = n + 1
        
        flat_field = np.ones_like(image)
        flat_field = np.tile(np.array([flat_field, flat_field]), (L, 1)).reshape((L, 2, nx, nx))
        Ds = psf.convolve(flat_field, betas)
        # No defocus, so D should be equal to D_d
        D = Ds[:, 0, :, :]
        D_d = Ds[:, 1, :, :]
        np.testing.assert_almost_equal(D, D_d, 8)
        print("sum_betas:", np.sum(betas*betas.conjugate()/(ns*np.pi)), Ds[0, 0, 0, 0])
        np.testing.assert_almost_equal(Ds, flat_field, 3)
        

    def test_convolve2(self):
        #First test if the same image is returned if betas are zero
        jmax = 10
        arcsec_per_px = 0.055
        diameter = 20.0
        wavelength = 5250.0
        nx = 10

        L = 5
        
        #######################################################################
        # Use the actual image

        psf = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = 0.)
        psf.create_basis(do_fft=True, do_defocus=True)

        betas = np.zeros((L, jmax), dtype='complex')


        Ds = np.tile(np.array([image, image]), (L, 1)).reshape((L, 2, nx, nx))
        Ds = psf.convolve(Ds, betas)
        #Df, Df_d = psf.multiply(fimage, betas)

        # No defocus, so D should be equal to D_d
        D = Ds[:, 0, :, :]
        D_d = Ds[:, 1, :, :]
        np.testing.assert_almost_equal(D, D_d, 8)
        
        #my_plot = plot.plot_map(nrows=1, ncols=2)
        #my_plot.plot(image, [0])
        #my_plot.plot(D, [1])
            
        #my_plot.save("test_deconvolve.png")
        #my_plot.close()


        ##TODO Normalization needs to be resolved
        ##There is still the Airy disk convolution, so one cannot
        ##expect equlity with the original image
        #threshold = np.ones_like(D)*0.18
        #np.testing.assert_array_less((misc.normalize(D) - misc.normalize(image))**2, threshold)
        

    def test_convolve3(self):
        #First test if the same image is returned if betas are zero
        jmax = 10
        arcsec_per_px = 0.055
        diameter = 20.0
        wavelength = 5250.0
        nx = 10

        L = 5
        
        #######################################################################
        # Now test actual convolution

        betas = np.random.normal(size=(L, jmax)) + 1.j*np.random.normal(size=(L, jmax))

        psf = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = 0.)
        psf.create_basis(do_fft=True, do_defocus=True)
        
        image_F = fft.fft2(fft.fftshift(image))
        Ds = np.tile(np.array([image_F, image_F]), (L, 1)).reshape((L, 2, nx, nx))
        Ds, _ = psf.multiply(Ds, betas)
        reconst = psf.deconvolve(Ds, betas=betas, gamma=1., do_fft=True, normalize = False)
        for l in np.arange(0, L):
            np.testing.assert_almost_equal(reconst[l], image, 15)
    
    def test_likelihood(self):
        
        jmax = 5
        arcsec_per_px = 0.055
        diameter = 20.0
        wavelength = 5250.0
        nx = 10
        defocus = 1.0

        gamma = 1.
        
        L = 1
        
        psf = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus)
        psf.create_basis(do_fft=True, do_defocus=True)


        betas = np.random.normal(size=(L, jmax)) + np.random.normal(size=(L, jmax))*1.j
        Ds = np.array([np.stack((D, D_d))])
        theta, data = psf.encode(betas, Ds, gamma)

        lik = psf.likelihood(theta, data)
        
        Ps = psf.get_FP(betas)
        P = Ps[:, 0, :, :]
        P_d = Ps[:, 1, :, :]

        num = D_d*P-D*P_d
        num *= num.conjugate()
        #lik_expected = np.sum(num/np.sqrt(P*P.conjugate() + gamma*P_d*P_d.conjugate())).real
        lik_expected = np.sum(num/(P*P.conjugate() + gamma*P_d*P_d.conjugate())).real
        np.testing.assert_almost_equal(lik, lik_expected, 6)

    def test_likelihood_grad(self):
        
        jmax = 5
        arcsec_per_px = 0.055
        diameter = 20.0
        wavelength = 5250.0
        nx = 10
        defocus = 1.0
        
        gamma = 1.
        L = 1
        
        psf = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus)
        psf.create_basis(do_fft=True, do_defocus=True)


        betas = np.random.normal(size=(L, jmax)) + np.random.normal(size=(L, jmax))*1.j
        Ds = np.array([np.stack((D, D_d))])
        theta, data = psf.encode(betas, Ds, gamma)

        delta_betas = 1.0e-7 + betas*1.0e-7

        lik = psf.likelihood(theta, data)
        liks = np.tile(lik, (betas.shape[0], betas.shape[1]))
        liks1_real = np.zeros_like(betas.real)
        liks1_imag = np.zeros_like(betas.imag)
        for l in np.arange(0, L):
            for i in np.arange(0, betas.shape[1]):
                delta = np.zeros_like(betas)
                delta[l, i] = delta_betas[l, i].real
                betas1 = betas+delta
                theta1, _ = psf.encode(betas1, Ds, gamma)
                
                liks1_real[l, i] = psf.likelihood(theta1, data)
    
                delta[l, i] = 1.j*delta_betas[l, i].imag
                betas1 = betas+delta
                theta1, _ = psf.encode(betas1, Ds, gamma)
    
                liks1_imag[l, i] = psf.likelihood(theta1, data)
        
        grads_expected = np.stack(((liks1_real - liks) / delta_betas.real, (liks1_imag - liks) / delta_betas.imag), axis=1).flatten()
    
        grads = psf.likelihood_grad(theta, data)

        np.testing.assert_array_almost_equal(grads, grads_expected, 2)
            
    def test_deconvolve(self):
        jmax = 5
        arcsec_per_px = 0.055
        diameter = 20.0
        wavelength = 5250.0
        nx = 10
        defocus = 1.0
        
        gamma = 3.
        L = 10

        #fimage = fft.fft2(fft.fftshift(image))#fft.fft2(image)
        #Ds = np.tile(np.array([fimage, fimage]), (L, 1)).reshape((L, 2, nx, nx))
        Ds = np.tile(np.array([image, image]), (L, 1)).reshape((L, 2, nx, nx))
        
        psf = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus)
        psf.create_basis(do_fft=True, do_defocus=True)


        betas = np.random.normal(size=(L, jmax)) + np.random.normal(size=(L, jmax))*1.j

        #Ds, norm = psf.multiply(Ds, betas)
        Ds = psf.convolve(Ds, betas, normalize=True)

        #Ds = fft.ifftshift(fft.ifft2(Ds)).real
        #norm = fft.ifftshift(fft.ifft2(norm)).real
        #norm = np.sum(norm, axis=(2, 3)).repeat(Ds.shape[2]*Ds.shape[3]).reshape((Ds.shape[0], Ds.shape[1], Ds.shape[2], Ds.shape[3]))
        #Ds /= norm

        my_plot = plot.plot(nrows=2, ncols=2)
        my_plot.colormap(image, [0,0])
        my_plot.colormap(Ds[0, 0], [1,0])
        my_plot.colormap(Ds[0, 1], [1,1])
            

        Ds = fft.fft2(fft.fftshift(Ds))
        #Ds = fft.fft2(Ds)
        
        image_back = psf.deconvolve(Ds, betas, gamma, do_fft = True, normalize = True)
        my_plot.colormap(image_back[0], [0,1])
        my_plot.save("test_deconvolve.png")
        my_plot.close()

        #fimage_back = psf.get_restoration(D, D_d, betas, gamma, do_fft = False)

        np.testing.assert_almost_equal(image_back, np.tile(image, (L, 1)).reshape((L, nx, nx)), 10)
        #np.testing.assert_almost_equal(fimage_back, fimage, 8)

        
if __name__ == '__main__':
    unittest.main()
