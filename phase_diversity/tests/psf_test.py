import sys
sys.path.append('..')
import numpy as np
import psf
import zernike
import numpy.fft as fft
from scipy.signal import correlate2d as correlate
import unittest
import utils
sys.path.append('../../utils')
import plot

image10x10 = np.array([[0.41960785, 0.38039216, 0.36862746, 0.38039216, 0.40784314, 0.40392157,
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
  0.45490196, 0.39607844, 0.36862746, 0.37254903]])

'''
image20x20 = np.array([[0.41960785, 0.38039216, 0.36862746, 0.38039216, 0.40784314, 0.40392157,
  0.38431373, 0.4509804, 0.45882353, 0.5137255, 0.49803922, 0.49803922,
  0.49019608, 0.4627451,  0.43529412, 0.42352942, 0.44705883, 0.46666667,
  0.52156866, 0.54509807],
 [0.4117647,  0.38039216, 0.39607844, 0.39215687, 0.34117648, 0.3529412,
  0.35686275, 0.37254903, 0.36862746, 0.38039216, 0.43137255, 0.43137255,
  0.4,        0.43529412, 0.44313726, 0.45490196, 0.47843137, 0.5137255,
  0.56078434, 0.56078434],
 [0.36862746, 0.34901962, 0.30980393, 0.3254902,  0.31764707, 0.3372549,
  0.3019608,  0.3254902,  0.33333334, 0.34901962, 0.35686275, 0.35686275,
  0.3882353,  0.42745098, 0.44313726, 0.45882353, 0.49411765, 0.52156866,
  0.5137255,  0.5294118 ],
 [0.3254902,  0.34509805, 0.3647059,  0.37254903, 0.41568628, 0.36078432,
  0.33333334, 0.32156864, 0.28235295, 0.30980393, 0.31764707, 0.3137255,
  0.3764706,  0.42352942, 0.44313726, 0.45882353, 0.49019608, 0.4745098,
  0.43529412, 0.43137255],
 [0.4392157,  0.4509804,  0.5019608,  0.4627451,  0.4745098,  0.43529412,
  0.36078432, 0.3254902,  0.2901961,  0.2627451,  0.34509805, 0.33333334,
  0.34901962, 0.37254903, 0.4392157,  0.4509804,  0.43137255, 0.43137255,
  0.38431373, 0.40392157],
 [0.54901963, 0.5058824,  0.56078434, 0.56078434, 0.5803922,  0.49803922,
  0.3882353,  0.34117648, 0.28627452, 0.30588236, 0.30588236, 0.34509805,
  0.34117648, 0.3882353,  0.40392157, 0.41960785, 0.4392157,  0.40784314,
  0.39215687, 0.38039216],
 [0.6039216,  0.61960787, 0.64705884, 0.61960787, 0.627451,   0.5568628,
  0.42745098, 0.3647059,  0.32941177, 0.32156864, 0.32156864, 0.3764706,
  0.34509805, 0.39607844, 0.41568628, 0.4117647,  0.43137255, 0.4117647,
  0.4117647,  0.42745098],
 [0.6666667,  0.7294118,  0.69803923, 0.7176471,  0.62352943, 0.5803922,
  0.45882353, 0.37254903, 0.3372549,  0.3372549,  0.37254903, 0.3764706,
  0.39607844, 0.40392157, 0.43529412, 0.44313726, 0.45882353, 0.46666667,
  0.43137255, 0.45490196],
 [0.6745098,  0.654902,   0.7019608,  0.6862745,  0.6431373,  0.5529412,
  0.42352942, 0.40392157, 0.37254903, 0.39215687, 0.38431373, 0.40784314,
  0.42352942, 0.43137255, 0.45882353, 0.47058824, 0.48235294, 0.4745098,
  0.4627451,  0.48235294],
 [0.6509804,  0.6901961,  0.6509804,  0.6392157,  0.58431375, 0.5294118,
  0.45490196, 0.39607844, 0.36862746, 0.37254903, 0.4117647,  0.4,
  0.4392157,  0.4509804,  0.4509804,  0.49411765, 0.47843137, 0.4627451,
  0.47058824, 0.49803922], 
 [0.61960787, 0.654902,   0.63529414, 0.6313726,  0.5686275,  0.4862745,
  0.40392157, 0.38039216, 0.40392157, 0.38039216, 0.42352942, 0.4392157,
  0.42745098, 0.48235294, 0.5058824,  0.5058824,  0.48235294, 0.45490196,
  0.47843137, 0.48235294],
 [0.6431373,  0.61960787, 0.5882353,  0.56078434, 0.53333336, 0.43137255,
  0.39215687, 0.37254903, 0.36862746, 0.38431373, 0.40784314, 0.43529412,
  0.48235294, 0.5411765,  0.53333336, 0.49411765, 0.4627451,  0.4509804,
  0.4745098,  0.5529412 ],
 [0.5764706,  0.58431375, 0.60784316, 0.5686275,  0.4862745,  0.4392157,
  0.38039216, 0.39215687, 0.41568628, 0.40784314, 0.43529412, 0.44705883,
  0.49803922, 0.5294118,  0.5176471,  0.4627451,  0.4627451,  0.45490196,
  0.49803922, 0.5372549 ],
 [0.58431375, 0.627451,   0.5764706,  0.56078434, 0.49019608, 0.4,
  0.3882353,  0.39215687, 0.4,        0.43137255, 0.41960785, 0.45490196,
  0.4862745,  0.5137255,  0.5254902,  0.5019608,  0.46666667, 0.47058824,
  0.5254902,  0.5803922 ],
 [0.5568628,  0.59607846, 0.59607846, 0.5411765,  0.52156866, 0.4117647,
  0.39215687, 0.40392157, 0.4392157,  0.4509804,  0.4509804,  0.45490196,
  0.49803922, 0.5294118,  0.5137255,  0.5411765,  0.5058824,  0.50980395,
  0.54509807, 0.6039216 ],
 [0.52156866, 0.5294118,  0.5372549,  0.49803922, 0.49411765, 0.4392157,
  0.38039216, 0.45490196, 0.4627451,  0.5058824,  0.5176471,  0.5411765,
  0.5921569,  0.59607846, 0.6039216,  0.5647059,  0.54901963, 0.54901963,
  0.58431375, 0.627451  ],
 [0.5137255,  0.50980395, 0.49803922, 0.46666667, 0.48235294, 0.46666667,
  0.42352942, 0.4627451,  0.49411765, 0.5647059,  0.5647059,  0.57254905,
  0.627451,   0.67058825, 0.6627451,  0.6313726,  0.6,        0.56078434,
  0.59607846, 0.6313726 ],
 [0.4862745,  0.49411765, 0.45490196, 0.42745098, 0.47058824, 0.44705883,
  0.4745098,  0.46666667, 0.54901963, 0.5686275,  0.58431375, 0.6784314,
  0.7176471,  0.70980394, 0.65882355, 0.63529414, 0.5803922,  0.5764706,
  0.59607846, 0.6431373 ],
 [0.4627451,  0.47058824, 0.41960785, 0.48235294, 0.42745098, 0.4862745,
  0.5019608,  0.5647059,  0.5137255,  0.5764706,  0.627451,   0.654902,
  0.7137255,  0.69411767, 0.6666667,  0.6117647,  0.5764706,  0.57254905,
  0.6039216,  0.6392157 ],
 [0.39215687, 0.4509804,  0.43137255, 0.44313726, 0.46666667, 0.4745098,
  0.49411765, 0.5137255,  0.5176471,  0.5686275,  0.5686275,  0.627451,
  0.63529414, 0.68235296, 0.6156863,  0.5921569,  0.5372549,  0.5411765,
  0.6,        0.60784316]])
'''

aperture_func = lambda xs: utils.aperture_circ(xs, 1.0, 15.0)

class test_phase_aberration(unittest.TestCase):
    
    def test(self):
        n_coefs = 25
        coefs = np.random.normal(size=n_coefs)
        xs = np.array([[[0.5, 0.5], [0.5, -0.5]], [[-0.5, -0.5], [-0.5, 0.5]]])
        pa = psf.phase_aberration(coefs)#zip(np.arange(1, n_coefs + 1), coefs))
        pa.calc_terms(xs)
        
        vals = pa()
        pol_vals = pa.get_pol_values()
        coefs1 = np.reshape(np.repeat(coefs, xs.shape[0]*xs.shape[1], axis=0), np.shape(pol_vals))
        np.testing.assert_almost_equal(np.sum(pol_vals*coefs1, axis=0), vals, 15)
        
        polar = utils.cart_to_polar(xs)
        
        for noll_index in np.arange(1, n_coefs + 1):
            n, m = zernike.get_nm(noll_index)
            z = zernike.zernike(n, m)
            z_val = z.get_value(polar)
            np.testing.assert_almost_equal(pol_vals[noll_index - 1], z_val, 15)
            
'''
# Returns the same result as scipy.signal.correlate2d
def corr2d(d):
    d1 = d - np.mean(d)
    nx = d.shape[0]
    ny = d.shape[1]
    corr = np.zeros((nx*2 - 1, ny*2 - 1), dtype='complex')
    counts = np.zeros((nx*2 - 1, ny*2 - 1))
    for i in np.arange(-nx + 1, nx):
        for j in np.arange(-ny + 1, ny):
            for k in np.arange(0, nx):
                if k + i >= 0 and k + i < nx:
                    for l in np.arange(0, ny):
                        if l + j >= 0 and l + j < ny:
                            corr[i, j] += d1[k+i, l+j]*d1[k, l].conj()
                            counts[i,j] += 1
    return fft.ifftshift(corr)

def comp_corr(wfs, mask):
    pupil = mask*np.exp(1j*wfs)
    #pupil = fft.ifftshift(pupil) 
    
    my_plot = plot.plot_map(nrows=1, ncols=2)
    my_plot.plot(correlate(pupil, pupil, mode='full').real, [0])
    my_plot.plot(corr2d(pupil).real, [1])
        
    my_plot.save("corr_comp.png")
    my_plot.close()
'''
        
def calc_psf_via_corr(wfs, mask):
 
    #psf = np.zeros((2*wfs.shape[0]-1, 2*wfs.shape[1]-1))
    pupil = mask*np.exp(1.j*wfs)
    corr = correlate(pupil, pupil.conj(), mode='full')
    psf = fft.fftshift(fft.ifft2(fft.ifftshift(corr))).real
    
    mser = np.sum(psf.real**2)
    msei = np.sum(psf.imag**2)
    my_plot = plot.plot_map(nrows=1, ncols=2)
    my_plot.plot(psf.real, [0])
    my_plot.plot(psf.imag, [1])
    my_plot.save("power.png")
    my_plot.close()
    print("mser, msei", mser, msei)
    
    #psf /= psf.sum()
    return psf


def calc_psf_via_fft(wfs, mask):
    pupil = mask*np.exp(1j*wfs)
    #pupil = fft.ifftshift(pupil)
    #pupil -= np.mean(pupil)
    
    nx = pupil.shape[0]
    ny = pupil.shape[1]
    pupil_mirror_x = pupil#[::-1,:]
    pupil_mirror_y = pupil#[:,::-1]
    pupil_mirror_xy = pupil#[::-1,::-1]
    
    #pupil_full = utils.resize(pupil.real) + 1.j*utils.resize(pupil.imag)
    pupil_full = np.zeros((nx*2 - 1, ny*2 - 1), dtype='complex')
    pupil_full[:nx, :ny] = pupil
    pupil_full[:nx, :ny] += pupil_mirror_xy
    #pupil_full[nx-1:, :ny] = pupil_mirror_x
    #pupil_full[:nx, ny-1:] = pupil_mirror_y
    #pupil_full[nx-1:, ny-1:] = pupil_mirror_xy
    #pupil_full = np.roll(np.roll(pupil_full, int(nx/2), axis=0), int(ny/2), axis=1)
    
    
    my_plot = plot.plot_map(nrows=1, ncols=2)
    my_plot.plot(pupil_full.real, [0])
    my_plot.plot(pupil_full.imag, [1])
        
    my_plot.save("pupil_full.png")
    my_plot.close()
    
    #pupil += pupil[::-1,::-1]#.conjugate()
    #pupil -= np.mean(pupil)
    #pupil /= np.std(pupil)

    #i_m = pupil.shape[0]/2
    #j_m = pupil.shape[1]/2
    #for i in np.arange(0, pupil.shape[0]):
    #    for j in np.arange(0, pupil.shape[1]):
    #        pupil[i, j] *= 4.*(np.abs(i-i_m)*np.abs(j-j_m)/pupil.shape[0]/pupil.shape[1])

    vals = fft.ifft2(pupil)
    vals = (vals*vals.conjugate()).real
    vals = fft.ifftshift(vals)
    
    #vals /= vals.sum()
    #vals += vals.T
    #vals += vals[::-1,::-1]

    #vals2 = fft.ifft2(pupil[::-1,::-1])
    #vals2 = (vals2*vals2.conjugate()).real
    #vals2 = fft.fftshift(vals2)

    #vals += vals2
    #print("MEAN:",np.mean(vals))
    vals -= np.mean(vals)
    #print("VALS:", vals, vals[::-1,::-1])
    #print("VALS:", vals.T)

    return vals


class test_psf(unittest.TestCase):


    '''
    def test_calc2(self):
        aperture_func = lambda xs: utils.aperture_circ(xs, 0.2, 15.0)

        n_coefs = 25
        coefs = np.random.normal(size=n_coefs)*10
        #coefs= [0., 10., 0.]
        size = 20
        psf_vals = np.zeros((size, size))

        pa = psf.phase_aberration(coefs)#zip(np.arange(1, n_coefs + 1), coefs))
        ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 0.0)
        #ctf = psf.coh_trans_func(aperture_func, pa, lambda u: 0.0)
            
        _ = psf.psf(ctf, size, size).calc()

        x1 = np.linspace(-1., 1., size)/np.sqrt(2)
        x2 = np.linspace(-1., 1., size)/np.sqrt(2)
        coords = np.dstack(np.meshgrid(x1, x2))
        pupil = aperture_func(coords)
        
        psf_vals = calc_psf_via_corr(pa(), pupil)
        psf_vals_expected = calc_psf_via_fft(pa(), pupil)

        my_plot = plot.plot_map(nrows=1, ncols=2)
        my_plot.plot(psf_vals, [0])
        my_plot.plot(psf_vals_expected, [1])
            
        my_plot.save("test_calc2.png")
        my_plot.close()

        np.testing.assert_almost_equal(psf_vals, psf_vals_expected, 8)
    
    def test_calc(self):
        n_coefs = 25
        coefs = np.random.normal(size=n_coefs)
        size = 3
        psf_vals = np.zeros((size, size))
            
        pa = psf.phase_aberration(coefs)#zip(np.arange(1, n_coefs + 1), coefs))
        ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 0.0)
        #ctf = psf.coh_trans_func(aperture_func, pa, lambda u: 0.0)
            
        psf_vals = psf.psf(ctf, size, size).calc()

        x1 = np.linspace(-1., 1., size)/np.sqrt(2)
        x2 = np.linspace(-1., 1., size)/np.sqrt(2)
        coords = np.dstack(np.meshgrid(x1, x2))
        pupil = aperture_func(coords)
        
        psf_vals_expected = calc_psf_via_corr(pa(), pupil)

        np.testing.assert_almost_equal(psf_vals, psf_vals_expected, 8)

    def test_deconvolve(self):
        nx = np.shape(image10x10)[0]
        ny = np.shape(image10x10)[1]
        image1 = utils.resize(image10x10)

        #######################################################################
        # First test that the convolution has no effect in case
        # we have a full aperture and zero wavefront aberration
        pa = psf.phase_aberration([])

        
        ctf = psf.coh_trans_func(lambda xs: 1.0, pa, lambda xs: 0.0)
        psf_ = psf.psf(ctf, nx, ny)
        
        D, D_d = psf_.convolve(image1)
        # No defocus, so D should be equal to D_d
        np.testing.assert_almost_equal(D, D_d, 8)



        threshold = np.ones_like(D)*0.01
        np.testing.assert_array_less((D - image1)**2, threshold)
        
        #######################################################################
        # Now test actual convolution

        n_coefs = 20
        coefs = np.random.normal(size=n_coefs)*10.
        pa = psf.phase_aberration(coefs)#zip(np.arange(1, n_coefs + 1), coefs))

        ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 100.*(2*np.sum(xs*xs, axis=2) - 1.))
        psf_ = psf.psf(ctf, nx, ny)
        
        image1_F = fft.fft2(image1)
        D, D_d = psf_.multiply(image1_F)
        reconst = psf_.deconvolve(D, D_d, gamma=1, do_fft=True)
        np.testing.assert_almost_equal(reconst, image1, 15)


       
    def test_likelihood(self):
        
        n_coefs = 20
        gamma = 1.
        
        nx = np.shape(image10x10)[0]
        ny = np.shape(image10x10)[1]

        #######################################################################
        # Create data
        pa = psf.phase_aberration(np.random.normal(size=n_coefs)*10.)
        ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 100.*(2*np.sum(xs*xs, axis=2) - 1.))
        psf_ = psf.psf(ctf, nx, ny)
        
        image1 = utils.resize(image10x10)
        image1_F = fft.fft2(image1)
        D, D_d = psf_.multiply(image1_F)
        
        #######################################################################
        # Calculate likelihood

        pa = psf.phase_aberration(n_coefs)#zip(np.arange(1, n_coefs + 1), coefs))

        ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 100.*(2*np.sum(xs*xs, axis=2) - 1.))
        psf_ = psf.psf(ctf, nx, ny)
        
        alphas = np.random.normal(size=n_coefs)*10.
        lik = psf_.likelihood(alphas, [D, D_d, gamma])

        #######################################################################
        # Check against expected value
        pa = psf.phase_aberration(alphas)

        ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 100.*(2*np.sum(xs*xs, axis=2) - 1.))
        psf_ = psf.psf(ctf, nx, ny)

        S = psf_.calc_otf(defocus = False)
        S_d = psf_.calc_otf(defocus = True)

        num = D_d*S - D*S_d
        num *= num.conjugate()
        den = S*S.conjugate()+gamma*S_d*S_d.conjugate() + 1e-15

        lik_expected = np.sum((num/den).real)
        
        np.testing.assert_almost_equal(lik, lik_expected, 7)
    '''
    '''
    def test_likelihood_grad(self):
        n_coefs = 20
        gamma = 1.
        
        nx = np.shape(image)[0]
        ny = np.shape(image)[1]

        #######################################################################
        # Create data
        pa = psf.phase_aberration(np.random.normal(size=n_coefs)*10.)
        ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 100.*(2*np.sum(xs*xs, axis=2) - 1.))
        psf_ = psf.psf(ctf, nx, ny)
        
        image1 = utils.resize(image)
        image1_F = fft.fft2(image1)
        D, D_d = psf_.multiply(image1_F)
        
        #######################################################################
        # Calculate gradients

        pa = psf.phase_aberration(n_coefs)#zip(np.arange(1, n_coefs + 1), coefs))

        ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 100.*(2*np.sum(xs*xs, axis=2) - 1.))
        psf_ = psf.psf(ctf, nx, ny)
        
        alphas = np.random.normal(size=n_coefs)*10.
        grads = psf_.likelihood_grad(alphas, [D, D_d, gamma])

        #######################################################################
        # Check against values calculated using finite differences
        delta_alphas = alphas*1.0e-7

        lik = psf_.likelihood(alphas, [D, D_d, gamma])
        liks = np.repeat(lik, len(alphas))
        liks1 = np.zeros_like(alphas)
        for i in np.arange(0, len(alphas)):
            delta = np.zeros_like(alphas)
            delta[i] = delta_alphas[i]
            liks1[i] = psf_.likelihood(alphas+delta, [D, D_d, gamma])

        grads_expected = (liks1 - liks) / delta_alphas

        np.testing.assert_almost_equal(grads, grads_expected, 8)
        '''

    def test_S_prime(self):
        n_coefs = 25
        gamma = 1.
        
        nx = np.shape(image10x10)[0]
        ny = np.shape(image10x10)[1]

        #######################################################################
        # Create data
        pa = psf.phase_aberration(np.random.normal(size=n_coefs)*10.)
        ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 100.*(2*np.sum(xs*xs, axis=2) - 1.))
        psf_ = psf.psf(ctf, nx, ny)
        
        image1 = utils.resize(image10x10)
        image1_F = fft.fft2(image1)
        D, D_d = psf_.multiply(image1_F)
        
        #######################################################################
        # Calculate gradients

        pa = psf.phase_aberration(n_coefs)#zip(np.arange(1, n_coefs + 1), coefs))

        ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 100.*(2*np.sum(xs*xs, axis=2) - 1.))
        psf_ = psf.psf(ctf, nx, ny)
        
        alphas = np.random.normal(size=n_coefs)*10.
        grads = psf_.S_prime(alphas, [D, D_d, gamma])

        #######################################################################
        # Check against values calculated using finite differences
        delta_alphas = alphas*1.0e-10
        pa = psf_.coh_trans_func.phase_aberr

        S = psf_.calc_otf(defocus=False)
        Ss = np.broadcast_to(S, (len(alphas), S.shape[0], S.shape[1]))
        print(Ss.shape)
        Ss1 = np.zeros_like(grads)
        for i in np.arange(0, len(alphas)):
            delta = np.zeros_like(alphas)
            delta[i] = delta_alphas[i]
            pa.set_alphas(alphas+delta)
            Ss1[i] = psf_.calc_otf(defocus=False)

        delta_alphas1 = np.reshape(np.repeat(delta_alphas, S.shape[0]*S.shape[1], axis=0), np.shape(grads))
        grads_expected = (Ss1 - Ss) / delta_alphas1
        
        num_plots = min(4, len(alphas))
        my_plot = plot.plot_map(nrows=4, ncols=num_plots)
        for i in np.arange(0, num_plots):
            my_plot.plot(grads[i].real, [0, i])
            my_plot.plot(np.abs(grads_expected[i].real-grads[i].real), [1, i])
            my_plot.plot(grads[i].imag, [2, i])
            my_plot.plot(np.abs(grads_expected[i].imag-grads[i].imag), [3, i])
            
        my_plot.save("test.png")
        my_plot.close()

        np.testing.assert_almost_equal(grads, grads_expected, 8)


        
if __name__ == '__main__':
    unittest.main()