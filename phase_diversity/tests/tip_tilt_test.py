import sys
sys.path.append('..')
import numpy as np
import tip_tilt
import unittest
import numpy.fft as fft
sys.path.append('../../utils')
import plot
import psf
import utils
import matplotlib.pyplot as plt

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

class test_tip_tilt(unittest.TestCase):
    
    def test_lik_and_grad(self):
        prior_prec = 1.
        k = 2
        l = 10
        
        nx = 20

        d0 = np.random.normal(size=(k, nx, nx))
        
        D = np.zeros((l, k, nx, nx), dtype='complex')
        #F = np.zeros((l, 1, nx, nx))
        for i in np.arange(0, l):
            x_shift = int(np.random.uniform(-nx/2, nx/2))
            y_shift = int(np.random.uniform(-nx/2, nx/2))
            D[i] = fft.fft2(np.roll(np.roll(d0, x_shift, axis=0), y_shift, axis=1))
            #F[i, 0] = np.absolute(D[i])[0]
        
        #D = np.random.normal(size=(l, k, 20, 20)) + np.random.normal(size=(l, k, 20, 20))*1.j
        S = np.ones((l, k, nx, nx), dtype='complex')
        
        xs = np.linspace(-1., 1., D.shape[2])
        coords = np.dstack(np.meshgrid(xs, xs)[::-1])
        tt = tip_tilt.tip_tilt(coords, prior_prec=prior_prec)
        tt.set_data(D, S)#, F)
        coords = fft.ifftshift(coords)
        
        theta = np.random.normal(size=2*(l+1), scale=1./np.sqrt(prior_prec + 1e-10))

        #######################################################################
        # Test likelihood
        #######################################################################
        lik = tt.lik(theta)

        a = theta[0:2*l].reshape((l, 2))
        a0 = theta[2*l:2*l+2]
        au = np.tensordot(a, coords, axes=(1, 2)) + np.tensordot(a0, coords, axes=(0, 2))

        C_T = np.transpose(np.absolute(S)*np.absolute(D)*np.absolute(D), axes=(1, 0, 2, 3)) # swap k and l
        D_T = np.transpose(np.angle(D)-np.angle(S), axes=(1, 0, 2, 3)) # swap k and l

        phi = D_T - au
        lik_expected = np.sum(C_T*np.cos(phi))
        lik_expected += np.sum(a*a)*prior_prec/2

        np.testing.assert_almost_equal(lik, lik_expected)
        
        #######################################################################
        # Test likelihood gradient
        #######################################################################

        grads = tt.lik_grad(theta)
        print(grads.shape)

        #######################################################################
        # Check against values calculated using finite differences
        delta_theta = np.ones_like(theta)*1e-8#+1e-12

        lik = tt.lik(theta)
        liks = np.repeat(lik, len(theta))
        liks1 = np.zeros_like(theta)
        for i in np.arange(0, len(theta)):
            delta = np.zeros_like(theta)
            delta[i] = delta_theta[i]
            liks1[i] = tt.lik(theta+delta)

        grads_expected = (liks1 - liks) / delta_theta
        print(grads_expected.shape)

        np.testing.assert_almost_equal(grads, grads_expected, 1)

    def test_calc(self):
        l = 10
        
        d0 = plt.imread('../granulation.png')[:, :, 0]
        d0 = d0[:199, :199]
        d0 = utils.downsample(d0)
        #d0 = d0[:39, :39]
        #d0 = utils.downsample(d0)

        #d0 = image20x20#np.random.normal(size=(k, nx, nx))
        #d0 = utils.upsample(d0)
        nx = d0.shape[0]

        #######################################################################
        # PSF for defocusing
        pa0 = psf.phase_aberration([200.])
        aperture_func = lambda xs: utils.aperture_circ(xs, coef=100., radius =1.)
        ctf0 = psf.coh_trans_func(aperture_func, pa0, lambda xs: 0.)
        
        diameter = 20.0
        wavelength = 5250.0
        arcsec_per_px = .5*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600#wavelength/diameter*1e-8*180/np.pi*3600
        psf0 = psf.psf(ctf0, nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
        #######################################################################


        d_shifted = np.zeros((l, nx, nx))
        d_d_shifted = np.zeros((l, nx, nx))
        
        D = np.zeros((l, 2, nx, nx), dtype='complex')
        #F = np.zeros((l, 1, nx, nx), dtype='complex')
        S = np.ones((l, 2, nx, nx), dtype='complex')
        for i in np.arange(0, l):
            x_shift = int(np.random.uniform(-nx/10, nx/10))
            y_shift = int(np.random.uniform(-nx/10, nx/10))
            #x_shift = int(np.random.normal()*10)
            #y_shift = int(np.random.normal()*10)
            d_shifted[i] = np.roll(np.roll(d0, x_shift, axis=0), y_shift, axis=1)
            D[i, 0] = fft.fft2(fft.fftshift(d_shifted[i]))
            #F[i, 0] = np.absolute(D[i, 0])

            #######################################################################
            # Just generate an arbitrarily defocused image
            _, d_d = psf0.convolve(utils.upsample(d_shifted[i]))
            d_d_shifted[i] = utils.downsample(d_d)
            D[i, 1] = fft.fft2(fft.fftshift(d_d_shifted[i]))
            #######################################################################
        
        #D = np.random.normal(size=(l, k, 20, 20)) + np.random.normal(size=(l, k, 20, 20))*1.j
        
        x_max = 1.
        x_min = -1.
        delta = 0.
        if (nx % 2) == 0:
            delta = (x_max - x_min)/nx
        xs = np.linspace(x_min, x_max-delta, nx)
        coords = np.dstack(np.meshgrid(xs, xs))
        #tt = tip_tilt.tip_tilt(coords, prior_prec=((x_max - x_min)/2)**2, num_rounds=1)
        tt = tip_tilt.tip_tilt(coords, initial_a=np.zeros((l+1, 2)), prior_prec=((x_max - x_min)/2)**2., num_rounds=1)
        tt.set_data(D, S)#, F)
        image, _, _ = tt.calc()

        my_plot = plot.plot(nrows=l, ncols=4)
            
        for i in np.arange(0, l):
            my_plot.colormap(d0, [i, 0])
            my_plot.colormap(d_shifted[i], [i, 1])
            my_plot.colormap(d_d_shifted[i], [i, 2])
            my_plot.colormap(image[i], [i, 3])
            #np.testing.assert_almost_equal(image[i], D0)
        my_plot.save("tip_tilt_test.png")
        my_plot.close()

        
if __name__ == '__main__':
    unittest.main()
