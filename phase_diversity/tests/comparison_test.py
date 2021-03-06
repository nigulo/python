import sys
sys.path.append('..')
sys.path.append('../../utils')
import numpy as np
import psf_basis
import psf
import unittest
import plot
import utils
import misc
import psf_basis_sampler
import matplotlib.pyplot as plt
import tip_tilt
import numpy.fft as fft

image_a = np.array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ]
                  ])

image_b = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ]
                  ])

image_c = np.array([[0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [1, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                    [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]
                  ])


def calibrate(arcsec_per_px, nx):
    coef = np.log2(float(nx)/11)
    return 2.6*arcsec_per_px*2.**coef

class test_comparison(unittest.TestCase):

    '''
    def test_aberration(self):

        jmax = 10
        #arcsec_per_px = 0.1
        diameter = 20.0
        wavelength = 5250.0
        defocus = 0.
        #arcsec_per_px = 0.057
        #arcsec_per_px = wavelength/diameter*1e-8*180/np.pi*3600

        arcsec_per_px_base = .5*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
        
        counter = 0
        for image in [image_a]:#, image_b, image_c]:
            image1 = image
            nx_orig = np.shape(image1)[0]
    
            image2 = utils.upsample(image1)
            nx = np.shape(image2)[0]
            
            #app_coef = 4.**(-np.log2(float(nx_orig)/11))
       
            #for arcsec_per_px in[0.22*wavelength/diameter*1e-8*180/np.pi*3600]:#, 2.*wavelength/diameter*1e-8*180/np.pi*3600]:
            for arcsec_per_px in[0.5*arcsec_per_px_base, arcsec_per_px_base, 2.*arcsec_per_px_base]:
                #arcsec_per_px *= app_coef
                print("arcsec_per_px=", arcsec_per_px)
            
                aperture_func = lambda xs: utils.aperture_circ(xs, coef=100., radius =1.)
                #aperture_func = lambda xs: utils.aperture_circ(xs, r=.1, coef=100.)
            
                ###################################################################
                pa1 = psf.phase_aberration(np.random.normal(size=jmax))
                defocus_func = lambda xs: defocus*2*np.sum(xs*xs, axis=2)
                ctf1 = psf.coh_trans_func(aperture_func, pa1, defocus_func)
                psf1 = psf.psf(ctf1, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
                D1, D1_d = psf1.convolve(image2)
    
                #my_plot = plot.plot(nrows=1, ncols=2)
                #my_plot.colormap(D0, [0])
                #my_plot.colormap(D1_d, [1])
                #my_plot.save("defous_test" + str(downsample_factor) + ".png")
                #my_plot.close()
    
                ###################################################################
                # Defocus in PSF basis
    
                #psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px1, diameter = diameter, wavelength = wavelength, defocus = defocus*(nx*arcsec_per_px)**2*1.77)
                print("arcsec_per_px before, after", arcsec_per_px, calibrate(arcsec_per_px, nx_orig))
                psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus*2.1)#/(arcsec_per_px**2)/1000)
                psf_b.create_basis()
                betas = np.random.normal(size=jmax) + 1.j*np.random.normal(size=jmax)
                #betas = np.zeros(jmax, dtype='complex')
                Ds = psf_b.convolve(image2, betas)
                D2 = Ds[0, 0]
                D2_d = Ds[0, 1]

                if image is image_a:
                    print("Params")
                    print("jmax", jmax)
                    print("nx", nx)
                    print("arcsec_per_px", calibrate(arcsec_per_px, nx_orig))
                    print("diameter", diameter)
                    print("wavelength", wavelength)
                    print("Ds", Ds)
                    print("betas", betas)
                    #print("D_d", D2_d)


                D2 = np.roll(np.roll(D2, 1, axis=0), 1, axis=1)
                D2_d = np.roll(np.roll(D2_d, 1, axis=0), 1, axis=1)
    
            
                #D1 /= np.std(D1)
                #D1_d /= np.std(D1_d)
                D1 = misc.normalize(D1)
                D1_d = misc.normalize(D1_d)
    
                #D2 /= np.std(D2)
                #D2_d /= np.std(D2_d)
                D2 = misc.normalize(D2)
                D2_d = misc.normalize(D2_d)

                my_plot = plot.plot(nrows=2, ncols=2)
                my_plot.colormap(D1, [0, 0])
                my_plot.colormap(D2, [0, 1])
                my_plot.colormap(D1_d, [1, 0])
                my_plot.colormap(D2_d, [1, 1])
                my_plot.save("comparison_test" + str(counter) + ".png")
                my_plot.close()
            
            
                #np.testing.assert_almost_equal(D1, D2, 1)
                #np.testing.assert_almost_equal(D1_d, D2_d, 1)
                
                counter += 1
    '''
    
    def test_inversion1(self):
        
        # Convolve with Zernike basis and reconstruct with  PSF basis
        # This is a null test with no aberration and defocus
        
        jmax = 1
        #arcsec_per_px = 0.057
        #arcsec_per_px = 0.011
        diameter = 50.0
        wavelength = 5250.0
        gamma = 1.0

        arcsec_per_px = .5*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
        defocus = 0.

        image = plt.imread('../granulation.png')[:, :, 0]
        image = misc.sample_image(image, .25)
        #image = plt.imread('granulation2.png')
        print("Image shape", image.shape)
        nx_orig = 50
        #start_index_max = max(0, min(image.shape[0], image.shape[1]) - nx_orig)
        start_index = 0#np.random.randint(0, start_index_max)
        
        image = image[start_index:start_index + nx_orig,start_index:start_index + nx_orig]
        
        nx_orig = np.shape(image)[0]
        image = utils.upsample(image)
        assert(np.shape(image)[0] == np.shape(image)[1])
        
        nx = np.shape(image)[0]
        

        print("arcsec_per_px=", arcsec_per_px)
    
        aperture_func = lambda xs: utils.aperture_circ(xs, coef=100., radius =1.)
        #aperture_func = lambda xs: utils.aperture_circ(xs, r=.1, coef=100.)
    
        ###################################################################
        pa1 = psf.phase_aberration([])
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)
        ctf1 = psf.coh_trans_func(aperture_func, pa1, defocus_func)
        psf1 = psf.psf(ctf1, nx_orig, arcsec_per_px = arcsec_per_px/10, diameter = diameter, wavelength = wavelength)
        
        #fimage = fft.fft2(image)
        #fimage = fft.fftshift(fimage)
        #DF, DF_d = psf1.multiply(fimage)

        #DF = fft.ifftshift(DF)
        #DF_d = fft.ifftshift(DF_d)
        #D = fft.ifft2(DF).real
        ##D_d = fft.ifft2(DF_d).real

        D, D_d = psf1.convolve(image)
        DF = fft.fft2(D)
        DF_d = fft.fft2(D_d)

        coords, _, _ = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
        tt = tip_tilt.tip_tilt(coords, prior_prec=((np.max(coords[0])-np.min(coords[0]))/2)**2, num_rounds=1)
        psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus, tip_tilt = tt)
        #psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus_psf_b, tip_tilt = tt)
        psf_b.create_basis()

        sampler = psf_basis_sampler.psf_basis_sampler(psf_b, gamma, num_samples=1)
        Ds = np.zeros((1, 2, nx, nx), dtype='complex')
        Ds[0, 0] = DF
        Ds[0, 1] = DF_d

        res = sampler.sample(Ds, "samples.png")
        if tt is not None:
            betas_est, a_est = res
        else:
            betas_est = res
            a_est = None

        image_est, F, Ps = psf_b.deconvolve(Ds, betas_est, gamma, ret_all = True, a_est=a_est, normalize=True)
        
        vmin = np.min(image)
        vmax = np.max(image)

        my_plot = plot.plot(nrows=1, ncols=3)
        my_plot.set_axis()
        
        image_est = fft.ifftshift(image_est, axes=(-2, -1))
        my_plot.colormap(image, [0], vmin=vmin, vmax=vmax)
        my_plot.colormap(D, [1], vmin=vmin, vmax=vmax)
        my_plot.colormap(image_est[0], [2], vmin=vmin, vmax=vmax)
            
        my_plot.save("test_inversion1.png")
        my_plot.close()

    def test_inversion2(self):
        
        # Convolve with Zernike basis and reconstruct with  PSF basis
        # This is a null test with no aberration but defocus
        
        jmax = 1
        #arcsec_per_px = 0.057
        #arcsec_per_px = 0.011
        diameter = 50.0
        wavelength = 5250.0
        gamma = 1.0

        arcsec_per_px = .5*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
        defocus = 1.2*np.pi

        image = plt.imread('../granulation.png')[:, :, 0]
        image = misc.sample_image(image, .25)
        #image = plt.imread('granulation2.png')
        print("Image shape", image.shape)
        nx_orig = 50
        #start_index_max = max(0, min(image.shape[0], image.shape[1]) - nx_orig)
        start_index = 0#np.random.randint(0, start_index_max)
        
        image = image[start_index:start_index + nx_orig,start_index:start_index + nx_orig]
        
        nx_orig = np.shape(image)[0]
        image = utils.upsample(image)
        assert(np.shape(image)[0] == np.shape(image)[1])
        
        nx = np.shape(image)[0]
        

        print("arcsec_per_px=", arcsec_per_px)
    
        aperture_func = lambda xs: utils.aperture_circ(xs, coef=100., radius =1.)
        #aperture_func = lambda xs: utils.aperture_circ(xs, r=.1, coef=100.)
    
        ###################################################################
        pa1 = psf.phase_aberration([])
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)
        ctf1 = psf.coh_trans_func(aperture_func, pa1, defocus_func)
        psf1 = psf.psf(ctf1, nx_orig, arcsec_per_px = arcsec_per_px/20, diameter = diameter, wavelength = wavelength)
        
        #fimage = fft.fft2(image)
        #fimage = fft.fftshift(fimage)
        #DF, DF_d = psf1.multiply(fimage)

        #DF = fft.ifftshift(DF)
        #DF_d = fft.ifftshift(DF_d)
        #D = fft.ifft2(DF).real
        ##D_d = fft.ifft2(DF_d).real

        D, D_d = psf1.convolve(image)
        DF = fft.fft2(D)
        DF_d = fft.fft2(D_d)

        coords, _, _ = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
        tt = tip_tilt.tip_tilt(coords, prior_prec=((np.max(coords[0])-np.min(coords[0]))/2)**2, num_rounds=1)
        psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = 0., tip_tilt = tt)
        #psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus_psf_b, tip_tilt = tt)
        psf_b.create_basis()

        sampler = psf_basis_sampler.psf_basis_sampler(psf_b, gamma, num_samples=1)
        Ds = np.zeros((1, 2, nx, nx), dtype='complex')
        Ds[0, 0] = DF
        Ds[0, 1] = DF_d

        res = sampler.sample(Ds, "samples.png")
        if tt is not None:
            betas_est, a_est = res
        else:
            betas_est = res
            a_est = None

        image_est, F, Ps = psf_b.deconvolve(Ds, betas_est, gamma, ret_all = True, a_est=a_est, normalize=True)
        
        vmin = np.min(image)
        vmax = np.max(image)

        my_plot = plot.plot(nrows=1, ncols=4)
        my_plot.set_axis()
        
        image_est = fft.ifftshift(image_est, axes=(-2, -1))
        my_plot.colormap(image, [0], vmin=vmin, vmax=vmax)
        my_plot.colormap(D, [1], vmin=vmin, vmax=vmax)
        my_plot.colormap(D_d, [2], vmin=vmin, vmax=vmax)
        my_plot.colormap(image_est[0], [3], vmin=vmin, vmax=vmax)
            
        my_plot.save("test_inversion2.png")
        my_plot.close()

        
if __name__ == '__main__':
    unittest.main()
