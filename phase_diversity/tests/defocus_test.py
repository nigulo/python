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
import scipy.optimize
import matplotlib.pyplot as plt

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

'''
class test_defocus(unittest.TestCase):

    def test(self):

        jmax = 2
        #arcsec_per_px = 0.1
        diameter = 20.0
        wavelength = 5250.0

        # Set arcsec_per_px to diffraction limit
        arcsec_per_px_base = .5*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600#wavelength/diameter*1e-8*180/np.pi*3600
        
        print("arcsec_per_px_base=", arcsec_per_px_base)
        counter = 0
        for image in [image_a]:#, image_b, image_c]:
            image1 = image
            nx_orig = np.shape(image1)[0]
    
            image2 = utils.upsample(image1)
            nx = np.shape(image2)[0]
            
            app_coef = 4.**(-np.log2(float(nx_orig)/11))
       
            #for arcsec_per_px in[0.22*wavelength/diameter*1e-8*180/np.pi*3600]:#, 2.*wavelength/diameter*1e-8*180/np.pi*3600]:
            for arcsec_per_px in[0.5*arcsec_per_px_base, arcsec_per_px_base, 2*arcsec_per_px_base]:
                arcsec_per_px *= app_coef
                print("arcsec_per_px=", arcsec_per_px)
                for defocus in[1.5, 5.]:
            
                    aperture_func = lambda xs: utils.aperture_circ(xs, coef=100., radius =1.)
                    #aperture_func = lambda xs: utils.aperture_circ(xs, r=.1, coef=100.)
                
                    ###################################################################
                    # Defocus via fourth Zernike term
                    pa0 = psf.phase_aberration([defocus])
                    ctf0 = psf.coh_trans_func(aperture_func, pa0, lambda xs: 0.)
                    psf0 = psf.psf(ctf0, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
                    #pa0.set_alphas([1.])
                    D0, D0_d = psf0.convolve(image2)
                    
                    ###################################################################
                    # Defocus defined by explicit function
                    pa1 = psf.phase_aberration([])
                    defocus_func = lambda xs: defocus*2*np.sum(xs*xs, axis=2)
                    ctf1 = psf.coh_trans_func(aperture_func, pa1, defocus_func)
                    psf1 = psf.psf(ctf1, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
                    D1, D1_d = psf1.convolve(image2)
        
                    #my_plot = plot.plot(nrows=1, ncols=2)
                    #my_plot.colormap(D0, [0])
                    #my_plot.colormap(D1_d, [1])
                    #my_plot.save("defous_test" + str(downsample_factor) + ".png")
                    #my_plot.close()
        
                    np.testing.assert_almost_equal(D0, D0_d)
                    np.testing.assert_almost_equal(D0_d, D1_d)
        
                    ###################################################################
                    # Defocus in PSF basis
        
                    #psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px1, diameter = diameter, wavelength = wavelength, defocus = defocus*(nx*arcsec_per_px)**2*1.77)
                    psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus*2.1)#/(arcsec_per_px**2)/1000)
                    psf_b.create_basis()
                    betas = np.zeros(jmax, dtype='complex')
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
                    my_plot.save("defous_test" + str(counter) + ".png")
                    my_plot.close()
                
                
                    #np.testing.assert_almost_equal(D1, D2, 1)
                    #np.testing.assert_almost_equal(D1_d, D2_d, 1)
                    
                    counter += 1
'''

class test_find_defocus(unittest.TestCase):

    def test(self):

        jmax = 0
        #arcsec_per_px = 0.1
        diameter = 50.0
        wavelength = 5250.0
        #arcsec_per_px = 0.057
        #arcsec_per_px = wavelength/diameter*1e-8*180/np.pi*3600

        #arcsec_per_px *=2
        
        #image = plt.imread('../granulation31x33arsec.png')
        ##image = misc.sample_image(image,.27)
        #image = misc.sample_image(image,.675)
        
        image = plt.imread('../granulation.png')[:, :, 0]
        image = misc.sample_image(image, .25)
        
        #image = plt.imread('granulation.png')[:, :, 0]
        #image = plt.imread('granulation2.png')
        print("Image shape", image.shape)
        image = image[0:50,0:50]
        #image = image_b
        
        image1 = image
        nx_orig = np.shape(image1)[0]

        # Set arcsec_per_px to diffraction limit
        arcsec_per_px_psf_b = .5*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
        arcsec_per_px = arcsec_per_px_psf_b/20#/nx_orig
        print("arcsec_per_px=", arcsec_per_px)

        #defocus=nx_orig*nx_orig/2
        defocus=2.*np.pi

        image2 = utils.upsample(image1)
        nx = np.shape(image2)[0]
        
        aperture_func = lambda xs: utils.aperture_circ(xs, coef=100., radius =1.)
        #aperture_func = lambda xs: utils.aperture_circ(xs, r=.1, coef=100.)
    
        ###################################################################
        # Defocus defined by explicit function
        
        
        pa1 = psf.phase_aberration([])
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)
        ctf1 = psf.coh_trans_func(aperture_func, pa1, defocus_func)
        psf1 = psf.psf(ctf1, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
        D1, D1_d = psf1.convolve(image2)

        D1 = misc.normalize(D1)
        D1_d = misc.normalize(D1_d)

        my_plot = plot.plot(nrows=3, ncols=2)
        my_plot.colormap(image2, [0, 0])
        my_plot.colormap(image2, [0, 1])
        my_plot.colormap(D1, [1, 0])
        my_plot.colormap(D1_d, [2, 0])
        my_plot.save("find_defocus.png")
        my_plot.close()

        min_loss = None

        cache = {}

        print("defocus", defocus)
        print("arcsec_per_px", arcsec_per_px)
        opt_defocus=125#defocus*15
        opt_arcsec_per_px=arcsec_per_px_psf_b#calibrate(arcsec_per_px, nx_orig)
        coef = .01

        def loss_fn(params):
            defocus = params[0]
            arcsec_per_px = params[1]
            if defocus < 0 or arcsec_per_px <= 0:
                return 1e100
            cache_key = (int(defocus/coef), int(arcsec_per_px/coef))
            if cache_key in cache.keys():
                return cache[cache_key]
            #psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus)
            psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus)
            psf_b.create_basis()
            betas = np.zeros(jmax, dtype='complex')
            Ds = psf_b.convolve(image2, betas)
            D2 = Ds[0, 0]
            D2_d = Ds[0, 1]

            D2 = np.roll(np.roll(D2, 1, axis=0), 1, axis=1)
            D2_d = np.roll(np.roll(D2_d, 1, axis=0), 1, axis=1)

            D2 = misc.normalize(D2)
            D2_d = misc.normalize(D2_d)
            #loss = np.sum((D1_d - D2_d)**2) + np.sum((D1 - D2)**2)
            loss = np.sum(np.abs(D1_d - D2_d)) + np.sum(np.abs(D1 - D2))
            cache[cache_key] = loss

            my_plot = plot.plot(nrows=3, ncols=2)
            my_plot.colormap(image2, [0, 0])
            my_plot.colormap(image2, [0, 1])
            my_plot.colormap(D1, [1, 0])
            my_plot.colormap(D2, [1, 1])
            my_plot.colormap(D1_d, [2, 0])
            my_plot.colormap(D2_d, [2, 1])
            my_plot.save("find_defocus.png")
            my_plot.close()

            return loss

    
        min_loss = None
        min_res = None

        for trial_no in np.arange(0, 1):
            res = scipy.optimize.fmin_cg(loss_fn, [opt_defocus, opt_arcsec_per_px], fprime=None, args=(), full_output=True, epsilon=[coef*opt_defocus, coef*opt_arcsec_per_px], gtol=1e-5)
            loss = res[1]
            #assert(loglik == lik_fn(res['x']))
            if min_loss is None or loss < min_loss:
                min_loss = loss
                min_res = res
        opt_defocus = min_res[0][0]
        opt_arcsec_per_px = min_res[0][1]

        print("opt_defocus", opt_defocus)
        print("opt_arcsec_per_px", opt_arcsec_per_px)
    
        #psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = opt_defocus)
        psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = opt_arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = opt_defocus)
        psf_b.create_basis()
        betas = np.zeros(jmax, dtype='complex')
        Ds = psf_b.convolve(image2, betas)
        D2 = Ds[0, 0]
        D2_d = Ds[0, 1]

        D2 = np.roll(np.roll(D2, 1, axis=0), 1, axis=1)
        D2_d = np.roll(np.roll(D2_d, 1, axis=0), 1, axis=1)

        D2 = misc.normalize(D2)
        D2_d = misc.normalize(D2_d)
    
        my_plot = plot.plot(nrows=3, ncols=2)
        my_plot.colormap(image2, [0, 0])
        my_plot.colormap(image2, [0, 1])
        my_plot.colormap(D1, [1, 0])
        my_plot.colormap(D2, [1, 1])
        my_plot.colormap(D1_d, [2, 0])
        my_plot.colormap(D2_d, [2, 1])
        my_plot.save("find_defocus.png")
        my_plot.close()
        
        

        
if __name__ == '__main__':
    unittest.main()
