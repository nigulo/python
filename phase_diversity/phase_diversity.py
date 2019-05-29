import os
os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=6


import numpy as np
import scipy.misc
import scipy.special as special
import scipy.signal as signal
import numpy.fft as fft
import utils
import sys
sys.path.append('../utils')
import plot
import misc
import psf_basis_sampler

import pymc3 as pm
import pyhdust.triangle as triangle
import psf_basis
import psf
import kolmogorov
import scipy.optimize
import matplotlib.pyplot as plt
import tip_tilt

import pickle


image1 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  ]
                  ])

image = np.array([[0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
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

state_file = None#state.pkl"
if len(sys.argv) > 1:
    state_file = sys.argv[1]

print(state_file)

def load(filename):
    if filename is not None:
        data_file = filename
        if os.path.isfile(data_file):
            return pickle.load(open(data_file, 'rb'))
    return None

def save(filename, state):
    if filename is None:
        filename = "state.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(state, f)

num_frames = 5

image = plt.imread('granulation1.png')
print("Image shape", image.shape)
image = image[0:100,0:100, 0]

nx_orig = np.shape(image)[0]
image = utils.upsample(image)
assert(np.shape(image)[0] == np.shape(image)[1])

image_orig = np.array(image)

state = load(state_file)

def get_params(nx):
    coef1 = 4.**(-np.log2(float(nx)/11))
    coef2 = 2.**(-np.log2(float(nx)/11))
    print("coef1, coef2", coef1, coef2)
    arcsec_per_px = coef1*0.2
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 0.#3.
    return (arcsec_per_px, defocus)

def calibrate(arcsec_per_px, nx):
    coef = np.log2(float(nx)/11)
    return 3.0*arcsec_per_px*2.**coef


if state == None:
    print("Creating new state")
    jmax = 10
    #arcsec_per_px = 0.057
    #arcsec_per_px = 0.011
    diameter = 20.0
    wavelength = 5250.0
    gamma = 1.0
    nx = np.shape(image)[0]

    arcsec_per_px, defocus = get_params(nx_orig)#wavelength/diameter*1e-8*180/np.pi*3600
    #arcsec_per_px1=wavelength/diameter*1e-8*180/np.pi*3600/4.58

    psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus*2.2)
    psf_b.create_basis()

    save(state_file, [jmax, arcsec_per_px, diameter, wavelength, defocus, gamma, nx, psf_b.get_state()])
else:
    print("Using saved state")
    jmax = state[0]
    arcsec_per_px = state[1]
    diameter = state[2]
    wavelength = state[3]
    defocus = state[4]
    gamma = state[5]
    nx = state[6]
    #arcsec_per_px1=wavelength/diameter*1e-8*180/np.pi*3600/4.58
    
    assert(nx == np.shape(image)[0])
    
    psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus*2.2)
    psf_b.set_state(state[7])
    

fimage = fft.fft2(image)
fimage = fft.fftshift(fimage)


#aperture_func = lambda xs: utils.aperture_circ(xs, diameter, 15.0)
aperture_func = lambda xs: utils.aperture_circ(xs, coef=15., radius =1.)
#defocus_func = lambda xs: 2.*np.pi*np.sum(xs*xs, axis=2)#10.*(2*np.sum(xs*xs, axis=2) - 1.)
#defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)
defocus_func = lambda xs: defocus*2*np.sum(xs*xs, axis=2)

my_plot = plot.plot(nrows=num_frames + 1, ncols=7)

image_est_mean = np.zeros((nx, nx))
image_est_tt_mean = np.zeros((nx, nx))
D_mean = np.zeros((nx, nx))
D_d_mean = np.zeros((nx, nx))
        
image_norm = misc.normalize(image)
-12.128653119414565
wavefront = kolmogorov.kolmogorov(fried = np.array([.1]), num_realizations=num_frames, size=4*nx_orig, sampling=1.)

sampler = psf_basis_sampler.psf_basis_sampler(psf_b, gamma, num_samples=1)


betass = []


Ds = np.zeros((num_frames, 2, nx, nx), dtype='complex')
Ps = np.zeros((num_frames, 2, nx, nx), dtype='complex')
Fs = np.zeros((num_frames, 1, nx, nx), dtype='complex')

D0 = None
for trial in np.arange(0, num_frames):
    
    pa = psf.phase_aberration(np.random.normal(size=2)*10, start_index=1)
    print("Alphas", pa.alphas)
    #pa = psf.phase_aberration(np.random.normal(size=5)*.001)
    #pa = psf.phase_aberration([])
    ctf = psf.coh_trans_func(aperture_func, pa, defocus_func)
    #ctf = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[0,trial,:,:]), defocus_func)
    psf_ = psf.psf(ctf, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
    
    D, D_d = psf_.multiply(fimage)
    #betas = np.random.normal(size=5) + 1.j*np.random.normal(size=5)
    #betas = np.zeros(jmax, dtype='complex')
    #D, D_d = psf_b.multiply(fimage, betas)

    D = fft.ifftshift(D)
    D_d = fft.ifftshift(D_d)
    
    #betas_est = np.zeros(jmax, dtype='complex')
    #D1, D1_d = psf_b.convolve(image, betas_est)
    
    betas_est = sampler.sample(D, D_d, "samples" + str(trial) + ".png")
    print("betas_est", len(betas_est))
    image_est, F, P, P_d = psf_b.deconvolve(D, D_d, betas_est, gamma, ret_all = True)
    Ds[trial, 0] = D
    Ds[trial, 1] = D_d
    Ps[trial, 0] = P
    Ps[trial, 1] = P_d
    Fs[trial, 0] = F

    #image_est = psf_basis.maybe_invert(image_est, image)

    betass.append(betas_est)
    #image_est = psf_.deconvolve(D, D_d, gamma, do_fft = True)


    D = fft.ifft2(D).real
    D_d = fft.ifft2(D_d).real

    #image_min = np.min(image)
    #image_max = np.max(image)
    
    D_norm = misc.normalize(D)
    D_d_norm = misc.normalize(D_d)
    image_est_norm = misc.normalize(image_est)


    #if D0 is not None:
    #    corr = signal.correlate2d(D0, D_norm, mode='full')
    #    plot_corr = plot.plot(nrows=1, ncols=1)
    #    plot_corr.colormap(corr)
    #    plot_corr.save("corr.png")
    #    plot_corr.close()
    #    
    #    shift = np.unravel_index(np.argmax(corr, axis=None), corr.shape) - np.array([int(corr.shape[0]/2), int(corr.shape[1]/2)])
    #    print("shift:", shift)
    #    D_norm = np.roll(np.roll(D_norm, int(shift[0]), axis=0), int(shift[1]), axis=1)
    #    D_d_norm = np.roll(np.roll(D_d_norm, int(shift[0]), axis=0), int(shift[1]), axis=1)
    #    image_est_norm = np.roll(np.roll(image_est_norm, int(shift[0]), axis=0), int(shift[1]), axis=1)
    #
    #else:
    #    D0 = D_norm


    my_plot.colormap(image, [trial, 0])
    my_plot.colormap(D, [trial, 1])
    my_plot.colormap(D_d, [trial, 2])
    my_plot.colormap(image_est, [trial, 3])

    my_plot.colormap(np.abs(image_est_norm-image_norm), [trial, 5])
    
    #image_est = fft.ifft2(fimage_est).real
    #image_est = np.roll(np.roll(image_est, int(nx/2), axis=0), int(ny/2), axis=1)
    image_est_mean += image_est_norm

    D_mean += D_norm
    D_d_mean += D_d_norm

    my_plot.save("estimates.png")




#tt = tip_tilt.tip_tilt(Ds, Ps, Fs, psf_b.coords)
#a, f = tt.calc()
#tt = tip_tilt.tip_tilt(np.array([np.stack((D, D_d))]), np.array([np.stack((P, P_d))]), np.array([[F]]), psf_b.coords)
#a, f = tt.calc()
#F *= np.exp(-1.j*f)

for trial in np.arange(0, num_frames):
    tt = tip_tilt.tip_tilt(np.array([Ds[trial]]), np.array([Ps[trial]]), np.array([Fs[trial]]), psf_.coords1)
    a, f = tt.calc()
    tt_phase = np.exp(1.j*np.tensordot(psf_.coords1, a[0], axes=(2, 0)))

    P = Ps[trial, 0]
    P_d = Ps[trial, 1]
    P *= tt_phase
    P_d *= tt_phase
    
    #tt = tip_tilt.tip_tilt(np.array([Ds[trial]]), np.array([Ps[trial]]), np.array([Fs[trial]]), psf_.coords1)
    #a, f = tt.calc()
    #tt_phase = np.exp(-1.j*np.tensordot(psf_.coords1, a[0], axes=(2, 0)))
    #tt_phase = np.exp(-1.j*np.tensordot(psf_b.coords, a[trial], axes=(2, 0)))


    image_est_tt, F, P, P_d = psf_basis.deconvolve_(Ds[trial, 0], Ds[trial, 1], P, P_d, betass[trial], gamma, ret_all = True)
    my_plot.colormap(image_est_tt, [trial, 4])

    image_est_tt_norm = misc.normalize(image_est_tt)
    image_est_tt_mean += image_est_tt_norm

    my_plot.colormap(np.abs(image_est_tt_norm-image_norm), [trial, 6])

image_est_mean /= num_frames
image_est_tt_mean /= num_frames
D_mean /= num_frames
D_d_mean /= num_frames

my_plot.colormap(image_norm, [num_frames, 0])
my_plot.colormap(D_mean, [num_frames, 1])
my_plot.colormap(D_d_mean, [num_frames, 2])
my_plot.colormap(image_est_mean, [num_frames, 3])
my_plot.colormap(image_est_tt_mean, [num_frames, 4])
my_plot.colormap(np.abs(image_est_mean-image_norm), [num_frames, 5])
my_plot.colormap(np.abs(image_est_tt_mean-image_norm), [num_frames, 6])



my_plot.save("estimates.png")
my_plot.close()
