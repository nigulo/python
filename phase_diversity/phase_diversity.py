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

num_frames = 10
max_frames = min(10, num_frames)
aberration_mode = "psf"

#image = plt.imread('granulation31x33arsec.png')
##image = misc.sample_image(image,.27)
#image = misc.sample_image(image,.675)

image = plt.imread('granulation.png')[:, :, 0]
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

image_orig = np.array(image)

state = load(state_file)


def center_and_normalize(ds):
    m = np.mean(ds)
    std = np.std(ds)
    return (ds - m)/std, m, std

def get_params(nx):
    #coef1 = 4.**(-np.log2(float(nx)/11))
    #coef2 = 2.**(-np.log2(float(nx)/11))
    #print("coef1, coef2", coef1, coef2)
    #arcsec_per_px = coef1*0.2
    arcsec_per_px = .25*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    print("arcsec_per_px=", arcsec_per_px)
    #defocus = (0., 4.967349723461641)#10.#10.#10.#7.5
    defocus = (2.*np.pi*100, 4.967349723461641)#10.#10.#10.#7.5
    #defocus = (6.28, 46.)#10.#10.#10.#7.5
    return (arcsec_per_px, defocus)


def calibrate(arcsec_per_px, nx):
    coef = np.log2(float(nx)/11)
    return 2.6*arcsec_per_px*2.**coef


if state == None:
    print("Creating new state")
    jmax = 20
    #arcsec_per_px = 0.057
    #arcsec_per_px = 0.011
    diameter = 50.0
    wavelength = 5250.0
    gamma = 1.0
    nx = np.shape(image)[0]

    arcsec_per_px, defocus1 = get_params(nx_orig)#wavelength/diameter*1e-8*180/np.pi*3600
    (defocus_psf, defocus_psf_b) = defocus1
    #arcsec_per_px1=wavelength/diameter*1e-8*180/np.pi*3600/4.58

    coords, _, _ = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
    #x_max = .5
    #x_min = -.5
    #delta = 0.
    #if (nx % 2) == 0:
    #    delta = (x_max - x_min)/nx
    #xs = np.linspace(x_min, x_max-delta, nx)
    #coords = np.dstack(np.meshgrid(xs, xs))
    
    tt = tip_tilt.tip_tilt(coords, prior_prec=((np.max(coords[0])-np.min(coords[0]))/2)**2)
    psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus_psf_b, tip_tilt = tt, prior_prec=np.concatenate(([0., 0.], np.linspace(0., 1, jmax-2)**2)))
    #psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus_psf_b, tip_tilt = tt)
    psf_b.create_basis()

    save(state_file, [jmax, arcsec_per_px, diameter, wavelength, defocus1, gamma, nx, psf_b.get_state()])
else:
    print("Using saved state")
    jmax = state[0]
    arcsec_per_px = state[1]
    diameter = state[2]
    wavelength = state[3]
    defocus1 = state[4]
    (defocus_psf, defocus_psf_b) = defocus1
    gamma = state[5]
    nx = state[6]
    #arcsec_per_px1=wavelength/diameter*1e-8*180/np.pi*3600/4.58
    print("jmax, arcsec_per_px, diameter, wavelength, defocus, gamma, nx", jmax, arcsec_per_px, diameter, wavelength, defocus1, gamma, nx)
    
    assert(nx == np.shape(image)[0])
    
    coords, _, _ = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
    #tt = tip_tilt.tip_tilt(coords, prior_prec=0.)
    tt = tip_tilt.tip_tilt(coords, prior_prec=((np.max(coords[0])-np.min(coords[0]))/2)**2)
    #psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus_psf_b, tip_tilt=tt)
    psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus_psf_b, tip_tilt=tt, prior_prec=np.concatenate(([0., 0.], np.linspace(0., 1, jmax-2)**2)))
    psf_b.set_state(state[7])
    

fimage = fft.fft2(image)
fimage = fft.fftshift(fimage)


#aperture_func = lambda xs: utils.aperture_circ(xs, diameter, 15.0)
aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
#defocus_func = lambda xs: 2.*np.pi*np.sum(xs*xs, axis=2)#10.*(2*np.sum(xs*xs, axis=2) - 1.)
#defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)
defocus_func = lambda xs: defocus_psf*np.sum(xs*xs, axis=2)

my_plot = plot.plot(nrows=max_frames + 1, ncols=5)
my_plot.set_axis()

image_est_mean = np.zeros((nx, nx))
D_mean = np.zeros((nx, nx))
D_d_mean = np.zeros((nx, nx))
        
#image_norm = misc.normalize(image)
wavefront = kolmogorov.kolmogorov(fried = np.array([.3]), num_realizations=num_frames, size=4*nx_orig, sampling=1.)

sampler = psf_basis_sampler.psf_basis_sampler(psf_b, gamma, num_samples=1)

betass = []

Ds = np.zeros((num_frames, 2, nx, nx), dtype='complex')
Ps = np.ones((num_frames, 2, nx, nx), dtype='complex')
Fs = np.ones((num_frames, 1, nx, nx), dtype='complex')
#true_alphas = np.zeros((num_frames, 2))
#true_Ps = np.ones((num_frames, 2, nx, nx), dtype='complex')

pa_null = psf.phase_aberration([])
ctf_null = psf.coh_trans_func(aperture_func, pa_null, defocus_func)
psf_null = psf.psf(ctf_null, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
psf_null.calc(defocus=False)
psf_null.calc(defocus=True)

image_center = image#misc.center(image)
vmin = np.min(image_center)
vmax = np.max(image_center)
###############################################################################
# Create abberrated images
###############################################################################
if aberration_mode == "psf_basis":
    jmax_temp = 10
    #psf_b_temp = psf_basis.psf_basis(jmax = jmax_temp, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus_psf_b, tip_tilt = None)
    psf_b_temp = psf_basis.psf_basis(jmax = jmax_temp, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus_psf_b, tip_tilt = None)
    psf_b_temp.set_state(state[7])
    #psf_b_temp.create_basis()


for trial in np.arange(0, num_frames):
    
    if aberration_mode == "psf":
        #pa = psf.phase_aberration(np.minimum(np.maximum(np.random.normal(size=2)*50, -50), 50), start_index=1)
        #pa = psf.phase_aberration(np.random.normal(size=jmax))
        #pa = psf.phase_aberration([])
        #ctf = psf.coh_trans_func(aperture_func, pa, defocus_func)
        ctf = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[0,trial,:,:]), defocus_func)
        psf_ = psf.psf(ctf, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
        DF, DF_d = psf_.multiply(fimage)

        DF = fft.ifftshift(DF)
        DF_d = fft.ifftshift(DF_d)

        D = fft.ifft2(DF).real
        D_d = fft.ifft2(DF_d).real
        
        #D, D_d = psf_.convolve(image)
        #DF = fft.fft2(D)
        #DF_d = fft.fft2(D_d)
        
    else:

        betas = np.zeros(jmax, dtype = 'complex')
        #betas = np.random.normal(size=jmax_temp) + 1.j*np.random.normal(size=jmax_temp)
        #betas*=1e-10
        #betas[::2] = 1.j*betas[::2]
        
        Ds1 = psf_b_temp.convolve(image, betas)
        D = Ds1[0, 0]
        D_d = Ds1[0, 1]

        DF = fft.fft2(D)
        DF_d = fft.fft2(D_d)
    
    
    Ds[trial, 0] = DF
    Ds[trial, 1] = DF_d
    Fs[trial, 0] = np.absolute(DF)
    #np.testing.assert_array_almost_equal(np.absolute(Ds[0, 0]), np.absolute(D))
    #Ps[trial, 0] = P
    #Ps[trial, 1] = P_d
    #Fs[trial, 0] = F

    

    print("np.min(image), np.max(image), np.min(D), np.max(D)", np.min(image_center), np.max(image_center), np.min(D), np.max(D))
    if trial < max_frames:
        my_plot.colormap(image_center, [trial, 0], vmin=vmin, vmax=vmax)
        my_plot.colormap(D, [trial, 1], vmin=vmin, vmax=vmax)
        my_plot.colormap(D_d, [trial, 2], vmin=vmin, vmax=vmax)

    D_mean += D
    D_d_mean += D_d
my_plot.save("estimates.png")


###############################################################################
# Estimate PSF
###############################################################################

#Ds, mean, std = center_and_normalize(Ds)

res = sampler.sample(Ds, "samples" + str(trial) + ".png")
if tt is not None:
    betas_est, a_est = res
else:
    betas_est = res
    a_est = None
#print("betas_est, a_est", betas_est, a_est)
image_est, F, Ps = psf_b.deconvolve(Ds, betas_est, gamma, ret_all = True, a_est=a_est, normalize=True)
#tt.set_data(Ds, Ps)#, F):
#image_est, F, Ps = psf_b.deconvolve(Ds, np.ones((num_frames, jmax), dtype='complex'), gamma, ret_all = True, a_est=np.zeros((num_frames+1, 2)), normalize=True)

#Ps = np.ones((num_frames, 2, nx, nx), dtype='complex')
#tt = tip_tilt.tip_tilt(coords, prior_prec=((np.max(coords[0])-np.min(coords[0]))/2)**2, num_rounds=1)
#tt.set_data(Ds, Ps)#, F)
#image_est, _, _ = tt.calc()
#image_est, _, _ = tt.deconvolve(Ds[:,0,:,:], Ps, a_est)

image_est = fft.ifftshift(image_est, axes=(-2, -1))
for trial in np.arange(0, num_frames):
    #image_est_norm = misc.normalize(image_est[trial])
    image_est_mean += image_est[trial]
    if trial < max_frames:
        my_plot.colormap(image_est[trial], [trial, 3], vmin=vmin, vmax=vmax)
        my_plot.colormap(np.abs(image_est[trial]-image_center), [trial, 4], vmin=vmin, vmax=vmax)

image_est_mean /= num_frames
D_mean /= num_frames
D_d_mean /= num_frames

my_plot.colormap(image_center, [max_frames, 0], vmin=vmin, vmax=vmax)
my_plot.colormap(D_mean, [max_frames, 1], vmin=vmin, vmax=vmax)
my_plot.colormap(D_d_mean, [max_frames, 2], vmin=vmin, vmax=vmax)
my_plot.colormap(image_est_mean, [max_frames, 3], vmin=vmin, vmax=vmax)
my_plot.colormap(np.abs(image_est_mean-image_center), [max_frames, 4], vmin=vmin, vmax=vmax)

my_plot.save("estimates.png")
my_plot.close()
