import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import scipy.special as special
import scipy.signal as signal
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import cm
import utils

import matplotlib.pyplot as plt


image = plt.imread('granulation.png')
    
print(image)
print(np.shape(image))


