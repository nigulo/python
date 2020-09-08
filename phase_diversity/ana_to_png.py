import numpy as np
#from pyana import pyana as pa
import pyana as pa
import glob
import scipy.io as io
from tqdm import tqdm
import os
import sys
sys.path.append('../utils')
import misc
import utils
from PIL import Image


dir_name = "."
if len(sys.argv) > 1:
    dir_name = sys.argv[1]


def convert(path):
    max_value = 0
    min_value = 65535
    
    for root, dirs, files in os.walk(path):
        for file in files:
            f = pa.fzread(path + "/" + file)
            image = f['data']
            max_val = np.max(image)
            min_val = np.min(image)
            if (max_val > max_value):
                max_value = max_val
            if (min_val < min_value):
                min_value = min_val
                
    max_value -= min_value
                
    i = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            print(file)
            f = pa.fzread(path + "/" + file)
            image = f['data']
                
            #image = image.astype(np.uint32)
            #print(image.shape, image[110:115,110:115])
            image -= min_value
            scale = 65535//max_value
            image *= scale
            #image = image.astype("int32")
            im = Image.fromarray(image, mode='I;16')
            im = im.convert("I")
            #im.save(path + "/" + file + ".png")
            im.save(path + f"/image{i}.png")
            i += 1


if __name__ == '__main__':

    convert(dir_name)
