import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import numpy.random as random
import sys
sys.setrecursionlimit(10000)
sys.path.append('../utils')

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from matplotlib import cm
import os.path

from astropy.io import fits
import misc

def PIL2array(img):
    return np.array(img.getdata(), np.float32).reshape(img.size[1], img.size[0])


def gen_images(in_dir, out_dir, image_file, image_size, tile, scale, num_subimages, num_angles, ret=False):
    assert(in_dir != out_dir)
    images = []
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            print(file)
            if image_file is not None and file[:len(image_file)] != image_file:
                continue
            if file[-5:] == '.fits':
                hdul = fits.open(in_dir + "/" + file)
                image = hdul[0].data
                hdul.close()
            else:
                image = plt.imread(in_dir + "/" + file)
                #image = plt.imread(dir + "/" + file)
                if len(image.shape) == 3:
                    image = image[:, :, 0]
            if scale != 1.:
                image = misc.sample_image(image, scale)
    
            image /= np.max(image)
            image *= 255
            image_orig = np.array(image)
            nx_orig = image.shape[0]
            ny_orig = image.shape[1]
    
            for i in np.arange(num_subimages):
                x = random.randint(nx_orig-2*image_size)
                y = random.randint(ny_orig-2*image_size)
                image = image_orig[x:x+2*image_size, y:y+2*image_size]
        
                nx = image.shape[0]
                ny = image.shape[1]
                image = Image.fromarray(image)
                image = image.convert("L")
                
                angles = np.linspace(0, 360, num_angles)#random.random(size=num_angles)*360.0
                for angle in angles:
                    rotated = image.rotate(angle)
                    rotated = rotated.crop(((nx-image_size)//2, (ny-image_size)//2,(nx+image_size)//2, (ny+image_size)//2))
                    if out_dir is not None:
                        rotated.save(out_dir + "/" + file + "_" + str(i) + "_" + str(int(angle)) + ".png")
                    if ret:
                        images.append(PIL2array(rotated))
    if ret:
        return images

if __name__ == '__main__':
    in_dir = "images"
    out_dir = "images_out"
    image_file = None#"icont"
    image_size = 100
    tile=False
    scale=1.0
    
    num_subimages = 100
    num_angles = 10

    gen_images(in_dir, out_dir, image_file, image_size, tile, scale, num_subimages, num_angles)
