import os

import os.path
from astropy.io import fits
path = "output"

all_files = list()


for root, dirs, files in os.walk(path):
    for file in files:
        all_files.append(file)

for file in all_files:
    try:
        hdul = fits.open(path + "/" + file)
        if len(hdul) == 0:
            hdul.close()
            os.remove(path + "/" + file)
        hdul.close
    except Exception as e:
        print(e)
        pass
            