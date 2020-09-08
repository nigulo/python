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

def coords_of_pos(coords, positions, pos):
    #print("pos", pos)
    max_pos = np.max(positions, axis = 0)
    if pos[0] < 0 or pos[1] < 0:
        # extrapolate left coord
        coord0 = coords_of_pos(coords, positions, [0, 0])
        if max_pos[0] == 0:
            if max_pos[1] == 0: # We have only single patch
                coord1 = coord0 - [nx, nx] + [nx//10, nx//10]
            else:
                coord1 = np.array([coord0[0] - nx + nx//10, 2*coord0[1] - coords_of_pos(coords, positions, [0, 1])[1]])
        elif max_pos[1] == 0:
            coord1 = np.array([2*coord0[0] - coords_of_pos(coords, positions, [1, 0])[0], coord0[1] - nx + nx//10])
        else:                
            coord1 = 2*coord0 - coords_of_pos(coords, positions, [1, 1])
        if pos[0] < 0:
            if pos[1] < 0:
                return coord1
            else:
                coord0 = coords_of_pos(coords, positions, [0, pos[1]])
                return np.array([coord1[0], coord0[1]])
        else:
            coord0 = coords_of_pos(coords, positions, [pos[0], 0])
            return np.array([coord0[0], coord1[1]])
    #print("max_pos", max_pos, positions)
    if pos[0] > max_pos[0] or pos[1] > max_pos[1]:
        # extrapolate left coord
        coord0 = coords_of_pos(coords, positions, max_pos)
        if max_pos[0] == 0:
            if max_pos[1] == 0: # We have only single patch
                coord1 = coord0 - [nx, nx] + [nx//10, nx//10]
            else:
                coord1 = np.array([coord0[0] - nx + nx//10, 2*coord0[1] - coords_of_pos(coords, positions, max_pos - [0, 1])[1]])
        elif max_pos[1] == 0:
            coord1 = np.array([2*coord0[0] - coords_of_pos(coords, positions, max_pos - [1, 0])[0], coord0[1] - nx + nx//10])
        else:
            coord1 = 2*coord0 - coords_of_pos(coords, positions, max_pos - [1, 1])
        if pos[0] > max_pos[0]:
            if pos[1] > max_pos[1]:
                return coord1
            else:
                coord0 = coords_of_pos(coords, positions, [max_pos[0], pos[1]])
                return np.array([coord1[0], coord0[1]])
        else:
            coord0 = coords_of_pos(coords, positions, [pos[0], max_pos[1]])
            return np.array([coord0[0], coord1[1]])
    filtr = np.all(positions == pos, axis=1)
    #print("pos, filtr", pos, filtr)
    return coords[filtr][0]

def crop(nx, i, coords, positions):
    coord = coords[i]
    pos = positions[i]
    top_left_coord = coords_of_pos(coords, positions, pos - [1, 1]) + [nx, nx]
    bottom_right_coord = coords_of_pos(coords, positions, pos + [1, 1])
    print("top_left_coord, bottom_right_coord", top_left_coord, bottom_right_coord)
    
    top_left_coord  = (top_left_coord + coord)//2
    bottom_right_coord = (bottom_right_coord + coord + [nx, nx])//2
    top_left_delta = top_left_coord - coord 
    bottom_right_delta = bottom_right_coord - coord - [nx, nx]

    print("pos, coord, i", pos, coord, i)
    return top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta


def convert(path):
    #max_value = 0
    #for root, dirs, files in os.walk(path):
    #    for file in files:
    #        if file[-4:] == '.sav'
    #            f = pa.fzread(path + "/" + file)
    #            image = f['data']
    #            max_val = np.max(image)
    #            if (max_val > max_value):
    #                max_value = max_val
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file[-4:] == '.sav':
                tmp = io.readsav(path + "/" + file)
                objs = tmp['img'] # MOMBD Restored objects
                
                # Get x and y positions of each one of the MOMFBD patches
                xl = tmp['yl'][:,0] - 1
                yl = tmp['xl'][0,:] - 1
            
                # Num patches
                num_x = len(xl)
                num_y = len(yl)
                assert(num_x == objs.shape[0])
                assert(num_y == objs.shape[1])
                
                nx = objs[0, 0].shape[0]
                assert(nx == objs[0, 0].shape[1])
                
                num_objs = num_x*num_y
                
                positions = np.empty((num_objs, 2), dtype="int")
                coords = np.empty((num_objs, 2), dtype="int")
                
                indx = np.repeat(np.arange(num_x, dtype="int"), num_y)
                indy = np.tile(np.arange(num_y, dtype="int"), num_x)
                
                obj_index = 0
                for i in range(num_x):
                    for j in range(num_y):
                        positions[obj_index, 0] = i
                        positions[obj_index, 1] = j

                        x0 = xl[indx[obj_index]]
                        y0 = yl[indy[obj_index]]
                        coords[obj_index, 0] = x0
                        coords[obj_index, 1] = y0

                        obj_index += 1

                cropped_objs = []
                cropped_coords = []
                full_shape = np.zeros(2, dtype="int")
        
                obj_index = 0
                for i in range(num_x):
                    for j in range(num_y):
                        obj = objs[i, j]
                        top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta = crop(nx, obj_index, coords, positions)
                        print("Crop:", top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta)
                        cropped_obj = obj[top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]]
                        cropped_objs.append(cropped_obj)
                        cropped_coords.append(top_left_coord)
                        full_shape += cropped_obj.shape
                        obj_index += 1
                
                
                
                max_pos = np.max(positions, axis = 0)
                min_coord = np.min(cropped_coords, axis = 0)
                full_shape[0] = full_shape[0] // (max_pos[1] + 1)
                full_shape[1] = full_shape[1] // (max_pos[0] + 1)
                print("full_shape", full_shape)
                full_obj = np.zeros(full_shape, dtype=np.int16)
                
                for i in np.arange(len(cropped_objs)):
                    x = cropped_coords[i][0]-min_coord[0]
                    y = cropped_coords[i][1]-min_coord[1]
                    s = cropped_objs[i].shape
                    print(x, y, s)
                    full_obj[x:x+s[0],y:y+s[1]] = cropped_objs[i]
                
                min_value = np.min(full_obj)
                full_obj -= min_value
                max_value = np.max(full_obj)
                scale = 65535//max_value
                full_obj *= scale
                im = Image.fromarray(full_obj, mode='I;16')
                im = im.convert("I")
                im.save(path + "/" + file + ".png")


if __name__ == '__main__':

    convert(dir_name)
