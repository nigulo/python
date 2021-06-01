import matplotlib as mpl
mpl.use('nbAgg')
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
import plot
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.optimize
from geomdl import fitting
import pickle

def load(f):
    if not os.path.exists(f):
        return None
    return pickle.load(open(f, 'rb'))

def save(obj, f):
    with open(f, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def calc_mode_radius(mode_index, params):
    r_mean = 0
    nu_mean = 0
    for nu, k in params:
        nu_mean += nu
        r_mean += k
    nu_mean /= len(params)
    r_mean /= len(params)
    print(f"Mode {mode_index}: ", nu_mean, r_mean)
    return nu_mean, r_mean

    
    
# F-mode = 0
# P modes = 1 ... 
def get_mode_prior(mode_index, k):
    for row in mode_priors:
        k_start = row[0]
        k_end = row[1]
        if k >= k_start and k < k_end:
            if mode_index >= row[2]:
                raise Exception("No mode prior found")
            return row[5 + mode_index * 3]

def get_num_components(k):
    for row in mode_priors:
        k_start = row[0]
        k_end = row[1]
        if k >= k_start and k < k_end:
            return int(row[2])
    return 0
    

def get_noise_var(data, ks, nus):
    k_indices = np.where(np.logical_and(ks >= 3500, ks <= 4500))[0]
    nu_indices = np.where(np.logical_and(nus >= 2, nus <= 4))[0]
    d = data[nu_indices]
    d = d[:, k_indices, :]
    d = d[:, :, k_indices]
    return np.var(d)

if (__name__ == '__main__'):

    argv = sys.argv
    i = 1
    
    if len(argv) <= 1:
        print("Usage: python fmode_priors.py input_path [year] [input_file]")
        sys.exit(1)
    year = ""
    input_file = None
    
    if len(argv) > i:
        input_path = argv[i]
    i += 1
    if len(argv) > i:
        year = argv[i]
    i += 1
    if len(argv) > i:
        input_file = argv[i]
        
    if len(year) > 0:
        input_path = os.path.join(input_path, year)
    
    MODE_VERTICAL = 0
    MODE_PERP_TO_RIDGE = 1
    mode = MODE_PERP_TO_RIDGE
    
    ny = 300
    map_scale = 0.05
    #k = np.arange(ny)
    #k = (k-((ny-1)/2))*(2.*180./(ny*map_scale))
    #k0 = np.where(k == np.min(np.abs(k)))[0][0]
    #k1 = np.where(k == np.max(np.abs(k)))[0][0]
    #k = k[k0:k1+1]
    
    k_min = 700
    k_max = 3000#sys.maxsize
    #if len(sys.argv) > 1:
    #    k_min = float(sys.argv[1])
    
    #if len(sys.argv) > 2:
    #    k_max = float(sys.argv[2])
    
    num_w = 0
    alpha_pol_degree = 2
    beta_pol_degree = 2
    w_pol_degree = 1
    scale_smooth_win = 11
    
    num_optimizations = 1
    approx_width = 100
    
    mode_priors = np.genfromtxt("GVconst.txt")

    if not os.path.exists("results"):
        os.mkdir("results")
        
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file[-5:] != ".fits":
                continue
            if input_file is not None and len(input_file) > 0 and file != input_file:
                continue
            file_prefix = file[:-5]
            
            output_dir = os.path.join("results", file_prefix)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
                
            print("==================================================================")
            print(file_prefix)
            print("==================================================================")
        
            hdul = fits.open(os.path.join(root, file))
            
            data = hdul[1].data
            #print(data.shape, nf, ny)
            #assert(data.shape[0] == nf)
            #assert(data.shape[1] == ny)
            #assert(data.shape[2] == ny)
            data = fft.fftn(data)/np.product(data.shape)
            #data = data[:data.shape[0]//2, :, :]
            data = np.real(data*np.conj(data))
            data = fft.fftshift(data)
            
            ks = np.linspace(-3600, 3600, data.shape[1])
            nus = np.linspace(-11.076389, 11.076389, data.shape[0])
                        
            hdul.close()

            k_max_ = ks[ks < k_max][-1]
            
            print("k_max", k_max_)
             
            
            print(data.shape, nus.shape)
            nus_filtered = np.array(nus)
            fltr = (nus_filtered > 2.) * (nus_filtered < 10.)
            nus_filtered = nus_filtered[fltr]
            data = data[fltr, :, :]
            #fltr = ks >= 0
            #ks_filtered = ks[fltr]
            #data = data[:, fltr, :]
            #data = data[:, :, fltr]
            
                        
            mode_params = load("mode_params.dat")
            if mode_params is None:

                mode_params = [[]]*len(nus_filtered)
                
                for nu_ind in range(0, len(nus_filtered)):
                    for k_ind1 in range(0, len(ks)):
                        for k_ind2 in range(0, len(ks)):
                            k1, k2 = np.abs(ks[k_ind1]), np.abs(ks[k_ind2])
                            k = np.sqrt(k1**2 + k2**2)
                            if k >= k_min and k <= k_max_:
                                num_components = get_num_components(k)

                                for mode_index in range(num_components):
                                    if mode_index > 0:
                                        break
                                    nu_prior = get_mode_prior(mode_index, k)
                                    nu_index = np.argmin(np.abs(np.abs(nus_filtered) - nu_prior))
                                    if nu_index == nu_ind:
                                        print(nu_prior, k)
                                        if len(mode_params[nu_ind]) == 0:
                                            mode_params[nu_ind] = [[]]*5
                                            mode_params[nu_ind][mode_index] = []
    
                                        mode_params[nu_ind][mode_index].append([nu_prior, k])
                save(mode_params, "mode_params.dat")                        
                                

    
            mode_stats = dict()
            colors = ["k", "b", "g", "r", "m"]
            
            '''
            for nu_ind in range(0, len(mode_params)):
                contains_data = False
                for mode_index in range(5):
                    if len(mode_params[nu_ind]) > mode_index:
                        contains_data = True
                if not contains_data:
                    continue
                for mode_index in range(5):
                    if mode_index not in mode_stats:
                        mode_stats[mode_index] = []
                    if len(mode_params[nu_ind][mode_index]) > 0:
                        nu_mean, r_mean = calc_mode_radius(mode_index, mode_params[nu_ind][mode_index])
                        mode_stats[mode_index].append([nu_mean, r_mean])
            
            
            fitted = dict()
            for mode_index in mode_stats.keys():
                deg = 1
                #if mode_index < 4: 
                #    deg = 2
                #else: 
                #    deg = 1
                ms = np.asarray(mode_stats[mode_index])
                coefs = np.polyfit(ms[:, 0], ms[:, 1], deg=deg)
                print(ms[:, 0], ms[:, 1], coefs)
                
                powers = np.arange(deg+1)[::-1]
                powers = np.reshape(np.repeat(powers, len(nus_filtered)), (len(powers), len(nus_filtered)))
                ws = np.reshape(np.repeat(coefs, len(nus_filtered)), (len(powers), len(nus_filtered)))
                rs = np.sum(ws*nus_filtered**powers, axis=0)
                fitted[mode_index] = rs
            save((fitted, nus_filtered), "ring_radii.dat")
            
            for i in range(0, len(nus_filtered), 10):
                fig = plot.plot(nrows=1, ncols=1, size=plot.default_size(data.shape[1]//2, data.shape[2]//2))
                fig.contour(ks, ks, data[i, :, :], levels=100)
                fig.set_axis_title(r"$\nu=" + str(nus_filtered[i]) + "$")
                for mode_index in fitted.keys():
                    r = fitted[mode_index][i]
                    if r > 0:
                        print(i, mode_index, r)
                        phi = np.linspace(0, 2*np.pi, 100)
                        x = np.cos(phi)*r
                        y = np.sin(phi)*r
                        fig.plot(x, y, f"{colors[mode_index]}.", lw=0.1, ms=0.1)
                fig.save(os.path.join(output_dir, f"fitted_rings{i}.png"))
            '''

            for nu_ind in range(0, len(mode_params)):
                contains_data = False
                for mode_index in range(5):
                    if len(mode_params[nu_ind]) > mode_index:
                        contains_data = True
                if not contains_data:
                    continue
                for mode_index in range(5):
                    if mode_index not in mode_stats:
                        mode_stats[mode_index] = [0.0]*len(nus_filtered)
                    if len(mode_params[nu_ind][mode_index]) > 0:
                        nu_mean, r_mean = calc_mode_radius(mode_index, mode_params[nu_ind][mode_index])
                        nu_index = np.argmin(np.abs(np.abs(nus_filtered) - nu_mean))
                        mode_stats[mode_index][nu_index] = r_mean

            for mode_index in mode_stats.keys():
                for nu_ind in range(len(mode_stats[mode_index])):
                    if mode_stats[mode_index][nu_ind] == 0:
                        #lower_r = 0
                        #upper_r = 0
                        for nu_ind2 in range(nu_ind-1, -1, -1):
                            if mode_stats[mode_index][nu_ind2] > 0:
                                lower_r = mode_stats[mode_index][nu_ind2]
                                mode_stats[mode_index][nu_ind] = lower_r
                                break
                        for nu_ind2 in range(nu_ind+1, len(mode_stats[mode_index])):
                            if mode_stats[mode_index][nu_ind2] > 0:
                                upper_r = mode_stats[mode_index][nu_ind2]
                                mode_stats[mode_index][nu_ind] += upper_r
                                mode_stats[mode_index][nu_ind] /= 2
                                #print(upper_r)
                                break
                        #print(nu_ind, lower_r, upper_r)
                        #mode_stats[mode_index][nu_ind] = (lower_r + upper_r)/2

            for mode_index in mode_stats.keys():
                for nu_ind in range(len(mode_stats[mode_index])):
                    print(mode_stats[mode_index][nu_ind])
                

            save((mode_stats, nus_filtered, fltr), "ring_radii.dat")

            for i in range(0, len(nus_filtered), 10):
                fig = plot.plot(nrows=1, ncols=1, size=plot.default_size(data.shape[1]//2, data.shape[2]//2))
                fig.contour(ks, ks, data[i, :, :], levels=100)
                fig.set_axis_title(r"$\nu=" + str(nus_filtered[i]) + "$")
                for mode_index in mode_stats.keys():
                    r = mode_stats[mode_index][i]
                    if r > 0:
                        print(i, mode_index, r)
                        phi = np.linspace(0, 2*np.pi, 100)
                        x = np.cos(phi)*r
                        y = np.sin(phi)*r
                        fig.plot(x, y, f"{colors[mode_index]}.", lw=0.1, ms=0.1)
                fig.save(os.path.join(output_dir, f"fitted_rings{i}.png"))
