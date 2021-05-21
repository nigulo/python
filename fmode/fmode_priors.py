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
from fmode import downsample

def load(f):
    if not os.path.exists(f):
        return None
    return pickle.load(open(f, 'rb'))

def save(obj, f):
    with open(f, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def calc_mode_radius(mode_index, params, nu_k_scale, func_type="lorenzian"):
    if func_type == "lorenzian":
        #alphas = []
        #betas = []
        r_mean = 0
        alpha0_mean = 0
        for alpha0, alpha1, alpha2, beta, scale in params:
            alpha0_mean += alpha0
            r_mean += (alpha1**2 + alpha2**2)/nu_k_scale**2
            #alphas.append(alpha)
            #betas.append(beta)
        alpha0_mean /= len(params)
        r_mean = np.sqrt(r_mean/len(params))
        print(f"Mode {mode_index}: ", alpha0_mean, r_mean)
        #alphas = np.asarray(alphas)
        #betas = np.asarray(betas)
        #indices = np.argsort
        return alpha0_mean, r_mean
    else:
        raise ValueError(f"func_type {func_type} not supported")
    

def get_num_params(func_type="lorenzian"):
    if func_type == "lorenzian":
        return 5
    else:
        raise ValueError(f"func_type {func_type} not supported")
    
    
# F-mode = 0
# P modes = 1 ... 
def get_alpha_prior(mode_index, k):
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
    k_max = 1500#sys.maxsize
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
            data = downsample(data)
            
            ks = np.linspace(-3600, 3600, data.shape[1])
            nus = np.linspace(-11.076389, 11.076389, data.shape[0])
                        
            hdul.close()

            k_max_ = ks[ks < k_max][-1]
                        
            print(data.shape, nus.shape)
            nus_filtered = np.asarray(nus)
            fltr = (nus_filtered > 2.) * (nus_filtered < 10.)
            nus_filtered = nus_filtered[fltr]
            data = data[fltr, :, :]
            fltr = ks >= 0
            ks_filtered = ks[fltr]
            data = data[:, fltr, :]
            data = data[:, :, fltr]
            
            
            data_mask = np.zeros_like(data, dtype=int)
            
            
            nu_k_scale = (nus[-1]-nus[0])/(ks[-1]-ks[0])
            coords = load("coords.dat")
            if coords is None:
                coords = np.empty((len(nus_filtered), len(ks_filtered), len(ks_filtered), 4))
                k_grid = np.transpose([np.repeat(ks_filtered, len(ks_filtered)), np.tile(ks_filtered, len(ks_filtered))])
                k_grid = np.reshape(k_grid, (data.shape[1], data.shape[2], 2))
                k_grid_scaled = k_grid*nu_k_scale
                k_grid2 = k_grid**2
                k_modulus = np.sqrt(np.sum(k_grid2, axis=2, keepdims=True))
                print(k_grid[:5, :5], k_grid[-5:, -5:])
                for i in range(len(nus_filtered)):
                    nu = nus_filtered[i]
                    nus_ = np.reshape(np.repeat([nu], k_grid.shape[0]*k_grid.shape[1]), (k_grid.shape[0], k_grid.shape[1], 1))
                    #nu2 = nu**2
                    #print(nus_.shape, k_grid_scaled.shape, k_modulus.shape)
                    coords[i] = np.concatenate([nus_, k_grid_scaled, k_modulus], axis=2)
                save(coords, "coords.dat")
    
            print("Coordinate grid created")
            
            '''
            phi_ks = np.linspace(0, 2*np.p-, 100, endpoint=False)
            for phi_k_i in range(len(phi_ks)):
                phi_k = phi_ks[phi_k_i]
                if phi_k_i == 0:
                    phi_k1 = 2*np.pi-phi_ks[-1]
                else:
                    phi_k1 = phi_ks[phi_k_i-1]
                phi_k2 = phi_ks[phi_k_i]
                dist = np.minimum(np.abs(coords[:, :, :, 5] - phi_k), np.abs(coords[:, :, :, 5] - np.pi - phi_k)
                dist1 = np.minimum(np.abs(coords[:, :, :, 5] - phi_k1), np.abs(coords[:, :, :, 5] - np.pi - phi_k1))
                dist2 = np.minimum(np.abs(coords[:, :, :, 5] - phi_k2), np.abs(coords[:, :, :, 5] - np.pi - phi_k2))
                fltr = (dist < dist1) * (dist < dist2))
                
                coords_slice = coords[fltr]
                data_slice = data[fltr]
            '''
            
            priors = load("priors2.dat")
            if priors is not None:
                params, bounds, mode_info, mode_params = priors
            else:
                params = []
                bounds = []
                mode_info = dict()
                mode_params = dict()
                #sampling_step = [coords[3, 0, 0, 0] - coords[0, 0, 0, 0], coords[0, 3, 0, 1] - coords[0, 0, 0, 1], coords[0, 0, 3, 2] - coords[0, 0, 0, 2]]
                #print(coords[0, 3, 0, 1], coords[0, 0, 0, 1])
                sampling_step = (coords[10, 0, 0, 0] - coords[0, 0, 0, 0])**2
                print("sampling_step", sampling_step)
                
                nus_ = coords[:, 0, 0, 0]
                for nu_ind in range(0, coords.shape[0]):
                    for k_ind1 in range(0, coords.shape[1]):
                        for k_ind2 in range(0, coords.shape[2]):
                            _, k1, k2, k = coords[nu_ind, k_ind1, k_ind2]
                            if k >= k_min and k <= k_max_:
                                data_mask[:, k_ind1, k_ind2] = 1
                                num_components = get_num_components(k)
                                
                                for i in range(num_components):
                                    alpha_prior0 = get_alpha_prior(i, k)
                                    nu_index = np.argmin(np.abs(nus_ - alpha_prior0))
                                    if nu_index == nu_ind:
                                        if i not in mode_info:
                                            mode_info[i] = []
                                            mode_params[i] = []
                                        mp = np.asarray(mode_params[i])
                                        if len(mp) > 0:
                                            dists = np.sum((mp - [alpha_prior0, k1, k2])**2, axis = 1)
                                            min_dist = np.sum((mp[np.argmin(dists)] - [alpha_prior0, k1, k2])**2)
                                        else:
                                            min_dist = sampling_step
                                        if min_dist >= sampling_step:
                                            mode_info[i].append(len(params))
                                            mode_params[i].append([alpha_prior0, k1, k2])
                                        
                                            params.append(alpha_prior0)
                                            bounds.append((alpha_prior0-.5 , alpha_prior0+.5))
                    
                                            params.append(k1)
                                            bounds.append((k1-.5 , k1+.5))
                    
                                            params.append(k2)
                                            bounds.append((k2-.5 , k2+.5))
                    
                                            beta_prior = 0.04#.2/num_components
                                            params.append(beta_prior)
                                            #params.append(1./100)
                                            bounds.append((1e-10 , 2*beta_prior))
                    
                                            scale_prior = 1.
                                            params.append(scale_prior)
                                            bounds.append((1e-10, 10.))
            
                for i in mode_info.keys():
                    mode_info[i] = np.asarray(mode_info[i])
                save((params, bounds, mode_info, mode_params), "priors2.dat")
            print("Priors set")
            #######################################################################    
            print("params", len(params))
            #y = y[:, k_indices]
            #k = k[k_indices]
                    

            ds_factor = 5
            num_params = get_num_params()
            params_ = []
            for i in range(data.shape[0]//ds_factor):
                mode_params_ = []
                for _ in range(5):
                    mode_params_.append([])
                params_.append(mode_params_)
            for mode_index in mode_params.keys():
                for i in range(len(mode_params[mode_index])):
                    alpha_prior0, k1, k2 = mode_params[mode_index][i]
                    start_index = mode_info[mode_index][i]
                    nu_ind = np.argmin(np.abs(nus_filtered-alpha_prior0))
                    #print(nu_ind, mode_index, len(params_est[i*num_params:i*num_params+num_params]))
                    params_[nu_ind//ds_factor][mode_index].append(params[start_index:start_index+num_params])
    
            mode_stats = dict()
            colors = ["k", "b", "g", "r", "m"]
            for i in range(0, len(params_)):
                contains_data = False
                for mode_index in range(5):
                    if len(params_[i][mode_index]) > 0:
                        contains_data = True
                if not contains_data:
                    continue
                for mode_index in range(5):
                    if mode_index not in mode_stats:
                        mode_stats[mode_index] = []
                    if len(params_[i][mode_index]) > 0:
                        nu_mean, r_mean = calc_mode_radius(mode_index, params_[i][mode_index], nu_k_scale)
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
            save(fitted, "ring_radii.dat")
            
            for i in range(len(nus_filtered)):
                fig = plot.plot(nrows=1, ncols=1, size=plot.default_size(data.shape[1], data.shape[2]))
                fig.contour(ks_filtered, ks_filtered, data[i, :, :])
                fig.set_axis_title(r"$\nu=" + str(coords[i, 0, 0, 0]) + "$")
                for mode_index in fitted.keys():
                    r = fitted[mode_index][i]
                    if r > 0:
                        print(i, mode_index, r)
                        phi = np.linspace(0, np.pi/2, 100)
                        x = np.cos(phi)*r
                        y = np.sin(phi)*r
                        fig.plot(x, y, f"{colors[mode_index]}.", lw=0.1, ms=0.1)
                fig.save(os.path.join(output_dir, f"fitted_rings{i}.png"))
                    

