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
import pickle
from multiprocessing import Process, Queue
import misc

num_proc = 128
nu_sampling = 4
k_sampling = 5
num_phi_interp = 20
num_nu_interp = 2
chunk_size = 15

percentile = .95

supported_funcs = ["lorenzian", "lorenzian+gaussian"]

def load(f):
    if not os.path.exists(f):
        return None
    return pickle.load(open(f, 'rb'))

def save(obj, f):
    with open(f, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
        
def f(d, q):
    (alphas1, betas1, scales1, x) = d
    r2 = np.sum((x-alphas1)**2, axis=3)
    q.put(scales1/(np.pi*betas1*(1+(np.sqrt(r2)/betas1)**2)))

def downsample(data, factors=(3, 2)):
    print(np.min(data), np.max(data), np.mean(data))
    data = fft.fftshift(fft.fftn(data))

    '''
    metric = 1./np.array([data.shape[0]/2/factors[0], data.shape[1]/2/factors[1], data.shape[2]/2/factors[1]])
    center = np.array([data.shape[0]/2, data.shape[1]/2, data.shape[2]/2])
    grid = misc.meshgrid(np.arange(data.shape[2]), np.arange(data.shape[1]), np.arange(data.shape[0]))
    grid = grid.astype(np.float64)
    grid -= center
    grid *= metric
    grid = np.sum(grid**2, axis=3)
    data[grid > 1] = 0
    '''
                
    nu_ind = int(round(data.shape[0]*(1-1/factors[0])*.5))
    kx_ind = int(round(data.shape[1]*(1-1/factors[1])*.5))
    ky_ind = int(round(data.shape[2]*(1-1/factors[1])*.5))
    data = data[nu_ind:-nu_ind, kx_ind:-kx_ind, ky_ind:-ky_ind]

    data = fft.ifftshift(data)
    data = fft.ifftn(data)
    print(np.mean(np.abs(np.real(data))))
    print(np.mean(np.abs(np.imag(data))))
    data = np.real(data)
    data[data <= 0] = 1e-13
    print(data.shape, np.min(data), np.max(data), np.mean(data))
    return data
    
    
def basis_func(coords, params, mode_params, func_type="lorenzian"):
    if func_type not in supported_funcs:
        raise ValueError(f"func_type {func_type} not supported")
        
    chunk_size = num_proc

    xs = coords[:, :, :, :3]
    nus = xs[:, 0, 0, 0]
    kxs = xs[0, :, 0, 1]
    kys = xs[0, 0, :, 2]
    ys = np.zeros_like(xs[:, :, :, 0])
    data_mask = np.zeros_like(ys, dtype=int)

    nus_inds = np.arange(len(nus))
    kxs_inds = np.arange(len(kxs))
    kys_inds = np.arange(len(kys))
    
    for mode_index in mode_params.keys():
        for nu_ind in mode_params[mode_index].keys():
            #print("Fitting mode", mode_index, nu_ind)
            for param_ind, nu, _, _, phi in mode_params[mode_index][nu_ind]:
                k = params[param_ind]
                kx = k*np.cos(phi)
                ky = k*np.sin(phi)
                alpha = np.array([nu, kx, ky])
                beta = params[param_ind + 1]

                radius = beta*np.tan(np.pi*(percentile-.5))

                if func_type == "lorenzian+gaussian":
                    sigma = params[param_ind + 3]
                    radius = max(radius, 2*sigma)

                #print(mode_index, nus[nu_ind], radius)
                fltr = np.abs(nus-nu) <= radius
                nus_close = nus[fltr]
                nus_close_inds = nus_inds[fltr]
                fltr = np.abs(kxs-kx) <= radius
                kxs_close = kxs[fltr]
                kxs_close_inds = kxs_inds[fltr]
                fltr = np.abs(kys-ky) <= radius
                kys_close = kys[fltr]
                kys_close_inds = kys_inds[fltr]
                for nu_i in range(len(nus_close)):
                    nu = nus_close[nu_i]
                    for kx_i in range(len(kxs_close)):
                        kx = kxs_close[kx_i]
                        for ky_i in range(len(kys_close)):
                            #print(nu_i, kx_i, ky_i)
                            x = np.array([nu, kx, kys_close[ky_i]])
                            r2 = np.sum((x-alpha)**2)
                            if r2 <= radius**2:
                                scale = params[param_ind + 2]
                                ys[nus_close_inds[nu_i], kxs_close_inds[kx_i], kys_close_inds[ky_i]] += scale/(np.pi*beta*(1+(np.sqrt(r2)/beta)**2))
                                data_mask[nus_close_inds[nu_i], kxs_close_inds[kx_i], kys_close_inds[ky_i]] = 1
                                if func_type == "lorenzian+gaussian":
                                    scale_gauss = params[param_ind + 4]
                                    ys[nus_close_inds[nu_i], kxs_close_inds[kx_i], kys_close_inds[ky_i]] += scale_gauss*np.exp(-.5*r2/(sigma**2))
                                #assert(np.product(np.isnan(ys).astype(int) == 0))
                                #print(nus_close_inds[nu_i], kxs_close_inds[kx_i], kys_close_inds[ky_i], ys[nus_close_inds[nu_i], kxs_close_inds[kx_i], kys_close_inds[ky_i]])
    #assert(np.all(np.abs(ys) < 1e5))
    
    '''    
    q = Queue()
    
    while len(alphas) > 0:
        print(len(alphas))
        chunk_size = min(chunk_size, len(alphas))
        ps = [Process(target=f, args=((alphas[i], betas[i], scales[i], x), q)) for i in range(chunk_size)]
        for p in ps:
            p.start()
        for _ in ps:
            ys += q.get()
        for p in ps:
            p.join()
        #alphas1 = np.tile(alphas[:chunk_size_, None, None, None, :], (1, coords.shape[0], coords.shape[1], coords.shape[2], 1))
        #betas1 = betas[:chunk_size_]
        #scales1 = scales[:chunk_size_]
        #r2 = np.transpose(np.sum((x-alphas1)**2, axis=4), (1, 2, 3, 0))
        ##fltr = r2 <= r2_max
        #ys += np.sum(scales1/(np.pi*betas1*(1+(np.sqrt(r2)/betas1)**2)), axis=3)
        alphas = alphas[chunk_size:]
        betas = betas[chunk_size:]
        scales = scales[chunk_size:]
    '''
    
    return ys, data_mask

def basis_func_grad(coords, params, mode_params, start_index=0, chunk_size=0, func_type="lorenzian"):
    if func_type not in supported_funcs:
        raise ValueError(f"func_type {func_type} not supported")

    xs = coords[:, :, :, :3]
    if chunk_size <= 0 or chunk_size > len(params):
        chunk_size = len(params)
        assert(start_index == 0)
    all_grads = np.zeros((coords.shape[0], coords.shape[1], coords.shape[2], chunk_size))

    num_params = 3
    if func_type == "lorenzian+gaussian":
        num_params = 5
    
    nus = xs[:, 0, 0, 0]
    kxs = xs[0, :, 0, 1]
    kys = xs[0, 0, :, 2]
    ys = np.zeros_like(xs[:, :, :, 0])
    
    nus_inds = np.arange(len(nus))
    kxs_inds = np.arange(len(kxs))
    kys_inds = np.arange(len(kys))
    
    for mode_index in mode_params.keys():
        for nu_ind in mode_params[mode_index].keys():
            #print("Fitting mode", mode_index, nu_ind)
            for param_ind, nu, _, _, phi in mode_params[mode_index][nu_ind]:
                param_ind -= start_index
                if param_ind < 0 or param_ind >= len(params):
                    continue
                k = params[param_ind]
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                kx = k*cos_phi
                ky = k*sin_phi
                alpha = np.array([nu, kx, ky])
                beta = params[param_ind + 1]
                radius = beta*np.tan(np.pi*(percentile-.5))
                if func_type == "lorenzian+gaussian":
                    sigma = params[param_ind + 3]
                    radius = max(radius, 2*sigma)
                fltr =np.abs(nus-nu) <= radius
                nus_close = nus[fltr]
                nus_close_inds = nus_inds[fltr]
                fltr = np.abs(kxs-kx) <= radius
                kxs_close = kxs[fltr]
                kxs_close_inds = kxs_inds[fltr]
                fltr = np.abs(kys-ky) <= radius
                kys_close = kys[fltr]
                kys_close_inds = kys_inds[fltr]
                for nu_i in range(len(nus_close)):
                    nu = nus_close[nu_i]
                    for kx_i in range(len(kxs_close)):
                        kx = kxs_close[kx_i]
                        for ky_i in range(len(kys_close)):
                            #print(nu_i, kx_i, ky_i)
                            x = np.array([nu, kx, kys_close[ky_i]])
                            r2 = np.sum((x-alpha)**2)
                            if r2 <= radius**2:
                                scale = params[param_ind + 2]
                                
                                coef1 = scale/np.pi
                                coef1a = coef1/beta
                                coef1b = 1./(1+r2/beta**2)**2
                                coef1c = 1./beta**2
                                coef2a = 1./(1+r2/beta**2)
                                
                                k_grad = coef1a*coef1b*coef1c*2.*((x[1]-alpha[1])*cos_phi+(x[2]-alpha[2])*sin_phi)
                                
                                beta_grad = coef1*(-coef2a + 2*r2*coef1b/beta**2)/beta**2
                                scale_grad = 1./(np.pi*beta)*coef2a
                                grads = [k_grad, beta_grad, scale_grad]
                                
                                if func_type == "lorenzian+gaussian":
                                    scale_gauss = params[param_ind + 4]
                                    gauss_val = np.exp(-.5*r2/(sigma**2))
                                    sigma_grad = scale_gauss*gauss_val*r2/(sigma**3)
                                    scale_grad_gauss = gauss_val
                                    grads.extend([sigma_grad, scale_grad_gauss])
                                
                                all_grads[nus_close_inds[nu_i], kxs_close_inds[kx_i], kys_close_inds[ky_i], param_ind:param_ind+num_params] = grads
    
    '''
    for i in range(len(params)):
        alphas = params[i, :3]
        beta = params[i, 3]
        scale = params[i, 4]

    
        #ys = np.zeros_like(x[:, :, :, 0])
        #r2_max = (approx_width*scale/beta-1)*beta
        r2 = np.sum((x-alphas)**2, axis=3)
        #fltr = r2 <= r2_max
        
        # scales1/(np.pi*betas1*(1+(np.sqrt(r2)/betas1)**2))
        coef1 = scale/np.pi
        coef1a = coef1/beta
        coef1b = 1./(1+r2/beta**2)**2
        coef1c = 1./beta**2
        coef2a = 1./(1+r2/beta**2)
        
        alpha_grad = np.tile((coef1a*coef1b*coef1c)[:, :, :, None], 3)*2.*(x-alphas)
        beta_grad = coef1*(-coef2a + 2*r2*coef1b/beta**2)/beta**2
        scale_grad = 1./(np.pi*beta)*coef2a
        grads = np.concatenate([alpha_grad, beta_grad[:, :, :, None], scale_grad[:, :, :, None]], axis=3)
        all_grads = np.concatenate([all_grads, grads], axis=3)
    '''
    return all_grads
    


def plot_mode(mode_index, params, nu_k_scale, fig, color, func_type="lorenzian"):
    if func_type == "lorenzian":
        #alphas = []
        #betas = []
        r_mean = 0
        alpha0_mean = 0
        for alpha0, alpha1, alpha2, beta, scale in params:
            fig.plot(alpha1/nu_k_scale, alpha2/nu_k_scale, f"{color}.")
            #alphas.append(alpha)
            #betas.append(beta)
        #alphas = np.asarray(alphas)
        #betas = np.asarray(betas)
        #indices = np.argsort
    else:
        raise ValueError(f"func_type {func_type} not supported")
    

def get_num_params(func_type="lorenzian"):
    if func_type == "lorenzian":
        return 3
    else:
        raise ValueError(f"func_type {func_type} not supported")
    

'''
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
'''

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len < 3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y

'''
def f(i):
    num_params = get_num_params()
    ys = basis_func(coords, params[i*num_params:i*num_params + num_params])
    print(ys.shape)
    return ys
'''

def interpolate_params(coords, params, mode_params, func_type="lorenzian"):
    print("params", len(params))
    nus = coords[:, 0, 0, 0]
    mode_params_out = dict()
    params_out = []
    if func_type == "lorenzian":
        for mode_index in mode_params.keys():
            if mode_index not in mode_params_out:
                mode_params_out[mode_index] = dict()
            all_nus = []
            all_nu_inds = []
            all_ks = []
            all_phis = []
            all_betas = []
            all_scales = []
            len_phis = []
            used_nus = []
            for nu_ind in mode_params[mode_index].keys():
                if nu_ind not in mode_params_out[mode_index]:
                    mode_params_out[mode_index][nu_ind] = []
                mode_per_nu = np.asarray(mode_params[mode_index][nu_ind])
                #print(mode_per_nu)
                param_indices = mode_per_nu[:, 0].astype(int)
                phis = mode_per_nu[:, 4]
                ks = params[param_indices]
                betas = params[param_indices+1]
                scales = params[param_indices+2]
                deg = 2
                coefs = np.polyfit(phis, ks, deg=2)
                powers = np.arange(deg+1)[::-1]
                phis_test = np.linspace(0, np.pi/2, int(num_phi_interp*np.mean(ks)/nu_k_scale/k_min))
                powers = np.reshape(np.repeat(powers, len(phis_test)), (len(powers), len(phis_test)))
                ws = np.reshape(np.repeat(coefs, len(phis_test)), (len(powers), len(phis_test)))
                ks_test = np.sum(ws*phis_test**powers, axis=0)
                betas_test = np.zeros_like(phis_test)
                scales_test = np.zeros_like(phis_test)
                norm = 0
                for i in range(len(phis)):
                    delta = np.abs(phis_test - phis[i])
                    betas_test += delta*betas[i]
                    scales_test += delta*scales[i]
                    norm += delta
                betas_test /= norm
                scales_test /= norm
                all_nus.extend(np.repeat(nus[nu_ind], len(ks_test)))
                all_nu_inds.extend(np.repeat(nu_ind, len(ks_test)))
                all_ks.extend(ks_test)
                all_phis.extend(phis_test)
                all_betas.extend(betas_test)
                all_scales.extend(scales_test)
                len_phis.append(len(phis_test))
                used_nus.append(nus[nu_ind])
            all_nus = np.asarray(all_nus)
            all_nu_inds = np.asarray(all_nu_inds)
            all_ks = np.asarray(all_ks)
            all_phis = np.asarray(all_phis)
            all_betas = np.asarray(all_betas)
            all_scales = np.asarray(all_scales)
            assert(all_nus.shape == all_ks.shape)
            assert(all_nus.shape == all_nu_inds.shape)
            assert(all_nus.shape == all_phis.shape)
            assert(all_nus.shape == all_betas.shape)
            assert(all_nus.shape == all_scales.shape)
            
            len_phis = np.asarray(len_phis)
            used_nus = np.asarray(used_nus)

            all_nus2 = []
            all_nu_inds2 = []
            all_ks2 = []
            all_phis2 = []
            all_betas2 = []
            all_scales2 = []

            min_nu = np.min(used_nus)
            max_nu = np.max(used_nus)
            start_index = np.argmin(np.abs(nus-min_nu))
            end_index = np.argmin(np.abs(nus-max_nu)) + 1
            for i in range(start_index, end_index, num_nu_interp):
                if i not in mode_params_out[mode_index]:
                    mode_params_out[mode_index][i] = []
                nu = nus[i]
                fltr1 = used_nus >= nu
                nus_fltr1 = used_nus[fltr1]
                upper = np.argmin(np.abs(nus_fltr1 - nu))
                upper_nu = nus_fltr1[upper]
                fltr2 = used_nus <= nu
                nus_fltr2 = used_nus[fltr2]
                lower = np.argmin(np.abs(nus_fltr2 - nu))
                lower_nu = nus_fltr2[lower]
                delta1 = np.abs(nu - upper_nu)
                delta2 = np.abs(nu - lower_nu)
                norm = delta1 + delta2
                print(delta1, delta2, len_phis[fltr1][upper], len_phis[fltr2][lower])
                if norm == 0:
                    delta1 = 1.
                    norm = 1.
                len_phis_test = int(np.round((len_phis[fltr1][upper]*delta2 + len_phis[fltr2][lower]*delta1)/norm))
                phis_test = np.linspace(0, np.pi/2, len_phis_test)
                upper = all_nus == upper_nu
                lower = all_nus == lower_nu
                upper_phis = all_phis[upper]
                lower_phis = all_phis[lower]
                upper_ks = all_ks[upper]
                lower_ks = all_ks[lower]
                upper_betas = all_betas[upper]
                lower_betas = all_betas[lower]
                upper_scales = all_scales[upper]
                lower_scales = all_scales[lower]
                for phi in phis_test:
                    upper = np.argmin(np.abs(upper_phis-phi))
                    lower = np.argmin(np.abs(lower_phis-phi))
                    k = (upper_ks[upper]*delta2 + lower_ks[lower]*delta1)/norm
                    beta = (upper_betas[upper]*delta2 + lower_betas[lower]*delta1)/norm
                    scale = (upper_scales[upper]*delta2 + lower_scales[lower]*delta1)/norm
                    all_ks2.append(k)
                    all_betas2.append(beta)
                    all_scales2.append(scale)
                all_phis2.extend(phis_test)
                all_nus2.extend(np.repeat(nu, len(phis_test)))
                all_nu_inds2.extend(np.repeat(i, len(phis_test)))
            
            all_nus = np.append(all_nus, all_nus2)
            all_nu_inds = np.append(all_nu_inds, all_nu_inds2)
            all_ks = np.append(all_ks, all_ks2)
            all_phis = np.append(all_phis, all_phis2)
            all_betas = np.append(all_betas, all_betas2)
            all_scales = np.append(all_scales, all_scales2)
            all_k1 = all_ks*np.cos(all_phis)
            all_k2 = all_ks*np.sin(all_phis)
            assert(all_nus.shape == all_ks.shape)
            assert(all_nus.shape == all_nu_inds.shape)
            assert(all_nus.shape == all_phis.shape)
            assert(all_nus.shape == all_betas.shape)
            assert(all_nus.shape == all_scales.shape)
            lens = np.arange(len(params_out), len(params_out) + len(all_ks)*3, 3)
            params_out.extend(np.concatenate([all_ks[:, None], all_betas[:, None], all_scales[:, None]], axis=1).flatten())
            #params_out.extend(np.concatenate([all_nus[:, None], all_k1[:, None], all_k2[:, None], all_betas[:, None], all_scales[:, None]], axis=1))
            for i in range(len(all_nu_inds)):
                nu_ind = all_nu_inds[i]
                mode_params_out[mode_index][nu_ind].append([lens[i], all_nus[i], all_k1[i], all_k2[i], all_phis[i]])

        params_out = np.asarray(params_out)
        
    else:
        raise ValueError(f"func_type {func_type} not supported")
    return params_out, mode_params_out

def fit(coords, params, mode_params):
    #import functools
    #from multiprocessing import Pool
    #num_params = get_num_params()
    #assert((len(params) % num_params) == 0)
    #fitted_data = np.zeros((coords.shape[0], coords.shape[1], coords.shape[2]))
    #with Pool(5) as p:
    #    fitted_data = functools.reduce(lambda x, y: x+y, \
    #        p.map(lambda i: basis_func(coords, params[i*num_params:i*num_params + num_params]), np.arange(len(params) // num_params)), \
    #        np.zeros((coords.shape[0], coords.shape[1], coords.shape[2])))
    #with Pool(4) as p:
    #    a = p.map(f, np.arange(len(params) // num_params))
    #for i in range(len(params) // num_params):
        #print("fit", i, len(params) // num_params)
    print(params)
    fitted_data, data_mask = basis_func(coords, params, mode_params)
    
    return fitted_data, data_mask


def calc_loglik(data_fitted, data, data_mask, sigma):
    loglik = -0.5 * np.sum(((data_fitted - data)*data_mask)**2/sigma) - 0.5*np.log(sigma) - 0.5*np.log(2.0*np.pi)
    return loglik        
        
def calc_loglik_grad(coords, data_fitted, data, data_mask, sigma, params, mode_params, func_type="lorenzian"):
    all_grads = np.empty_like(params)
    delta2 = ((data_fitted-data)*data_mask**2)[:, :, :, None]
    chunk_size_ = chunk_size*get_num_params(func_type)
    chunk_start = 0
    while len(params) > 0:
        chunk_size_ = min(chunk_size_, len(params))
        params1 = params[:chunk_size_]
        grads = basis_func_grad(coords, params1, mode_params, start_index=chunk_start, chunk_size=chunk_size_)
        print(chunk_start, chunk_size_, len(params1), grads.shape)
        all_grads[chunk_start:chunk_start+chunk_size_] = np.sum(np.tile(delta2, len(params1))*grads, axis=(0, 1, 2))
        chunk_start += chunk_size_
        params = params[chunk_size_:]
    return -all_grads/sigma

def bic(loglik, n, k):
    return np.log(n)*k - 2.*loglik

'''
def find_areas(x, y, alphas, betas, ws, scales, noise_std, k):
    ###########################################################################
    # Remove noise 
    #y_base = calc_y(x, [], [], ws, scale)
    #y -= y_base
    #inds = np.where(y > noise_std)[0]
    #y = y[inds]
    #x = x[inds]
    ###########################################################################
    y_base = calc_y(x, [[]], [[]], [ws], [scales], [k])
    ys_fit = []
    num_components = len(alphas)
    for i in np.arange(num_components):
        y_fit = calc_y(x, [alphas[i:i+1]], [betas[i:i+1]], [ws], [scales], [k])
        y_fit -= y_base
        ys_fit.append(y_fit)
    ys_fit = np.asarray(ys_fit)
    components_inds = np.argmax(ys_fit, axis=0)
    
    #print("components_inds", components_inds)
    areas = np.zeros(num_components)
    counts = np.zeros(num_components)
    ranges = np.zeros((num_components, 2))
    ranges[:, 0] = sys.float_info.max
    ranges[:, 1] = sys.float_info.min

    for i in np.arange(len(y)):
        component_ind = components_inds[i]
        areas[component_ind] += y[i]
        #######################################################################
        # Subtract contribution from polynomial base and other components
        areas[component_ind] -= y_base[i]
        for j in np.arange(num_components):
            if j != component_ind:
                areas[component_ind] -= ys_fit[j][i]
        #######################################################################
        counts[component_ind] += 1.
        # Update component ranges
        if x[i] < ranges[component_ind, 0]:
            ranges[component_ind, 0] = x[i]
        if x[i] > ranges[component_ind, 1]:
            ranges[component_ind, 1] = x[i]
        
    for i in np.arange(num_components):
        areas[i] *= (ranges[i, 1]-ranges[i, 0])/counts[i]
    
    return areas, ranges
'''    
    
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
        print("Usage: python fmode.py input_path [year] [input_file]")
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
    
    
    map_scale = 0.05

    #ks = np.linspace(-3600, 3600, 300)
    #print(ks)
    #nus = np.linspace(-11.076389, 11.076389, 641)
    #print(nus)
    
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
            
            noise_var = get_noise_var(data, ks, nus)        
            sig_var = np.var(data) - noise_var
            true_sigma = np.sqrt(noise_var)
            
            #######################################################################
            '''
            for i in range(0, len(nus), 10):
                log_data = np.log(data[i])
                
                fig = plot.plot(nrows=1, ncols=1, size=plot.default_size(data.shape[1], data.shape[2]))
                fig.contour(ks, ks, data[i], levels=100)
                fig.save(os.path.join(output_dir, f"ring_diagram{i}.png"))
    
                #fig, ax = plt.subplots(nrows=1, ncols=1)
                #ax.contour(ks, ks, np.log(data[i]), levels=levels)
                #fig.savefig(os.path.join(output_dir, f"ring_diagram{i}.png"))
                #plt.close(fig)
            '''
            #######################################################################
            
            hdul.close()
                
            #levels = np.linspace(np.min(np.log(data))+2, np.max(np.log(data))-2, 200)        
            #fig, ax = plt.subplots(nrows=1, ncols=1)
            #ax.contour(ks[len(ks)//2-1:], nus[len(nus)//2+1:], np.log(data[:data.shape[0]//2, 0, :data.shape[2]//2 + 1]), levels=levels)
            #fig.savefig(os.path.join(output_dir, "spectrum1.png"))
            #plt.close(fig)
            #sys.exit()
            
            f1 = open(os.path.join(output_dir, 'areas.txt'), 'w')
            f1.write('k num_components f_mode_area\n')
            
            results = []
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
            
            
            #data_mask = np.zeros_like(data, dtype=int)
            
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
            
            #for k_ind1 in range(0, coords.shape[1]):
            #    for k_ind2 in range(0, coords.shape[2]):
            #        _, _, _, k = coords[0, k_ind1, k_ind2]
            #        if k >= k_min and k <= k_max_:
            #            data_mask[:, k_ind1, k_ind2] = 1

            priors = load("priors.dat")
            if priors is not None:
                params, bounds, mode_params = priors
            else:
                ring_radii = load("ring_radii.dat")
                assert(ring_radii is not None)
                params = []
                bounds = []
                #mode_info = dict()
                mode_params = dict()
                #sampling_step = [coords[3, 0, 0, 0] - coords[0, 0, 0, 0], coords[0, 3, 0, 1] - coords[0, 0, 0, 1], coords[0, 0, 3, 2] - coords[0, 0, 0, 2]]
                #print(coords[0, 3, 0, 1], coords[0, 0, 0, 1])
                sampling_step = (coords[10, 0, 0, 0] - coords[0, 0, 0, 0])**2
                print("sampling_step", sampling_step)

                for mode_index in range(5):
                    if mode_index not in mode_params:
                        #mode_info[mode_index] = dict()
                        mode_params[mode_index] = dict()
                    mode_radii = ring_radii[mode_index]
                    for nu_ind in range(0, len(mode_radii), nu_sampling):
                        r = mode_radii[nu_ind]
                        num_points_per_arc = int(round(k_sampling*r/k_min))
                        if r >= k_min and r <= k_max_:
                            if nu_ind not in mode_params[mode_index]:
                                mode_params[mode_index][nu_ind] = []
                            nu = nus_filtered[nu_ind]
                            print(nu)
                            for phi in np.linspace(0, 0.5*np.pi, num_points_per_arc):
                                kx = r*np.cos(phi)
                                ky = r*np.sin(phi)
                                k1 = nu_k_scale*ks_filtered[np.argmin(np.abs(ks_filtered-kx))]
                                k2 = nu_k_scale*ks_filtered[np.argmin(np.abs(ks_filtered-ky))]
    
                                #mode_info[mode_index][nu_ind].append(len(params))
                                print(len(mode_params[mode_index][nu_ind]))
                                print(len(params), nu, k1, k2)
                                mode_params[mode_index][nu_ind].append([len(params), nu, k1, k2, np.arctan2(k2, k1)])
                                
                                k = np.sqrt(k1**2 + k2**2)
                                params.append(k)

                                beta_prior = (r-700)*(.1-.033)/2300 + .033#.2/num_components

                                bounds.append((k-6*beta_prior, k+6*beta_prior))
                                
                                #params.append(nu)
                                #bounds.append((nu-.5 , nu+.5))
        
                                #params.append(k1)
                                #bounds.append((k1-.5 , k1+.5))
        
                                #params.append(k2)
                                #bounds.append((k2-.5 , k2+.5))
        
                                params.append(beta_prior)
                                #params.append(1./100)
                                bounds.append((1e-4 , .1))
        
                                scale_prior = np.mean(data[nu_ind])#.01
                                params.append(scale_prior)
                                bounds.append((0, None))
                
            
                #for i in mode_info.keys():
                #    mode_info[i] = np.asarray(mode_info[i])
                save((params, bounds, mode_params), "priors.dat")
            print("Priors set")
            #######################################################################    
            print("params", len(params))
            #y = y[:, k_indices]
            #k = k[k_indices]
                    
            alphas_est = []
            betas_est = []
            ws_est = []
            scales_est = []
                
            min_loglik = None
            def lik_fn(params):
                interp_params, interp_mode_params = interpolate_params(coords, params, mode_params)
                data_fitted, data_mask = fit(coords, interp_params, interp_mode_params)
                #data_fitted, data_mask = fit(coords, params, mode_params)
                
                for i in range(0, coords.shape[0], 10):
                    fig = plot.plot(nrows=1, ncols=4, size=plot.default_size(2*data.shape[1], 2*data.shape[2]))
                    #fig.contour(coords[0, :, 0, 1], coords[0, 0, :, 2], fitted_data[i, :, :])
                    fig.set_axis_title(r"$\nu=" + str(coords[i, 0, 0, 0]) + "$")
                    fig.colormap((data[i]*data_mask[i])[::-1], cmap_name="gnuplot", show_colorbar=True, ax_index=0)
                    fig.colormap((data_fitted[i]*data_mask[i])[::-1], cmap_name="gnuplot", show_colorbar=True, ax_index=1)
                    fig.colormap((np.abs(data[i] - data_fitted[i])*data_mask[i])[::-1], cmap_name="gnuplot", show_colorbar=True, ax_index=2)
                    fig.colormap((data_mask[i])[::-1], cmap_name="gnuplot", show_colorbar=True, ax_index=3)
                    fig.save(os.path.join(output_dir, f"fitted_data{i}.png"))
                
                
                
                loglik = -calc_loglik(data_fitted, data, data_mask, true_sigma)
                print(loglik)
                return loglik, -calc_loglik_grad(coords, data_fitted, data, data_mask, true_sigma, params, mode_params)

            

            min_res = None
            
            for trial_no in np.arange(0, num_optimizations):                    
                #print("params", params)
                    
                #initial_lik = lik_fn(params)
                #res = scipy.optimize.minimize(lik_fn, params, method='CG', jac=None, options={'disp': True, 'gtol':1e-7})#, 'eps':.1})
                res = scipy.optimize.minimize(lik_fn, params, method='L-BFGS-B', jac=True, bounds=bounds, options={'disp': True, 'gtol':1e-7})
                loglik = res['fun']
                if min_loglik is None or loglik < min_loglik:
                    min_loglik = loglik
                    min_res = res['x']
            
            params_est = min_res
            num_params = get_num_params()
            params_ = []
            for _ in range(len(nus_filtered)//nu_sampling+1):
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
                    params_[nu_ind//nu_sampling][mode_index].append(params_est[start_index:start_index+num_params])
    
            colors = ["k", "b", "g", "r", "m"]
            for i in range(0, len(params_)):
                contains_data = False
                for mode_index in range(5):
                    if len(params_[i][mode_index]) > 0:
                        contains_data = True
                if not contains_data:
                    continue
                fig = plot.plot(nrows=1, ncols=1, size=plot.default_size(data.shape[1], data.shape[2]))
                fig.contour(ks_filtered, ks_filtered, data[i*nu_sampling, :, :])
                fig.set_axis_title(r"$\nu=" + str(coords[i*nu_sampling, 0, 0, 0]) + "$")
                fig.colormap(np.log(data[i, :, :]), cmap_name="gnuplot", show_colorbar=True)
                for mode_index in range(5):
                    if len(params_[i][mode_index]) > 0:
                        plot_mode(mode_index, params_[i][mode_index], nu_k_scale, fig, colors[mode_index])
                fig.save(os.path.join(output_dir, f"ring_diagram{i}.png"))
                
                
                #f1.write('%s %s %s' % (str(k_value), opt_num_components, areas[0]) + "\n")
                #print("Lowest BIC", min_bic)
                #print("Num components", opt_num_components)
                #print("Areas", areas)
                #f1.flush()
                
            
            f1.close()
            
            
