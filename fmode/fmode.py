import matplotlib as mpl
mpl.use('nbAgg')
print(mpl.get_backend())
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

chunk_size = 1 # For memory purposes


def basis_func(coords, params, func_type="lorenzian"):
    chunk_size_ = chunk_size
    if func_type == "lorenzian":
        ai1 = np.arange(0, len(params), 5)
        ai2 = np.arange(1, len(params), 5)
        ai3 = np.arange(2, len(params), 5)
        ai = np.stack([ai1, ai2, ai3], axis=1)
        bi = np.arange(3, len(params), 5)
        si = np.arange(4, len(params), 5)
        a = np.zeros((21, 231, 150, 150, 3))
        alphas = params[ai]
        betas = params[bi]
        scales = params[si]

        x = coords[:, :, :, :3]
        ys = np.zeros_like(x[:, :, :, 0])
        while len(alphas) > 0:
            chunk_size_ = min(chunk_size_, len(alphas))
            alphas1 = np.tile(alphas[:chunk_size_, None, None, None, :], (1, coords.shape[0], coords.shape[1], coords.shape[2], 1))
            betas1 = betas[:chunk_size_]
            scales1 = scales[:chunk_size_]
            r2 = np.transpose(np.sum((x-alphas1)**2, axis=4), (1, 2, 3, 0))
            #fltr = r2 <= r2_max
            ys += np.sum(scales1/(np.pi*betas1*(1+(np.sqrt(r2)/betas1)**2)), axis=3)
            alphas = alphas[chunk_size_:]
            betas = betas[chunk_size_:]
            scales = scales[chunk_size_:]
        #alphas = np.array([params[0], params[1], params[2]])
        #beta = params[3]
        #scale = params[4]
        #r2_max = (approx_width*scale/beta-1)*beta
        #r2 = np.sum((x-alphas)**2, axis=3)
        #fltr = r2 <= r2_max
        #ys = scales/(np.pi*betas*(1+(np.sqrt(r2)/betas)**2))
    else:
        raise ValueError(f"func_type {func_type} not supported")
    return ys

def basis_func_grad(coords, params, func_type="lorenzian"):
    if func_type == "lorenzian":
        x = coords[:, :, :, :3]
        all_grads = np.empty((coords.shape[0], coords.shape[1], coords.shape[2], 0))
        for i in range(len(params) // 5):
            alphas = params[i*5:i*5+3]
            beta = params[i*5+3]
            scale = params[i*5+4]
        
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
    
        return all_grads
        
    else:
        raise ValueError(f"func_type {func_type} not supported")


def plot_mode(params, nu_k_scale, fig, color, func_type="lorenzian"):
    if func_type == "lorenzian":
        #alphas = []
        #betas = []
        for alpha0, alpha1, alpha2, beta, scale in params:
            print(alpha1/nu_k_scale, alpha2/nu_k_scale)
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

def interpolate_params(coords, params, mode_info, func_type="lorenzian"):
    coords = coords[::3, ::3, ::3, :]
    coords = np.reshape(coords, (coords.shape[0]*coords.shape[1]*coords.shape[2], coords.shape[3]))
    if func_type == "lorenzian":
        x = coords[:, :3]
        for mode_index in mode_info.keys():
            mode_params = []
            for index in mode_info[mode_index]:
                mode_params.append(params[index:index+3])
            mode_params = np.asarray(mode_params)
            size = int(np.sqrt(len(mode_params)))
            print(mode_params.shape)
            surf = fitting.interpolate_surface(mode_params[:, :3], size_u=size, size_v=size, degree_u=2, degree_v=2)
            surf.delta = 0.05
            from geomdl.visualization import VisMPL as vis
            surf.vis = vis.VisSurface(config=vis.VisConfig(ctrlpts=False, trims=False))
            print(mpl.get_backend())
            surf.render(fig_save_as=f"surface{mode_index}.png", display_plot=False)
            print(f"rendered {mode_index}")
            sys.exit()
            
            
            surf.evaluate(start=[0.0, 0.0], stop=[1.0, 1.0])
            print(surf.evalpts)
            
            rings = dict()
            for i in range(0, len(mode_params), 5):
                nu = params[i]
                nu_index = np.argmin(np.abs(nus_filtered - nu))
                if nu_index not in rings:
                    rings[nu_index] = []
                rings[nu_index] = params[i:i+5]

    
        
    else:
        raise ValueError(f"func_type {func_type} not supported")

def fit(coords, params):
    interpolate_params(coords, params, mode_info)
    #import functools
    #from multiprocessing import Pool
    num_params = get_num_params()
    assert((len(params) % num_params) == 0)
    #fitted_data = np.zeros((coords.shape[0], coords.shape[1], coords.shape[2]))
    #with Pool(5) as p:
    #    fitted_data = functools.reduce(lambda x, y: x+y, \
    #        p.map(lambda i: basis_func(coords, params[i*num_params:i*num_params + num_params]), np.arange(len(params) // num_params)), \
    #        np.zeros((coords.shape[0], coords.shape[1], coords.shape[2])))
    #with Pool(4) as p:
    #    a = p.map(f, np.arange(len(params) // num_params))
    #for i in range(len(params) // num_params):
        #print("fit", i, len(params) // num_params)
    fitted_data = basis_func(coords, params)
    return fitted_data


def calc_loglik(data_fitted, data, data_mask, sigma):
    loglik = -0.5 * np.sum(((data_fitted - data)*data_mask)**2/sigma) - 0.5*np.log(sigma) - 0.5*np.log(2.0*np.pi)
    return loglik        
        
def calc_loglik_grad(coords, data_fitted, data, data_mask, sigma, params, func_type="lorenzian"):
    all_grads = np.empty_like(params)
    delta2 = ((data_fitted-data)*data_mask**2)[:, :, :, None]
    chunk_size_ = chunk_size*get_num_params(func_type)
    chunk_start = 0
    while len(params) > 0:
        chunk_size_ = min(chunk_size_, len(params))
        params1 = params[:chunk_size_]
        grads = basis_func_grad(coords, params1)
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
        print("Usage: python fmode input_path [year] [input_file]")
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
    
    #k = np.linspace(0, 3600, 151)
    ks = np.linspace(-3600, 3600, 300)
    print(ks)
    
    #cadence = 45.0
    #nf = 641
    #omega = np.arange((nf-1)/2+1)
    #omega = omega*2.*np.pi/(nf*cadence/1000.)
    #nu = omega/(2.*np.pi)
    
    #nu = np.linspace(0, 11.076389, 320)
    nus = np.linspace(-11.076389, 11.076389, 641)
    print(nus)
    
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
            
            noise_var = get_noise_var(data, ks, nus)        
            sig_var = np.var(data) - noise_var
            true_sigma = np.sqrt(noise_var)
            
            #######################################################################
            '''
            for i in range(0, 320, 10):            
                levels = np.linspace(np.min(np.log(data[i]))+2, np.max(np.log(data[i]))-2, 200)
                
                fig = plot.plot(nrows=1, ncols=1, size=plot.default_size(data.shape[1], data.shape[2]))
                fig.contour(ks, ks, data[i, :, :])
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
            
            params = []
            bounds = []
            
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
            
            coords = np.empty((len(nus_filtered), len(ks_filtered), len(ks_filtered), 4))
            nu_k_scale = (nus[-1]-nus[0])/(ks[-1]-ks[0])
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
    
            print("Coordinate grid created")
            mode_info = dict()
            mode_params = dict()
            
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
                data_fitted = fit(coords, params)
                return -calc_loglik(data_fitted, data, data_mask, true_sigma)
            
            def jac(params):
                data_fitted = fit(coords, params)
                return calc_loglik_grad(coords, data_fitted, data, data_mask, true_sigma, params)
                
            

            min_res = None
            for trial_no in np.arange(0, num_optimizations):                    
                #print("params", params)
                    
                #initial_lik = lik_fn(params)
                #res = scipy.optimize.minimize(lik_fn, params, method='CG', jac=None, options={'disp': True, 'gtol':1e-7})#, 'eps':.1})
                res = scipy.optimize.minimize(lik_fn, params, method='L-BFGS-B', jac=jac, bounds=bounds, options={'disp': True, 'gtol':1e-7})
                loglik = res['fun']
                if min_loglik is None or loglik < min_loglik:
                    min_loglik = loglik
                    min_res = res['x']

            params_est = min_res
          
            num_params = get_num_params()
            params_ = []
            for i in range(data.shape[0]):
                mode_params = []
                for _ in range(5):
                    mode_params.append([])
                params_.append(mode_params)
            for mode_index in range(5):
                for i in range(len(mode_info)):
                    mode_i, nu_ind, _, _ = mode_info[i]
                    if mode_i == mode_index:
                        #print(nu_ind, mode_index, len(params_est[i*num_params:i*num_params+num_params]))
                        params_[nu_ind][mode_index].append(params_est[i*num_params:i*num_params+num_params])
    
            colors = ["k", "b", "g", "r", "m"]
            for i in range(0, len(params_)):
                contains_data = False
                for mode_index in range(5):
                    if len(params_[i][mode_index]) > 0:
                        contains_data = True
                if not contains_data:
                    continue
                fig = plot.plot(nrows=1, ncols=1, size=plot.default_size(data.shape[1], data.shape[2]))
                fig.contour(ks_filtered, ks_filtered, data[i, :, :])
                fig.set_axis_title(r"$\nu=" + str(coords[i, 0, 0, 0]) + "$")
                #fig.colormap(np.log(data[i, :, :]), cmap_name="gnuplot", show_colorbar=True)
                for mode_index in range(5):
                    if len(params_[i][mode_index]) > 0:
                        print(i, mode_index)
                        plot_mode(params_[i][mode_index], nu_k_scale, fig, colors[mode_index])
                fig.save(os.path.join(output_dir, f"ring_diagram{i}.png"))
            
                #f1.write('%s %s %s' % (str(k_value), opt_num_components, areas[0]) + "\n")
                #print("Lowest BIC", min_bic)
                #print("Num components", opt_num_components)
                #print("Areas", areas)
                #f1.flush()
                
            
            f1.close()
            
            
