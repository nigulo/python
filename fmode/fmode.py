import matplotlib as mpl
mpl.use('Agg')
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
import plot
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.optimize

def basis_func(coords, params, func_type="lorenzian"):
    if func_type == "lorenzian":
        chunk_size = 8 # For memory purposes
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
        chunk_start = 0
        while len(alphas) > 0:
            chunk_size = min(chunk_size, len(alphas))
            alphas1 = np.tile(alphas[chunk_start:chunk_start+chunk_size, None, None, None, :], (1, coords.shape[0], coords.shape[1], coords.shape[2], 1))
            betas1 = betas[chunk_start:chunk_start+chunk_size]
            scales1 = scales[chunk_start:chunk_start+chunk_size]
            r2 = np.transpose(np.sum((x-alphas1)**2, axis=4), (1, 2, 3, 0))
            #fltr = r2 <= r2_max
            ys += np.sum(scales1/(np.pi*betas1*(1+(np.sqrt(r2)/betas1)**2)), axis=3)
            chunk_start += chunk_size
            alphas = alphas[chunk_start:]
            betas = betas[chunk_start:]
            scales = scales[chunk_start:]
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

def f(i):
    num_params = get_num_params()
    ys = basis_func(coords, params[i*num_params:i*num_params + num_params])
    print(ys.shape)
    return ys

def fit(coords, params):
    #import functools
    #from multiprocessing import Pool
    print("fit start")
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
    print("fit end")
    return fitted_data


def calc_loglik(data_fitted, data, data_mask, sigma):
    loglik = -0.5 * np.sum(((data_fitted - data)*data_mask)**2/sigma) - 0.5*np.log(sigma) - 0.5*np.log(2.0*np.pi)
    return loglik        
    

def calc_loglik_grad(coords, data_fitted, data, data_mask, sigma, params, func_type="lorenzian"):
    if func_type == "lorenzian":
        x = coords[:, :, :, :3]
        all_grads = np.array([])
        for i in range(len(params) // 5):
            alphas = params[i*5:i*5+3]
            beta = params[i*5+3]
            scale = params[i*5+4]
        
            #ys = np.zeros_like(x[:, :, :, 0])
            #r2_max = (approx_width*scale/beta-1)*beta
            r2 = np.sum((x-alphas)**2, axis=3)
            #fltr = r2 <= r2_max
            
            coef2 = scale/np.pi
            coef3 = np.sqrt(r2)
            coef4 = 1+(coef3/beta)**3
            alpha_grad = np.tile((-2*coef2/(beta*coef4*coef3/beta))[:, :, :, None], 3)*(x-alphas)
            beta_grad = coef2*(-1/(1+(coef3/beta)**2)*beta**2+2*coef3/(beta**3*coef4))
            scale_grad = 1./(np.pi*beta*(1+coef3/beta)**2)
            grads = np.concatenate([alpha_grad, beta_grad[:, :, :, None], scale_grad[:, :, :, None]], axis=3)
            print(grads.shape)
            print(data_fitted.shape)
            print(((data_fitted-data)*data_mask)[:, :, :, None].shape)
            print(np.tile(((data_fitted-data)*data_mask)[:, :, :, None], 5).shape)
            grads = -np.sum(np.tile(((data_fitted-data)*data_mask)[:, :, :, None], 5)*grads, axis=(0, 1, 2))/sigma
            print(grads.shape)
            all_grads = np.concatenate([all_grads, grads])
    
        return all_grads
        
    else:
        raise ValueError(f"func_type {func_type} not supported")

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
            k_grid = np.transpose([np.tile(ks_filtered, len(ks_filtered)), np.repeat(ks_filtered, len(ks_filtered))])
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
            mode_info = []
            
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
            
            for k_ind1 in range(0, coords.shape[1], 30):
                for k_ind2 in range(0, coords.shape[2], 30):
                    _, k1, k2, k = coords[0, k_ind1, k_ind2]
                    if k >= k_min and k <= k_max_:
                        data_mask[:, k_ind1, k_ind2] = 1
                        num_components = get_num_components(k)
                        
                        nus_ = coords[:, k_ind1, k_ind2, 0]
                        for i in range(num_components):
                            alpha_prior0 = get_alpha_prior(i, k)
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
    
                            nu_index = np.argmin(np.abs(nus_ - alpha_prior0))
                            mode_info.append((i, nu_index, k_ind1, k_ind2))
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
                return grad(data_fitted, data, data_mask, true_sigma, params)
                
            
            
            min_res = None
            for trial_no in np.arange(0, num_optimizations):                    
                #print("params", params)
                    
                #initial_lik = lik_fn(params)
                #res = scipy.optimize.minimize(lik_fn, params, method='CG', jac=None, options={'disp': True, 'gtol':1e-7})#, 'eps':.1})
                res = scipy.optimize.minimize(lik_fn, params, method='L-BFGS-B', jac=None, bounds=bounds, options={'disp': True, 'gtol':1e-7})
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
                fig.set_axis_title(f"$\nu={coords[i, 0, 0, 0]}$")
                fig.contour(ks, ks, data[i, :, :])
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
            
            
