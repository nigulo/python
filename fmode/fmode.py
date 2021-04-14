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

mode_priors = np.genfromtxt("GVconst.txt")

def basis_func(coords, params, func_type="lorenzian"):
    if func_type == "lorenzian":
        x = coords[:, :, :, :3]
        alphas = np.array([params[0], params[1], params[2]])
        beta = params[3]
        scale = params[4]
        ys = scale/(np.pi*beta*(1+(np.sqrt(np.sum((x-alphas)**2))/beta)**2))
    else:
        raise ValueError(f"func_type {func_type} not supported")
    return ys

def plot_mode(params, nu_k_scale, fig, style, func_type="lorenzian"):
    if func_type == "lorenzian":
        #alphas = []
        #betas = []
        for alpha0, alpha1, alpha2, beta, scale in params:
            fig.plot(alpha1/nu_k_scale, alpha2/nu_k_scale, style)
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
    
'''
def get_normals(k_nu, k_scale, nu_scale):
    k_nu = np.array(k_nu)
    k_nu[:, 0] /= k_scale
    k_nu[:, 1] /= nu_scale
    normals = k_nu[1:, :] - k_nu[:-1, :]
    normals /= np.sqrt(np.sum(normals**2, axis=1, keepdims=True))
    tangentials = np.array(normals)
    tangentials[:, 0] *= k_scale/100
    tangentials[:, 1] *= nu_scale/100
    normals = normals[:, 1]*1.j + normals[:, 0]
    normals *= 1.j
    normals = np.concatenate((np.real(normals)[:, None], np.imag(normals)[:, None]), axis=1)
    assert(np.all(normals == np.real(normals)))
    normals[:, 0] *= k_scale/100
    normals[:, 1] *= nu_scale/100
    return normals, tangentials

def get_slice(k, nu, ks, nus, tangential, eps=1e-2):
    k_nu = np.array([k, nu])
    indices = []
    for i in range(len(ks)):
        for j in range(len(nus)):
            delta = np.array([ks[i], nus[j]]) - k_nu
            if abs(np.sum(delta*tangential)) <= eps:
                indices.append([i, j])
    return np.sort(np.asarray(indices))
'''

'''
def get_alpha_prior(i, k_y):
    assert(i >= 0 and i <= 3)
    g_sun=274.*1e-6 # Mm s^(-2)
    R_sun=696. # Mm
    A=g_sun/R_sun
    if i == 0:
        return (1000./(2*np.pi))*np.sqrt(A*k_y) #units=mHz
    else:
        return (1000./(2*np.pi))*np.sqrt((float(i) +.5)*A*k_y)
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
def fit(x, coefs):
    y = np.zeros_like(x)
    power = 0
    for coef in coefs[::-1]:
        y += coef*(x**power)
        power += 1
    return y

def get_mode_params(mode_index, alphas, betas, scales, k):
    alphas_mode = []
    betas_mode = []
    scales_mode = []
    indices_to_delete = []
    for j in range(len(alphas)):
        if len(alphas[j]) > mode_index:
            alphas_mode.append(alphas[j][mode_index])
            betas_mode.append(betas[j][mode_index])
            assert(scales[j][mode_index] > 0)
            scales_mode.append(scales[j][mode_index])
        else:
            indices_to_delete.append(j)
    return np.asarray(alphas_mode), np.asarray(betas_mode), np.asarray(scales_mode), np.delete(k, indices_to_delete)
'''

def fit(coords, params):
    num_params = get_num_params()
    assert((len(params) % num_params) == 0)
    fitted_data = np.zeros((coords.shape[0], coords.shape[1], coords.shape[2]))
    for i in range(len(params) // num_params):
        fitted_data += basis_func(coords, params[i*num_params:i*num_params + num_params])
    return fitted_data

'''
def calc_y(x, alphas, betas, ws, scales, k):
    alphas_fitted = []
    betas_fitted = []
    ws_fitted = []
    scales_fitted = []
    if len(alphas) > 1:
        for i in range(5):
            alphas_to_fit, betas_to_fit, scales_to_fit, k1 = get_mode_params(i, alphas, betas, scales, k)
            #print("alphas, betas", alphas_to_fit.shape, betas_to_fit.shape, k.shape)
            alpha_coefs = np.polyfit(k1, alphas_to_fit, alpha_pol_degree)
            beta_coefs = np.polyfit(k1, betas_to_fit, beta_pol_degree)
            #scale_coefs = np.polyfit(k1, scales_to_fit, scale_pol_degree)
            #print("coefs", alpha_coefs, beta_coefs)
    
            alphas_fitted.append(fit(k1, alpha_coefs))
            betas_fitted.append(fit(k1, beta_coefs))
            scales_fitted.append(smooth(x=scales_to_fit, window_len=scale_smooth_win)[scale_smooth_win//2:-(scale_smooth_win//2)])
            assert(len(scales_fitted[-1]) == len(scales_to_fit))
            assert(np.all(scales_fitted[-1] > 0))
    
        for j in range(len(ws[0])):
            w_coefs = np.polyfit(k, np.asarray(ws)[:, j], w_pol_degree)
            ws_fitted.append(fit(k, w_coefs))    
    else:
        for i in range(len(alphas[0])):
            alphas_fitted.append([alphas[0][i]])
        for i in range(len(betas[0])):
            betas_fitted.append([betas[0][i]])
        for i in range(len(scales[0])):
            scales_fitted.append([scales[0][i]])
        for i in range(len(ws[0])):
            ws_fitted.append([ws[0][i]])
    
    ys = np.empty((len(x), len(alphas)))
    for j in range(len(alphas)):
        y = np.zeros_like(x)
        for i in np.arange(len(alphas[j])):
            if len(alphas_fitted[i]) > j:
                alpha = alphas_fitted[i][j]
                beta = betas_fitted[i][j]
                scale = scales_fitted[i][j]
                y += scale/(np.pi*beta*(1+((x-alpha)/beta)**2))
        for i in np.arange(len(ws[j])):
            y += ws_fitted[i][j]*x**i
        ys[:, j] = y
    return ys
'''

def calc_loglik(data_fitted, data, data_mask, sigma):
    loglik = -0.5 * np.sum(((data_fitted - data)*data_mask)**2/sigma) - 0.5*np.log(sigma) - 0.5*np.log(2.0*np.pi)
    return loglik        
    
def bic(loglik, n, k):
    return np.log(n)*k - 2.*loglik

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
    
def get_noise_var(data, ks, nus):
    k_indices = np.where(np.logical_and(ks >= 3500, ks <= 4500))[0]
    nu_indices = np.where(np.logical_and(nus >= 2, nus <= 4))[0]
    d = data[nu_indices]
    d = d[:, k_indices, :]
    d = d[:, :, k_indices]
    return np.var(d)

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
            d1 = fft.fftshift(data[i])
            #sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
            #import plot
            #test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(d1.shape[1], d1.shape[0]))
            #test_plot.colormap(np.log(d1), cmap_name="gnuplot", show_colorbar=True)
            #test_plot.save("spectrum1a.png")
            #test_plot.close()
            
            levels = np.linspace(np.min(np.log(d1))+2, np.max(np.log(d1))-2, 200)
            
            fig, ax = plt.subplots(nrows=1, ncols=1)
            kxy = np.concatenate([-k[1:][::-1], k[:-1]])
            ax.contour(kxy, kxy, np.log(d1), levels=levels)
            fig.savefig(os.path.join(output_dir, f"ring_diagram{i}.png"))
            plt.close(fig)
        '''
        #######################################################################
        
        '''
        data1 = data[:, 0, :data.shape[2]//2 + 1]
        data2 = data[:, :data.shape[1]//2 + 1, 0]
        data3 = np.empty((data1.shape[0], data1.shape[1]))
        data4 = np.empty((data1.shape[0], data1.shape[1]))
        last_j = 0
        for i in range(int(round(data1.shape[1]/np.sqrt(2)))):
            j = int(round(i * np.sqrt(2)))
            data3[:, j] = data[:, i, i]
            data4[:, j] = data[:, i, -i]
            for j1 in range(last_j+1, j):
                #print(last_j, j, j1)
                data3[:, j1] = (data3[:, last_j] + data3[:, j])/2
                data4[:, j1] = (data4[:, last_j] + data4[:, j])/2
            last_j = j
        if j == data1.shape[1] - 2:
            j += 1
            data3[:, j] = (data3[:, j - 1] + data[:, i+1, i+1])/2
            data4[:, j] = (data4[:, j - 1] + data[:, i+1, -(i+1)])/2
            
        assert(j == data1.shape[1] - 1)
        data = data1#(data1 + data2 + data3 + data4)/4
        '''
        
        hdul.close()
            
        levels = np.linspace(np.min(np.log(data))+2, np.max(np.log(data))-2, 200)        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.contour(ks[len(ks)//2-1:], nus[len(nus)//2+1:], np.log(data[:data.shape[0]//2, 0, :data.shape[2]//2 + 1]), levels=levels)
        '''
        alphas = dict()
        for j in range(len(k)):
            num_components = get_num_components(k[j])
            for i in range(num_components):
                if i not in alphas:
                    alphas[i] = []
                alphas[i].append(np.array([k[j], get_alpha_prior(i, k[j])]))
        for i in alphas:
            k_nu = np.asarray(alphas[i])
            alpha_coefs = np.polyfit(k_nu[:, 0], k_nu[:, 1], alpha_pol_degree)
            alphas_fitted = fit(k_nu[:, 0], alpha_coefs)
            k_nu = np.concatenate((k_nu[:, 0][:,None], alphas_fitted[:, None]), axis=1)
            normals, tangentials = get_normals(k_nu, k[-1]-k[0], nu[-1]-nu[0])
            #print(normals)
            for j in range(0, len(normals), 10):
                k_nu_ = np.linspace(k_nu[j, :]-normals[j], k_nu[j, :]+normals[j])
                ax.plot(k_nu_[:, 0], k_nu_[:, 1], "k-")
                k_nu_ = np.linspace(k_nu[j, :]-tangentials[j], k_nu[j, :]+tangentials[j])
                ax.plot(k_nu_[:, 0], k_nu_[:, 1], "k-")
                #if i == 0:
                #    indices = get_slice(k_nu[j, 0], k_nu[j, 1], k, nu, tangentials[j])
                #    print(indices)
                #    ax.plot(k[indices[::10, 0]], nu[indices[::10, 1]], "k+")
        '''
        fig.savefig(os.path.join(output_dir, "spectrum1.png"))
        plt.close(fig)
        #sys.exit()
        
        f1 = open(os.path.join(output_dir, 'areas.txt'), 'w')
        f1.write('k num_components f_mode_area\n')
        
        results = []
        k_max_ = ks[ks < k_max][-1]
        
        params = []
        bounds = []
        
        print(data.shape, nus.shape)
        nus_filtered = np.asarray(nus)
        fltr = (nus_filtered > 2.) * (nus_filtered <10.)
        nus_filtered = nus_filtered[fltr]
        data = data[fltr, :, :]
        
        data_mask = np.zeros_like(data, dtype=int)
        
        coords = np.empty((len(nus_filtered), len(ks), len(ks), 4))
        nu_k_scale = (nus[-1]-nus[0])/(ks[-1]-ks[0])
        k_grid = np.transpose([np.tile(ks, len(ks)), np.repeat(ks, len(ks))])
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
            '''
            for j1 in range(len(ks)):
                k1 = ks[j1]
                k12 = k1**2
                for j2 in range(len(ks)):
                    k2 = ks[j2]
                    k22 = k2**2
                    #phi = np.arctan2(k2, k1)
                    k122 = k12 + k22
                    k = np.sqrt(k122)
                    #theta = np.arctan2(nu, k)
                    coords[i, j1, j2] = [nu, k1*nu_k_scale, k2*nu_k_scale, k]#np.sqrt(nu2 + k122), phi theta]
            '''
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
        
        for k_ind1 in range(coords.shape[2]):
            for k_ind2 in range(coords.shape[3]):
                _, k1, k2, k = coords[0, k_ind1, k_ind2]
                if k >= k_min and k <= k_max_:
                    data_mask[:, k_ind1, k_ind2] = 1
                    num_components = get_num_components(k)
                    
                    nus_ = coords[:, k_ind1, k_ind2, 0]
                    for i in np.arange(num_components):
                        alpha_prior0 = get_alpha_prior(i, k)
                        params.append(alpha_prior0)
                        print("alpha_prior", alpha_prior, i, k)
                        bounds.append((alpha_prior0-.5 , alpha_prior0+.5))

                        params.append(k1)
                        bounds.append((k1-.5 , k1+.5))

                        params.append(k2)
                        bounds.append((k2-.5 , k2+.5))

                        beta_prior = 0.04#.2/num_components
                        params.append(beta_prior)
                        #params.append(1./100)
                        bounds.append((1e-10 , 2*beta_prior))

                        params.append(scale_prior)
                        bounds.append((1e-10, 10.))

                        mode_info.append((i == 0, np.argmin(np.abs(nus_ - alpha_prior)), k_ind1, k_ind2))
                                                    
    
        #######################################################################    
        
        #y = y[:, k_indices]
        #k = k[k_indices]
                
        alphas_est = []
        betas_est = []
        ws_est = []
        scales_est = []
            
        min_loglik = None
        def lik_fn(params):
            '''
            alphas = []
            betas = []
            ws = []
            scales = []
            i = 0
            for k_i in np.arange(len(k_inds)):
                num_components = all_num_components[k_i]
                alphas.append(params[i:i+num_components])
                i += num_components
                betas.append(params[i:i+num_components])
                i += num_components
                ws.append(params[i:i+num_w])
                i += num_w
                scales.append(params[i:i+num_components])
                i += num_components
            assert(len(scales) == len(alphas))
            '''
            data_fitted = fit(coords, params)
            return -calc_loglik(data_fitted, data, data_mask, true_sigma)
        
        '''
        min_res = None
        for trial_no in np.arange(0, num_optimizations):                    
            print("params", params)
                
            #initial_lik = lik_fn(params)
            #res = scipy.optimize.minimize(lik_fn, params, method='CG', jac=None, options={'disp': True, 'gtol':initial_lik*1e-7})#, 'eps':.1})
            res = scipy.optimize.minimize(lik_fn, params, method='L-BFGS-B', jac=None, bounds=bounds, options={'disp': True, 'gtol':1e-7})
            loglik = res['fun']
            if min_loglik is None or loglik < min_loglik:
                min_loglik = loglik
                min_res = res['x']
        params_est = min_res
        '''
        params_est = params
        '''
        i = 0
        for k_i in np.arange(len(k_inds)):
            num_components = all_num_components[k_i]
            alphas_est.append(min_res[i:i+num_components])
            i += num_components
            betas_est.append(min_res[i:i+num_components])
            i += num_components
            ws_est.append(min_res[i:i+num_w])
            i += num_w
            scales_est.append(min_res[i:i+num_components])
            i += num_components
        '''
        #plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')
        data_fitted = fit(coords, params_est)
        b = bic(calc_loglik(data_fitted, data, data_mask, true_sigma), np.sum(data_mask), len(params))
        print("BIC", b)
        
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
                    params_[nu_ind][mode_index].append(params_est[i*num_params])

        styles = ["k-", "b-", "g-", "r-", "m-"]
        for i in range(len(params_)):
            fig = plot.plot(nrows=1, ncols=1, size=plot.default_size(data.shape[1], data.shape[0]))
            fig.colormap(np.log(data[i, :, :]), cmap_name="gnuplot", show_colorbar=True)
            for mode_index in range(5):
                plot_mode(params_[i][mode_index], nu_k_scale, fig, styles[mode_index])
            fig.save(f"ring_diagram{i}.png")
                    
            
            
        '''
        for k_i in np.arange(len(k_indices)):
            k_index = k_indices[k_i]
            num_components = all_num_components[k_i]
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(x, y[:, k_i], 'x', label='data')
            ax.plot(x, y_mean_est[:, k_i], label='estimated regression line', lw=3., c='r')
            ax.set_title('Num. clusters ' + str(num_components))
            ax.legend(loc=0)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.savefig(f"{output_dir}/fit{k_index}_{num_components}.png")
            plt.close(fig)
        '''

        '''        
        #if b < min_bic:
        min_bic = b
        opt_alphas = alphas_est
        opt_betas = betas_est
        opt_ws = ws_est
        opt_scales = scales_est
        opt_num_components = num_components
        
        #scale =  np.sum(y)*x_range/len(y)/opt_num_components
        
        print("alphas", opt_alphas)
        print("betas", opt_betas)
        print("ws", opt_ws)
        print("scales", opt_scales)

        #######################################################################
        # Plot results
        levels = np.linspace(np.min(np.log(y))+2, np.max(np.log(y))-2, 200)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig2, ax2 = plt.subplots(nrows=1, ncols=1)
        ax.contour(k, x, np.log(y), levels=levels)
        styles = ["k-", "b-", "g-", "r-", "m-"]
        for i in range(5):
            alphas_to_fit, betas_to_fit, scales_to_fit, k1 = get_mode_params(i, opt_alphas, opt_betas, opt_scales, k)
            alpha_coefs = np.polyfit(k1, alphas_to_fit, alpha_pol_degree)
            beta_coefs = np.polyfit(k1, betas_to_fit, beta_pol_degree)
            scales_fitted = smooth(x=scales_to_fit, window_len=scale_smooth_win)[scale_smooth_win//2:-(scale_smooth_win//2)]

            #scale_coefs = np.polyfit(k1, scales_to_fit, scale_pol_degree)
            #print("coefs", alpha_coefs, beta_coefs)
            
            alphas_fitted = fit(k, alpha_coefs)
            betas_fitted = fit(k, beta_coefs)
            #scales_fitted = fit(k, scale_coefs)
            ax.plot(k, alphas_fitted, "k-")
            ax.plot(k, alphas_fitted - betas_fitted, "k:")
            ax.plot(k, alphas_fitted + betas_fitted, "k:")

            ax2.plot(k1, scales_fitted, styles[i], label=f"Mode {i+1}")
            ax2.legend(loc=0)
            ax2.set_xlabel(r'$k$')

        fig.savefig(os.path.join(output_dir, "spectrum2.png"))
        plt.close(fig)

        fig2.savefig(os.path.join(output_dir, "scales.png"))
        plt.close(fig2)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        styles = ["k-", "b-", "g-", "r-"]
        for j in range(len(opt_ws[0])):
            w_coefs = np.polyfit(k, np.asarray(opt_ws)[:, j], w_pol_degree)
            ws_fitted = fit(k, w_coefs)
            ax.plot(k, ws_fitted, styles[j], label=f"Coef. {j+1}")
            ax.set_xlabel(r'$k$')
            ax.legend(loc=0)

        fig.savefig(os.path.join(output_dir, "baseline_weights.png"))
        plt.close(fig)
        
        #######################################################################

        y_mean_est = calc_y(x, opt_alphas, opt_betas, opt_ws, opt_scales, k)
        
        colors = ['blue', 'red', 'green', 'peru', 'purple']

        for k_i in np.arange(len(k_indices)):
            k_index = k_indices[k_i]
            fig, ax = plt.subplots(nrows=1, ncols=1)
            plt.figure(figsize=(7, 7))
            ax.plot(x, y[:, k_i], 'x', label='data')
            #plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')
            ax.plot(x, y_mean_est[:, k_i], label='estimated regression line', lw=3., c='r')
                
            areas, ranges = find_areas(x, y[:, k_i], opt_alphas[k_i], opt_betas[k_i], opt_ws[k_i], opt_scales[k_i], true_sigma, k[k_i])
            for i in np.arange(len(areas)):
                ax.axvspan(ranges[i, 0], ranges[i, 1], alpha=0.5, color=colors[i])
            
            k_value = k[k_i]
            ax.set_title("Spectrum at k=" + str(k_value) + ", num. components=" + str(opt_num_components))
            ax.legend(loc=0)
            ax.set_xlabel(r'$\nu$')
            ax.set_ylabel('Amplitude')
            fig.savefig(os.path.join(output_dir, f"areas{k_index}.png"))
    
            plt.close(fig)
        
            f1.write('%s %s %s' % (str(k_value), opt_num_components, areas[0]) + "\n")
            print("Lowest BIC", min_bic)
            print("Num components", opt_num_components)
            print("Areas", areas)
            f1.flush()
            results.append([k_value, opt_num_components, areas[0]])
            
        
        f1.close()
        
        results = np.asarray(results)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.figure(figsize=(7, 7))
        ax.plot(results[:, 0], results[:, 2], 'k-')
        ax.set_xlabel(r'$k$')
        ax.set_ylabel('F-mode area')
        fig.savefig(os.path.join(output_dir, "areas.png"))

        plt.close(fig)
        '''
        
