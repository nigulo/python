import matplotlib as mpl
mpl.use('Agg')
import sys
import os
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

ny = 300
map_scale = 0.05
#k = np.arange(ny)
#k = (k-((ny-1)/2))*(2.*180./(ny*map_scale))
#k0 = np.where(k == np.min(np.abs(k)))[0][0]
#k1 = np.where(k == np.max(np.abs(k)))[0][0]
#k = k[k0:k1+1]

k = np.linspace(0, 3600, 151)

print(k)

cadence = 45.0
nf = 641
#k = np.linspace(0, 4500, 193)
omega = np.arange((nf-1)/2+1)
omega = omega*2.*np.pi/(nf*cadence/1000.)
nu = omega/(2.*np.pi)
#nu = np.linspace(0, 11.076389, 320)

nu = np.linspace(0, 11.076389, 320)
print(nu)

k_min = 700
k_max = 1500#sys.maxsize
#if len(sys.argv) > 1:
#    k_min = float(sys.argv[1])

#if len(sys.argv) > 2:
#    k_max = float(sys.argv[2])

num_w = 1

num_samples = 1000
num_cores = 6
colors = ['blue', 'red', 'green', 'peru', 'purple']

num_optimizations = 1

mode_priors = np.genfromtxt("GVconst.txt")

# F-mode = 0
# P modes = 1 ... 
def get_alpha_prior(i, k):
    for row in mode_priors:
        k_start = row[0]
        k_end = row[1]
        if k >= k_start and k < k_end:
            if i >= row[2]:
                raise "No mode prior found"
            return row[5 + i * 3]

def get_num_components(k):
    for row in mode_priors:
        k_start = row[0]
        k_end = row[1]
        if k >= k_start and k < k_end:
            return int(row[2])
    raise "No numbwr of components found for k=" + str(k)

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

def calc_y(x, alphas, betas, ws, scales):
    ys = np.empty((len(x), len(alphas)))
    for j in range(len(alphas)):
        y = np.zeros_like(x)
        for i in np.arange(len(ws[j])):
            y += ws[j][i]*x**i
        for i in np.arange(len(alphas[j])):
            alpha = alphas[j][i]
            beta = betas[j][i]
            y += 1./(np.pi*beta*(1+((x-alpha)/beta)**2))
        ys[:, j] = y*scales[j]
    return ys

def calc_loglik(y, y_true, sigma):
    loglik = -0.5 * np.sum((y - y_true)**2/sigma) - 0.5*np.log(sigma) - 0.5*np.log(2.0*np.pi)
    return loglik        
    
def bic(loglik, n, k):
    return np.log(n)*k - 2.*loglik

def find_areas(x, y, alphas, betas, ws, scale, noise_std):
    ###########################################################################
    # Remove noise 
    #y_base = calc_y(x, [], [], ws, scale)
    #y -= y_base
    #inds = np.where(y > noise_std)[0]
    #y = y[inds]
    #x = x[inds]
    ###########################################################################
    y_base = calc_y(x, [[]], [[]], [ws], [scale])
    ys_fit = []
    num_components = len(alphas)
    for i in np.arange(num_components):
        y_fit = calc_y(x, [alphas[i:i+1]], [betas[i:i+1]], [ws], [scale])
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

def smooth(x=None, y=None):
    if x is not None:
        x = x[:-2]+x[1:-1]+x[2:]
        x /= 3
    
    if y is not None:
        y = y[:-2]+y[1:-1]+y[2:]
        y /= 3

    return x, y

    
def get_noise_var(data, k, nu):
    k_indices = np.where(np.logical_and(k >= 3500, k <= 4500))[0]
    nu_indices = np.where(np.logical_and(nu >= 2, nu <= 4))[0]
    y = data[nu_indices]
    y = y[:, k_indices]

    _, y = smooth(None, y)
    return np.var(y)

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
        print(data.shape, nf, ny)
        assert(data.shape[0] == nf)
        assert(data.shape[1] == ny)
        assert(data.shape[2] == ny)
        data = fft.fftn(data)/np.product(data.shape)
        data = data[:data.shape[0]//2, :, :]
        data = np.real(data*np.conj(data))
        
        #######################################################################
        d1 = fft.fftshift(data[100])
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
        fig.savefig(os.path.join(output_dir, "spectrum1.png"))
        plt.close(fig)
        #######################################################################
        
        
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
        
        hdul.close()
    
        levels = np.linspace(np.min(np.log(data))+2, np.max(np.log(data))-2, 200)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.contour(k, nu, np.log(data), levels=levels)
        fig.savefig(os.path.join(output_dir, "spectrum2.png"))
        plt.close(fig)
        
        f1 = open(os.path.join(output_dir, 'areas.txt'), 'w')
        f1.write('k num_components f_mode_area\n')
        
        results = []
        k_index = np.min(np.where(k >= k_min)[0])
        k_max_ = k[k < k_max][-1]
        
        params = []
        bounds = []
        
        x = np.asarray(nu, dtype='float')
        inds = np.where(x > 2.)[0]
        x = x[inds]
        y = data[inds, :]
        inds = np.where(x < 10.)[0]
        x = x[inds]
        y = data[inds, :]
        x_range = max(x) - min(x)
        
        all_num_components = []
        scales = []
        k_indices = []
        
        while k_index < data.shape[1]:
            y1 = data[:, k_index]
            
            if np.min(k[k_index:]) > k_max_:
                break

            num_components = get_num_components(k[k_index])            
            all_num_components.append(num_components)
            scales.append(np.sum(y1)*x_range/len(y1)/num_components)
            k_indices.append(k_index)
                            
            for i in np.arange(num_components):
                alpha_prior = get_alpha_prior(i, k[k_index])
                params.append(alpha_prior)
                print("alpha_prior", alpha_prior, i, k[k_index])
                bounds.append((alpha_prior-.5 , alpha_prior+.5))
            for i in np.arange(num_components):
                beta_prior = .2/num_components
                params.append(beta_prior)
                #params.append(1./100)
                bounds.append((.0001 , 2*beta_prior))
            for i in np.arange(num_w):
                params.append(0.)
                bounds.append((-100 , 100))

            k_index += 1
        #######################################################################    
        
        noise_var = get_noise_var(data, k, nu)
        
        sig_var = np.var(y) - noise_var
        true_sigma = np.sqrt(noise_var)

        y = y[:, k_indices]
        
        print("scales", scales)
        
        alphas_est = []
        betas_est = []
        ws_est = []
            
        min_loglik = None
        def lik_fn(params):
            alphas = []
            betas = []
            ws = []
            i = 0
            for k_i in np.arange(len(k_indices)):
                num_components = all_num_components[k_i]
                alphas.append(params[i:i+num_components])
                i += num_components
                betas.append(params[i:i+num_components])
                i += num_components
                ws.append(params[i:i+num_w])
                i += num_w
            assert(len(scales) == len(alphas))
            
            y_mean_est = calc_y(x, alphas, betas, ws, scales)
            return -calc_loglik(y_mean_est, y, true_sigma)
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
        i = 0
        for k_i in np.arange(len(k_indices)):
            num_components = all_num_components[k_i]
            alphas_est.append(min_res[i:i+num_components])
            i += num_components
            betas_est.append(min_res[i:i+num_components])
            i += num_components
            ws_est.append(min_res[i:i+num_w])
            i += num_w

        #plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')
        y_mean_est = calc_y(x, alphas_est, betas_est, ws_est, scales)
        b = bic(calc_loglik(y_mean_est, y, true_sigma), len(y), len(k_indices)*(2*num_components + num_w))
        print("BIC", b)
        
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
        
        #if b < min_bic:
        min_bic = b
        opt_alphas = alphas_est
        opt_betas = betas_est
        opt_ws = ws_est
        opt_num_components = num_components
        
        scale =  np.sum(y)*x_range/len(y)/opt_num_components
        
        print("alphas", opt_alphas)
        print("betas", opt_betas)
        print("ws", opt_ws)
        
        for k_i in np.arange(len(k_indices)):
            k_index = k_indices[k_i]
            fig, ax = plt.subplots(nrows=1, ncols=1)
            plt.figure(figsize=(7, 7))
            ax.plot(x, y[:, k_i], 'x', label='data')
            #plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')
            y_mean_est = calc_y(x, opt_alphas, opt_betas, opt_ws, scales)
            ax.plot(x, y_mean_est[:, k_i], label='estimated regression line', lw=3., c='r')
                
            areas, ranges = find_areas(x, y[:, k_i], opt_alphas[k_i], opt_betas[k_i], opt_ws[k_i], scales[k_i], true_sigma)
            for i in np.arange(len(areas)):
                ax.axvspan(ranges[i, 0], ranges[i, 1], alpha=0.5, color=colors[i])
            
            k_value = k[k_index]
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
