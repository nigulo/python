import matplotlib as mpl
mpl.use('nbAgg')
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
import plot
from astropy.io import fits
import numpy as np
import numpy.fft as fft
import pickle


def is_odd(num):
    return num & 0x1


def downsample(data, factors=(3, 2)):
    print(np.min(data), np.max(data), np.mean(data))
    data = fft.fftshift(fft.fftn(data))

    '''
    import misc
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
    
def load(f):
    if not os.path.exists(f):
        return None
    return pickle.load(open(f, 'rb'))

def save(obj, f):
    with open(f, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
        
def find_mode_params(r, histogram, ks, nu):
    if nu < 3:
        smooth_win = 100
    elif nu < 4:
        smooth_win = 200
    else:
        smooth_win = 300
    mode_guess_ind = np.argmin(np.abs(ks - r))
    mode_guess_amplitude = histogram[mode_guess_ind]
    max_amplitude = 0
    max_ind = mode_guess_ind
    for k_ind in range(mode_guess_ind, -1, -1):
        if histogram[k_ind] > max_amplitude and abs(ks[k_ind] - ks[max_ind]) < smooth_win:
            max_amplitude = histogram[k_ind]
            max_ind = k_ind

    assert(max_ind <= mode_guess_ind)

    min_amplitude = max_amplitude
    min_ind = max_ind

    for k_ind in range(max_ind, -1, -1):
        if histogram[k_ind] < min_amplitude and abs(ks[k_ind] - ks[min_ind]) < smooth_win:
            min_amplitude = histogram[k_ind]
            min_ind = k_ind

    assert(min_ind < max_ind)
    return ks[min_ind], ks[max_ind] + 2*(ks[max_ind] - ks[min_ind]), ks[max_ind]   
        
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
    
    nu_min = 2
    nu_max = 10
    
    k_min = 700
    k_max = 3000#sys.maxsize
    #if len(sys.argv) > 1:
    #    k_min = float(sys.argv[1])
    
    #if len(sys.argv) > 2:
    #    k_max = float(sys.argv[2])
        
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
            #data = downsample(data)
            
            ks = np.linspace(-3600, 3600, data.shape[1])
            #nus = np.linspace(-11.076389, 11.076389, data.shape[0])
            

            hdul.close()

            ring_radii, nus, fltr = load("ring_radii.dat")

            k_len = data.shape[1]
            k_len_half = k_len//2 + is_odd(k_len)
                        
            nrows = 6
            ncols = 6
            num_plots = len(nus)#nrows*ncols
            data = data[fltr]
            plot_step = len(nus)//num_plots
            ks_hist = np.linspace(0, ks[-1], k_len_half)
            
            #fig = plot.plot(nrows=nrows, ncols=ncols, size=plot.default_size(data.shape[1]//3, data.shape[2]//3), smart_axis="x")
                        
            nu_start, nu_end = 2, 5


            num_plots_done = 0
            for nu_ind in range(0, len(nus), plot_step):
                if nus[nu_ind] <= nu_start or nus[nu_ind] > nu_end:
                    continue
                if num_plots_done >= num_plots:
                    break
                num_plots_done += 1
                data_slice = data[nu_ind]
                histogram = np.zeros(k_len_half)
                for kx_ind in range(k_len):
                    for ky_ind in range(k_len):
                        dist = int(np.sqrt((kx_ind - k_len_half)**2 + (ky_ind - k_len_half)**2))
                        if dist < len(histogram):
                            histogram[dist] += data_slice[kx_ind, ky_ind]
                fig = plot.plot(nrows=1, ncols=1, size=plot.default_size(300, 200))
                
                ax_index = None#fig.get_current_ax()
                
                fig.set_axis_title(r"$\nu=" + str(nus[nu_ind]) + "$", ax_index=ax_index)
                fig.plot(ks_hist[1:], histogram[1:], "k-", ax_index=ax_index)
                
                #max_val = np.max(histogram[1:])
                #max_val = np.max(histogram[1:])
                
                (_, _), (y_min, y_max) = fig.get_axis_limits(ax_index)
                
                for mode_index in [0]:#ring_radii.keys():
                    mode_radii = np.asarray(ring_radii[mode_index])
                    r = mode_radii[nu_ind]#nus_filtered == abs(nus[nu_ind])]
                    assert(r > 0)
                    left, right, peak = find_mode_params(r, histogram, ks_hist, nus[nu_ind])
                    fig.plot([r, r], [y_min, y_max], "k--", ax_index=ax_index)
                    fig.plot([peak, peak], [y_min, y_max], "r--", ax_index=ax_index)
                    fig.plot([left, left], [y_min, y_max], "g--", ax_index=ax_index)
                    fig.plot([right, right], [y_min, y_max], "b--", ax_index=ax_index)
                    
                #fig.next_ax()    

                fig.save(os.path.join(output_dir, f"histogram{nu_ind}.png"))
            #fig.save(os.path.join(output_dir, f"histograms.png"))
                        
                    

            
            
