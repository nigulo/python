import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
import pandas as pd

###############################################################################
# Load rotational periods
try:
    from itertools import izip_longest  # added in Py 2.6
except ImportError:
    from itertools import zip_longest as izip_longest  # name change in Py 3.x

try:
    from itertools import accumulate  # added in Py 3.2
except ImportError:
    def accumulate(iterable):
        'Return running totals (simplified version).'
        total = next(iterable)
        yield total
        for value in iterable:
            total += value
            yield total

def make_parser(fieldwidths):
    cuts = tuple(cut for cut in accumulate(abs(fw) for fw in fieldwidths))
    pads = tuple(fw < 0 for fw in fieldwidths) # bool values for padding fields
    flds = tuple(izip_longest(pads, (0,)+cuts, cuts))[:-1]  # ignore final one
    parse = lambda line: tuple(line[i:j] for pad, i, j in flds if not pad)
    # optional informational function attributes
    parse.size = sum(abs(fw) for fw in fieldwidths)
    parse.fmtstring = ' '.join('{}{}'.format(abs(fw), 'x' if fw < 0 else 's')
                                                for fw in fieldwidths)
    return parse


def load_rot_periods(path=""):
    #fieldwidths = (10, 8, 8, 6, 8, 9, 11, 9, 4)  # negative widths represent ignored padding fields
    fieldwidths = (10, 12, 18, 6, 12, 6)  # negative widths represent ignored padding fields
    parse = make_parser(fieldwidths)
    rot_periods = dict()
    line_no = 0
    #with open(path+"mwo-rhk.dat", "r") as ins:
    with open(path+"mwo-gpresults.dat", "r") as ins:
        for line in ins:
            if line_no < 5:
                line_no += 1
                continue
            fields = parse(line)
            star = fields[0].strip()
            star = star.replace(' ', '')
            star = star.upper()
            try:
                #p_rot = float(fields[7].strip())
                p_rot = float(fields[3].strip())
            except ValueError:
                p_rot = None
            if p_rot != None:
                rot_periods[star] = p_rot
    return rot_periods

def load_r_hk(path=""):
    i = 0
    data = dict()
    #fieldwidths = (10, 8, 8, 6, 8, 9, 11, 9, 4)  # negative widths represent ignored padding fields
    fieldwidths = (10, 8, 8, 6, 8, 9, 11, 9, 4)  # negative widths represent ignored padding fields
    parse = make_parser(fieldwidths)
    #with open(path+"mwo-rhk.dat", "r") as ins:
    with open("mwo-rhk.dat", "r") as ins:
        for line in ins:
            if i < 5:
                i += 1
                continue
            fields = parse(line)
            star = fields[0].strip()
            star = star.replace(' ', '')
            star = star.upper()
            try:
                r_hk = float(fields[6].strip())
            except ValueError:
                r_hk = None
            if r_hk != None:
                data[star] = r_hk
    return data

def load_ro(path=""):
    i = 0
    data = dict()
    #fieldwidths = (10, 8, 8, 6, 8, 9, 11, 9, 4)  # negative widths represent ignored padding fields
    fieldwidths = (10, 8, 8, 6, 8, 9, 11, 9, 4)  # negative widths represent ignored padding fields
    parse = make_parser(fieldwidths)
    #with open(path+"mwo-rhk.dat", "r") as ins:
    with open("mwo-rhk.dat", "r") as ins:
        for line in ins:
            if i < 5:
                i += 1
                continue
            fields = parse(line)
            star = fields[0].strip()
            star = star.replace(' ', '')
            star = star.upper()
            try:
                bmv = float(fields[4].strip())
            except ValueError:
                bmv = 0.0
            try:
                p_rot = float(fields[7].strip())
            except ValueError:
                p_rot = None
            ro = None
            if p_rot != None and bmv != None:
                if bmv >= 1.0:
                    tau = 25.0
                else:
                    tau = np.power(10.0, -3.33 + 15.382*bmv - 20.063*bmv**2 + 12.540*bmv**3 - 3.1466*bmv**4)
                ro = np.log10(4*np.pi*tau/p_rot)
                data[star] = ro
    return data
    
def load_spec_types(path=""):
    fieldwidths = (10, 8, 10, 6, 8, 4)  # negative widths represent ignored padding fields
    parse = make_parser(fieldwidths)
    spec_types = dict()
    line_no = 0
    with open(path+"mwo-hr.dat", "r") as ins:
        for line in ins:
            if line_no < 19:
                line_no += 1
                continue
            fields = parse(line)
            star = fields[0].strip()
            star = star.replace(' ', '')
            star = star.upper()
            spec_type = fields[2].strip()
            spec_types[star] = spec_type
    return spec_types
###############################################################################

#rot_period_data = np.genfromtxt("periods.txt", usecols=(0,1), dtype=None)
#for [star, rot_period] in rot_period_data:
#    rot_periods[star] = rot_period

def get_seasons(dat, num_days, seasonal):
    seasons = list()
    #res = list()
    last_t = float('-inf')
    season_start = float('-inf')
    season = list()
    for t, y in dat:
        if (seasonal and t - last_t > num_days/3) or t - season_start >= num_days:
            if np.shape(season)[0] > 0:
                #res.append([(last_t + season_start)/2, season_mean/np.shape(season)[0]])
                seasons.append(np.asarray(season))
            season_start = t
            season = list()
        last_t = t
        season.append([t, y])
    if np.shape(season)[0] > 0:
        #res.append([(last_t + season_start)/2, season_mean/np.shape(season)[0]])
        seasons.append(np.asarray(season))
    return seasons


def resample_seasons(seasons):
    indices = np.random.choice(np.shape(seasons)[0], np.shape(seasons)[0], replace=True, p=None)
    resampled_seasons=list()
    for i in np.arange(0, len(seasons)):
        season = seasons[i]
        season_std = np.std(season[:,1])
        season_indices = np.random.choice(len(seasons[indices[i]]), len(season), replace=True, p=None)
        resampled_season = seasons[indices[i]][season_indices]
        resampled_season[:,0] = season[:,0]
        resampled_season_mean = np.mean(resampled_season[:,1])
        resampled_season_std = np.std(resampled_season[:,1])
        if resampled_season_std != 0: # Not right, but what can we do in case of seasons with low number of observations
            resampled_season[:,1] = resampled_season_mean + (resampled_season[:,1] - resampled_season_mean) * season_std / resampled_season_std
        resampled_seasons.append(resampled_season)
    return resampled_seasons

def get_seasonal_noise_var(t, y, per_point=True):
    total_var = np.var(y)
    seasons = get_seasons(zip(t, y), 1.0, True)
    if per_point:
        noise_var = np.zeros(len(t))
    else: 
        noise_var = np.zeros(np.shape(seasons)[0])
    i = 0
    for season in seasons:
        if np.shape(season[:,1])[0] < 10:
            var = total_var # Is it good idea?
        else:
            var = np.var(season[:,1])
        if per_point:
            season_len = np.shape(season)[0]
            for j in np.arange(i, i + season_len):
                noise_var[j] = var
            i += season_len
        else:
            noise_var[i] = var
            i += 1
    assert(i == len(noise_var))
    return noise_var

def get_test_point_noise_var(t, y, t_test, sliding=False, season_length = 1.0):
    noise_var = np.zeros(len(t_test))
    if sliding:
        i = 0
        for t1 in t_test:
            indices1 = np.where(t >= t1 - season_length)[0]
            y_window = y[indices1]
            t_window = t[indices1]
            indices2 = np.where(t_window <= t1 + season_length)[0]
            y_window = y_window[indices2]
            while np.shape(y_window)[0] < 10:
                season_length *= 1.5
                indices1 = np.where(t >= t1 - season_length)[0]
                y_window = y[indices1]
                t_window = t[indices1]
                indices2 = np.where(t_window <= t1 + season_length)[0]
                y_window = y_window[indices2]
            noise_var[i] = np.var(y_window)
            i += 1
        assert(i == len(noise_var))
    else:
        seasons = get_seasons(zip(t, y), season_length, True)
        seasonal_noise_var = get_seasonal_noise_var(t, y, False)    
        seasons_with_noise = zip(seasons, seasonal_noise_var)
        i = 0
        for ti in t_test:
            for j in np.arange(0, len(seasons_with_noise)):
                season0, var = seasons_with_noise[j]
                if j < len(seasons_with_noise) - 1:
                    season1, _ = seasons_with_noise[j+1]
                    if min(season0[:,0]) <= ti and min(season1[:,0]) >= ti:
                        noise_var[i] = var
                        break
                else:
                    noise_var[i] = var
                    
            i += 1
        assert(i == len(noise_var))
    return noise_var

'''
    Gets the seasonal means (one per each season)
'''
def get_seasonal_means(t, y):
    seasons = get_seasons(zip(t, y), 1.0, True)
    means = np.zeros((np.shape(seasons)[0], 2))
    i = 0
    for season in seasons:
        means[i] = [np.mean(season[:,0]), np.mean(season[:,1])]
        i += 1
    return means


def get_seasonal_means_per_point(t, y):
    seasons = get_seasons(zip(t, y), 1.0, True)
    means = np.zeros(len(y))
    i = 0
    for season in seasons:
        season_mean = np.mean(np.mean(season[:,1]))
        for j in np.arange(0, len(season[:,1])):
            means[i] = season_mean
            i += 1
    return means

def find_local_maxima(x):
    maxima_inds = argrelextrema(x, np.greater_equal)[0]
    maxima_inds = maxima_inds[np.where(maxima_inds > 0)] # Omit leftmost point
    maxima_inds = maxima_inds[np.where(maxima_inds < len(x) - 1)] # Omit rightmost point
    filtered_maxima_inds = list()
    start = 0
    for i in np.arange(1, len(maxima_inds)):
        if maxima_inds[i] != maxima_inds[i-1] + 1:
            filtered_maxima_inds.append(int((maxima_inds[start] + maxima_inds[i-1]))/2)
            start = i
    filtered_maxima_inds.append(int((maxima_inds[start] + maxima_inds[-1]))/2)
    return np.asarray(filtered_maxima_inds)

def daily_averages(t, y, noise):
    t_res = list()
    y_res = list()
    noise_res = list()
    start = 0
    for i in np.arange(1, len(t)):
        if int(t[i]) != int(t[i-1]):
            t_res.append(np.mean(t[start:i]))
            y_res.append(np.mean(y[start:i]))
            if not all(noise[start:i] == noise[start]):
                print "Something wrong!!!"
            noise_res.append(noise[start])
            start = i
    t_res.append(np.mean(t[start:]))
    y_res.append(np.mean(y[start:]))
    if not all(noise[start:] == noise[start]):
        print "Something wrong!!!"
    noise_res.append(noise[start])
    return (np.asarray(t_res), np.asarray(y_res), np.asarray(noise_res))


def estimate_with_se(x, f, num_bootstrap=1000):
    estimate = f(x)
    bs_estimates = np.zeros(num_bootstrap)
    for j in np.arange(0, num_bootstrap):
        x_bs = x[np.random.choice(np.shape(x)[0], np.shape(x)[0], replace=True, p=None)]
        bs_estimates[j] = f(x_bs)
    return (estimate, np.std(bs_estimates))

def mean_with_se(x, num_bootstrap=1000):
    mean = np.mean(x)
    bs_means = np.zeros(num_bootstrap)
    for j in np.arange(0, num_bootstrap):
        x_bs = x[np.random.choice(np.shape(x)[0], np.shape(x)[0], replace=True, p=None)]
        bs_means[j] = np.mean(x_bs)
    return (mean, np.std(bs_means))
    
def mode_with_se(samples, num_bootstrap=1000):
    x_freqs = gaussian_kde(samples)
    x = np.linspace(min(samples), max(samples), 1000)
    #density.covariance_factor = lambda : .25
    #density._compute_covariance()
    mode = x[np.argmax(x_freqs(x))]
    bs_modes = np.zeros(num_bootstrap)
    for j in np.arange(0, num_bootstrap):
        samples_bs = x[np.random.choice(np.shape(x)[0], np.shape(x)[0], replace=True, p=None)]
        x_bs_freqs = gaussian_kde(samples_bs)
        x_bs = np.linspace(min(samples_bs), max(samples_bs), 1000)
        #density.covariance_factor = lambda : .25
        #density._compute_covariance()
        bs_modes[j] = x_bs[np.argmax(x_bs_freqs(x_bs))]
    return (mode, np.std(bs_modes))
   
def read_bglst_cycles(file):
    max_bic = None
    min_bic = None
    all_cycles = dict()
    data = pd.read_csv(file, names=['star', 'f', 'sigma', 'normality', 'bic'], header=None, dtype=None, sep='\s+', engine='python').as_matrix()
    
    #data = np.genfromtxt(file, dtype=None, skip_header=1)
    for [star, f, std, normality, bic] in data:
        #if star == 'SUNALL':
        #    star = 'SUN'
        #print star, cyc, std_2
        if not np.isnan(f):
            if not all_cycles.has_key(star):
                all_cycles[star] = []
            cycles = all_cycles[star]
            log_bic = np.log(bic)
            if max_bic is None or log_bic > max_bic:
                max_bic = log_bic
            if min_bic is None or log_bic < min_bic:
                min_bic = log_bic
                
            cyc = 1.0/f
            
            f_samples = np.random.normal(loc=f, scale=std, size=1000)
            cyc_std = np.std(np.ones(len(f_samples))/f_samples)
            if cyc_std < cyc:
                cycles.append((cyc*365.25, cyc_std*3*365.25, log_bic)) # three sigma
                all_cycles[star] = cycles
    return min_bic, max_bic, all_cycles
