import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
import pandas as pd
from scipy import stats

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
    fieldwidths = (10, 12, 18, 6, 3, 9, 6)  # negative widths represent ignored padding fields
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
            try:
                #p_rot = float(fields[7].strip())
                p_rot_err = float(fields[5].strip())
            except ValueError:
                p_rot = None
            if p_rot != None and p_rot != None:
                rot_periods[star] = (p_rot, p_rot_err)
    rot_periods["SUN"] = (26.47, 0.21)
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

def get_seasons(dat, num_years, seasonal):
    seasons = list()
    #res = list()
    last_t = float('-inf')
    season_start = float('-inf')
    season = list()
    for t, y in dat:
        if (seasonal and t - last_t > num_years/3) or t - season_start >= num_years:
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

def get_seasonal_noise_var(t, y, per_point=True, remove_trend=False, mode=0, num_years=1.0, t_out=None):
    if t_out is None:
        t_out = t
    total_var = np.var(y)
    seasons = get_seasons(zip(t, y), num_years, num_years==1.0)
    if per_point:
        noise_var = np.zeros(len(t_out))
    else: 
        noise_var = np.zeros(np.shape(seasons)[0])
    i = 0
    max_var = 0
    for season in seasons:
        if np.shape(season[:,1])[0] < 10:
            if mode == 1:
                var = -1#total_var # Is it good idea?
            else:
                var = total_var # Is it good idea?
        else:
            y_season = season[:,1]
            if remove_trend:
                slope, intercept, r_value, p_value, std_err = stats.linregress(season[:,0], y_season)
                #print "slope, intercept", slope, intercept
                fit_trend = season[:,0] * slope + intercept
                y_season = y_season - fit_trend
                
            var = np.var(y_season)
        max_var=max(max_var, var)
        if per_point:
            season_len = np.shape(season)[0]
            j = 0
            season_end = max(season[:,0])
            while i + j < len(t_out) and t_out[i + j] <= season_end:
                noise_var[i + j] = var
                j += 1
            i += j
        else:
            noise_var[i] = var
            i += 1
    assert(i == len(noise_var))
    if mode == 1:
        for j in np.arange(0, len(noise_var)):
            if noise_var[j] < 0:
                jj = j - 1
                while jj >= 0 and noise_var[jj] < 0:
                    jj -= 1
                if jj < 0:
                    jj = j + 1
                    while jj < len(noise_var) and noise_var[jj] < 0:
                        jj += 1
                if jj < len(t):
                    noise_var[j] = noise_var[jj]
                else:    
                    noise_var[j] = max_var
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
    if len(maxima_inds) > 0:
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
                print("Something wrong!!!")
            noise_res.append(noise[start])
            start = i
    t_res.append(np.mean(t[start:]))
    y_res.append(np.mean(y[start:]))
    if not all(noise[start:] == noise[start]):
        print("Something wrong!!!")
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
                cycles.append((cyc*365.25, cyc_std*365.25, log_bic)) # three sigma
                all_cycles[star] = cycles
    return min_bic, max_bic, all_cycles

def read_gp_cycles(file):
    time_ranges = load_time_ranges()
    data = pd.read_csv(file, names=['star', 'validity', 'cyc', 'sigma', 'bic', 'spread', 'ell'], header=None, dtype=None, sep='\s+', engine='python').as_matrix()
    gp_cycles = dict()
    #for [star, count, count_used, validity, cyc, std, normality, bic, bic_diff] in data:
    for [star, validity, cyc, std, bic, spread, ell] in data:
        if star == 'SUN':
            star = 'Sun'
        if not gp_cycles.has_key(star):
            gp_cycles[star] = list()
        all_cycles = gp_cycles[star]
        cycles = list()
        if not np.isnan(cyc) and cyc < time_ranges[star] / 1.5 and std < cyc/4:
            cycles.append(cyc)
            cycles.append(std)
            cycles.append(bic)
            cycles.append(spread)
        all_cycles.append(np.asarray(cycles))
    return gp_cycles

def load_time_ranges():
    time_ranges_data = np.genfromtxt("time_ranges.dat", usecols=(0,1,2), dtype=None)
    time_ranges = dict()
    for [star, time_range, end_time] in time_ranges_data:
        if star == 'SUN':
            star = 'Sun'
        time_ranges[star] = time_range
    return time_ranges
    
def inducing_points_to_front(t, y):
    seasons = get_seasons(zip(t, y), 1.0, True)
    noise_var = get_seasonal_noise_var(t, y, per_point = False)
    t_out = np.zeros(len(t))
    y_out = np.zeros(len(y))
    noise_out = np.zeros(len(t))
    i1 = 0
    i2 = len(seasons)
    season_index = 0
    for season in seasons:
        season_mid = (season[0,0] + season[-1,0])/2
        min_diff = abs(season[0,0] - season_mid)
        min_index = 0
        j = 0
        for [t, y] in season:
            if abs(t - season_mid) < min_diff:
                min_diff = abs(t - season_mid)
                min_index = j
            j += 1
        j = 0
        for [t, y] in season:
            if j == min_index:
                t_out[i1] = t
                y_out[i1] = y
                noise_out[i1] = noise_var[season_index]
                i1 += 1
            else:
                t_out[i2] = t
                y_out[i2] = y
                noise_out[i2] = noise_var[season_index]
                i2 += 1
            j += 1
        season_index += 1
    assert(i1 == len(seasons))
    return t_out, y_out, noise_out

def downsample(t, y, noise, min_time_diff=1.0/365.25, average=False):
    t_out = list()
    y_out = list()
    counts = list()
    noise_out = list()
    for i in np.arange(0, len(t)):
        found = False
        for j in np.arange(0, len(t_out)):
            t_out_j = t_out[j]
            if abs(t[i] - t_out_j) <= min_time_diff:
                found = True
                if average:
                    y_out[j] += y[i]
                    counts[j] += 1             
                break
        if not found:
            t_out.append(t[i])
            y_out.append(y[i])
            noise_out.append(noise[i])
            counts.append(1.0)
    y_out = np.asarray(y_out)
    if average:
        y_out /= counts
    return np.asarray(t_out), y_out, np.asarray(noise_out)
    
