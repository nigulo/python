# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:58:15 2016

@author: nigul
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
import os
import os.path
import subprocess
from scipy import stats
import sys
import random
from scipy.signal import argrelextrema

p_values = np.array([0.01])
max_iters = 100
num_resamples = 1000
num_bootstrap = 500
min_freq = 0
max_freq = 0.5
n_out = 1000
freqs_to_calc = np.linspace(min_freq, max_freq, n_out)
spectra_path = "new_d2_spectra"

star_name = None
if len(sys.argv) > 1:
    star_name = sys.argv[1]

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

def calc_d2(data, freqs, with_smoothing=False):
    t = data[:,0]
    y = data[:,1]
    y2 = y * y
    y2_sum = np.dot(y, y)
    y_sum = np.sum(y)
    n = len(t)
    n2 = n * n
    var = np.var(y)
    res = list()
    for w in freqs:
        if with_smoothing:
            if w == 0.0:
                t_coh = (max(t) - min(t))*10
            else:
                t_coh = 1.0 / w # we use small smoothing with coh_len=1
            i = 0
            disp = 0.0
            norm = 0.0
            for ti in t:
                j = 0
                for tj in t:
                    g = np.exp(-((ti - tj) / t_coh) ** 2) * (1 + 2 * np.cos((ti - tj) * w))
                    disp += g * (y[i] - y[j]) ** 2
                    norm += g
                    j += 1
                i += 1
            res.append([w, disp / norm / var])
        else:
            phase = t * w * 2 * np.pi
            c = np.cos(phase)
            s = np.sin(phase)
            c_sum = np.sum(c)
            s_sum = np.sum(s)
            norm = 1 + 2 * (c_sum ** 2 + s_sum ** 2) / n2
            disp = y2_sum - y_sum / n + 2 * (np.dot(c, y2) * c_sum + np.dot(s, y2) * s_sum - np.dot(c, y) ** 2 - np.dot(s, y) ** 2) / n
            res.append([w, disp / norm / var / n])
    return np.asarray(res)


def configure_d2(star, path, suffix='', num_bootstrap = 0, input_path = None):
    file = star + ".dat"
    data = np.loadtxt(path + "/" + file)
    duration = data[-1,0] - data[0,0]
    if input_path == None:
        input_path = path + "/" + file

    param_file_name = "parameters" + suffix + ".txt"
    if star_name != None:
        param_file_name = "parameters" + star + suffix + ".txt"

    tscale = 1
    
    output_path = spectra_path +"/" + star + suffix
    parameters_txt = open(param_file_name, 'w')
    parameters_txt.write(
        "binary=0\n"
        + "phaseSelFn=cosine\n"
        + "timeSelFn=none\n"
        + "bufferSize=10000\n"
        + "dims=1\n"
        + "numProcs=1\n"
        + "regions=0-0\n"
        + "numVars=1\n"
        + "varIndices=0\n"
        + "minPeriod=2\n"
        + "maxPeriod=" + str(round(duration * tscale / 1.5)) + "\n"
        + "tScale=" + str(tscale) + "\n"
        + "bootstrapSize=" + str(num_bootstrap) + "\n"
        + "varScales=1\n"
        + "filePath=" + input_path + "\n"
        + "outputFilePath=" + output_path +"\n")
    parameters_txt.close()
    return (param_file_name, output_path)

#def identical_line(freq, prev_freq):
#    if abs(prev_freq - freq) / abs(prev_freq) < 0.01:
#        return True
#    else:
#        return False

#def find_minimum(freqs, disps, prev_freq):
#    minima_indices = argrelextrema(disps, np.less)
#    minima_disps = disps[minima_indices]
#    minima_freqs = freqs[minima_indices]
#    min_disp_ind = np.argmin(minima_disps)
#    if identical_line(minima_freqs[min_disp_ind], prev_freq):
#        new_disps = np.delete(minima_disps, min_disp_ind)
#        min_disp_ind2 = np.argmin(new_disps)
#        print min_disp_ind, min_disp_ind2
#        if min_disp_ind2 >= min_disp_ind:
#            min_disp_ind2 += 1
#        min_disp_ind = min_disp_ind2
#    return (minima_freqs[min_disp_ind], minima_disps[min_disp_ind])

def run_external(param_file_name, output_path, bootstrap = False):
    print "Calculating D2 extrenally for " + param_file_name
    p = subprocess.Popen("./D2 " + param_file_name, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retval = p.wait()
    if bootstrap:
        suffix = "_bootstrap.csv"
        cols = (0, 2, 4)
    else:
        suffix = "_min.csv"
        cols = (0, 1)
    res = np.loadtxt(output_path + suffix, usecols=cols)
    return res

def run_d2(star, path, p_values=np.array([0.01, 0.05]), extrenal=False):
    file = star + ".dat"
    data = np.loadtxt(path + "/" + file)

    if extrenal:
        (param_file_name, output_path) = configure_d2(star, path)
        res = run_external(param_file_name, output_path)
    else:
        res = calc_d2(data, freqs_to_calc)

    freqs = res[:,0]
    disps = res[:,1]
    min_disp_ind = np.argmin(disps[1:-1])
    min_disp = disps[min_disp_ind]

    #best_freq, min_disp = find_minimum(freqs, disps, prev_freq);

    num_indep = 1
    if num_resamples == 0:
        if num_indep == 0:
            num_indep = len(freqs)
        z0s = -np.log(np.ones_like(len(p_values))-np.power(np.ones_like(len(p_values))-p_values, 1.0/num_indep))
    else:
        nh_data_file = "temp/" + star + "_nh.dat"
        if extrenal:
            (param_file_name_nh, output_path_nh) = configure_d2(star, path, '_nh', 0, nh_data_file)

        nh_disps = list()
        seasons = get_seasons(data, 1.0, True)
        for nh_ind in np.arange(0, num_resamples):
            if nh_ind % 10 == 0:
                print "Calculating null spectrum: " + str(nh_ind)
            resampled_data = list()
            for s in resample_seasons(seasons):
                for d in s:
                    resampled_data.append(d)
                    
            resampled_data = np.asarray(resampled_data)

            if extrenal:
                np.savetxt(nh_data_file, resampled_data, fmt='%f')
                res_nh = run_external(param_file_name_nh, output_path_nh)
            else:
                res_nh = calc_d2(resampled_data, freqs_to_calc)
                #np.savetxt(nh_data_file, resampled_data, fmt='%f')
                #(param_file_name_nh, output_path_nh) = configure_d2(star, path, '_nh', 0, nh_data_file)
                #res_nh1 = run_external(param_file_name_nh, output_path_nh)
                #print (res_nh[-500:-200,1] - res_nh1[-500:-200,1]) / res_nh[-500:-200,1]
            freqs_nh = res_nh[:,0]
            disps_nh = res_nh[:,1]

            ## Take the maximum peak
            #assert(num_indep == 1) # otherwise something is wrong
            #min_disp_ind = np.argmin(disps_nh[1:-1])
            #nh_disps.append(disps_nh[min_disp_ind + 1])
            ## Take a peak at the position of found minimum
            assert(num_indep == 1) # otherwise something is wrong
            disp_ind = random.randrange(np.shape(disps_nh)[0] - 1)
            nh_disps.append(disps_nh[min_disp_ind + 1])
            ## Little hack, just taking random spectral lines instead of independent
            #indices = np.random.choice(np.shape(disps_nh)[0] - 1, num_indep, replace=True, p=None)
            #nh_disps.extend(disps_nh[indices])
        nh_disps = np.sort(nh_disps)
        #z0_index = int(p_value * num_resamples * num_indep)
        #assert(z0_index >= 1)
        #z0 = nh_disps[z0_index]
        
        z0_indexes = p_values * num_resamples * num_indep
        assert(np.all(z0_indexes >= 1))
        z0s = nh_disps[(z0_indexes).astype(int) - 1]
        #print min_disp_ind
        
        result = list()

        if extrenal:
            (param_file_name_bs, output_path_bs) = configure_d2(star, path, '_bs', num_bootstrap)
            res_bs = run_external(param_file_name_bs, output_path_bs, True)
        else:
            res_bs = list()
            for bs_ind in np.arange(0, num_bootstrap):
                if bs_ind % 10 == 0:
                    print "Calculating bootstrap spectrum: " + str(bs_ind)
                bs_data = data[np.random.choice(np.shape(data)[0], np.shape(data)[0], replace=True, p=None),:]
                for freq, disp in calc_d2(bs_data, freqs_to_calc):
                    res_bs.append([bs_ind, freq, disp])
            res_bs = np.asarray(res_bs)
                
        bs_inds = res_bs[:,0]
        freqs_bs = res_bs[:,1]
        disps_bs = res_bs[:,2]
        ###############################################################
        for z0, p_value in zip(z0s, p_values):
            #plt.plot(freqs, disps, 'b-', freqs, np.zeros(len(freqs)) + z0, 'k--')
            #plt.savefig(star + "_spec" + str(p_value) + ".png")
            #plt.close()
            print min_disp, z0
            if min_disp < z0:
                best_freq = freqs[min_disp_ind + 1]
                print best_freq
                
                min_freqs_bs = list()
                for bs_ind in np.arange(0, num_bootstrap):
                    inds = np.where(bs_inds == bs_ind)
                    min_disp_ind_bs = np.argmin(disps_bs[inds])
                    best_freq_bs = freqs_bs[inds][min_disp_ind_bs]
                    if abs(best_freq_bs - best_freq)/best_freq < 0.25: #To exclude mismatched minima
                        min_freqs_bs.append(best_freq_bs)

                (skewKurt, normality) = stats.normaltest(min_freqs_bs) # normality is p-value
                plt.hist(min_freqs_bs, num_bootstrap/10)
                plt.savefig(spectra_path + "/"+ star + "_" + str(p_value) + "_" + str(best_freq) + '.png')
                plt.close()
                ###############################################
                result.append((freqs, disps, p_value, z0, best_freq, min_disp, np.std(min_freqs_bs), normality))
            else:
                result.append((freqs, disps, p_value, z0, 0, 0, 0, 0))
        return result

if star_name != None:
    f1=open(spectra_path + '/results_' + star_name + '.txt', 'w')
else:
    f1=open(spectra_path + '/results.txt', 'w')
    

for root, dirs, files in os.walk("detrended"):
    for file in files:
        if file[-4:] == ".dat":
            star = file[:-4]
            if star_name != None and star != star_name:
                continue
            print star
            data = np.loadtxt("detrended/"+star+".dat", usecols=(0,1))
            t = data[:,0]
            y = data[:,1]
            time_range = max(t) - min(t)

            all_res = list()

            res = run_d2(star, "detrended", p_values = p_values)
            all_res.extend(res)

            fig = plt.gcf()
            fig.set_size_inches(18, 6)
            #plt.plot(freqs, np.ones(len(freqs)) - disps, 'b-', freqs1, np.ones(len(freqs1)) - disps1, 'r-', freqs, np.ones(len(freqs)) - z0, 'k--')

            best_freqs = list()
            min_disps = list()

            for (freqs, disps, p_value, z0, best_freq, min_disp, std, normality) in res:
                y1 = y
                if best_freq > 0:
                    for i in np.arange(0, max_iters):
                        y_fit = LombScargle(t, y1).model(t, best_freq)
                        
                        y1 -= y_fit

                        residue = np.column_stack((t, y1))
                        np.savetxt("d2_input/" + star + "_" + str(i) + ".dat", residue, fmt='%f')
                        prev_res = (freqs, disps, p_value, z0, best_freq, min_disp, std, normality)

                        res = run_d2(star + "_" + str(i), "d2_input", p_values = np.array([p_value]))
                        if len(res) == 1:
                            freqs, disps, p_value, z0, best_freq, min_disp, min_freqs_std, normality = res[0]
                            
                            print p_value, z0, best_freq, min_disp, min_freqs_std, normality
                            if best_freq > 0:
                                all_res.extend(res)
                            else:
                                break
                        else:
                            break
            
            best_freq_p_values = list()
            best_freqs = list()
            min_disps = list()
            best_freq_stds = list()
            normalities = list()
            for _, _, p_value, _, best_freq, min_disp, best_freq_std, normality in all_res:
                if best_freq > 0:
                    best_freq_p_values.append(p_value)
                    best_freqs.append(best_freq)
                    min_disps.append(min_disp)
                    best_freq_stds.append(best_freq_std)
                    normalities.append(normality)
            #if len(all_res) > 0:
            #    _, _, _, _, best_freqs, min_disps, _, _ = zip(*all_res)
            #else:
            #    _, _, _, _, best_freq, min_disp, _, _ = all_res
            #    best_freqs = np.array([best_freq])
            #    min_disps = np.array([min_disp])
            #print all_res
            #inds = np.where(best_freqs > 0)
            #best_freqs = best_freqs[inds]
            #min_disps = min_disps[inds]
            if (len(best_freqs) > 0):
                plt.stem(best_freqs, np.ones(len(min_disps)) - min_disps)
                
            # filter out longer periods than 2/3 of the data_span
            filtered_freq_ids = np.where(best_freqs*time_range >= 1.5)
            best_freqs = best_freqs[filtered_freq_ids]
            best_freq_p_values = best_freq_p_values[filtered_freq_ids]
            best_freq_stds = best_freq_stds[filtered_freq_ids]
            normalities = normalities[filtered_freq_ids]
                
            f1.write(star +  " " + (' '.join(['%s %s %s %s' % (1/f, 1/(f - std) - 1/(f + std), p_value, n) for p_value, f, std, n in zip(best_freq_p_values, best_freqs, best_freq_stds, normalities)])) + "\n")

            (freqs, disps, p_value, z0, best_freq, min_disp, std, normality) = all_res[0]
            plt.plot(freqs, np.ones(len(freqs)) - disps, 'b-', freqs, np.ones(len(freqs)) - z0, 'k--')

            plt.xlabel("Frequency")
            plt.ylabel("Dispersion")
            plt.savefig(spectra_path + "/" + star + ".png")
            plt.close()
            f1.flush()
