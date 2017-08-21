# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:35:12 2016

@author: nigul
"""

import numpy as np
from scipy import optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy.linalg as la
import sys
from filelock import FileLock
import mw_utils

import os
import os.path

epsilon = 0.0001

def get_noise_cov(t, y):
    seasons = mw_utils.get_seasons(zip(t, y), 1.0, True)
    cov = np.identity(len(t))
    i = 0
    for season in seasons:
        var = np.var(season[:,1])
        season_len = np.shape(season)[0]
        for j in np.arange(i, i + season_len):
            cov[j, j] = var
        i += season_len
    return cov

class GPR():

    def __init__(self, name, cov_types, t, y, params, learning_rates, inertia, params_to_learn, stationary_noise = True):
        self.name = name
        self.cov_types = cov_types        
        self.t = t
        self.y = y
        self.params = np.array(params)
        self.learning_rates = np.array(learning_rates)
        self.inertia = np.array(inertia)
        self.momenta = np.zeros(len(params))
        self.params_to_learn = params_to_learn
        self.n = len(t)
        self.stationary_noise = stationary_noise
        
        if not self.stationary_noise:
            self.noise_cov = get_noise_cov(self.t, self.y)

    def calc_cov(self, params, t1, t2, data_or_test):
        K = np.zeros((len(t1), len(t2)))
        param_index = 0
        for cov_type in self.cov_types:
            if cov_type == "square_exponential":
                inv_len_scale2 = params[param_index]
                sigma_f = params[param_index+1]
                for i in np.arange(0, len(t1)):
                    for j in np.arange(0, len(t2)):
                        K[i, j] += sigma_f * np.exp(-0.5 * inv_len_scale2 * (t1[i] - t2[j])**2)
                param_index += 2
            elif cov_type == "periodic":
                amplitude = params[param_index]
                harmonicity = params[param_index+1]
                freq = params[param_index+2]
                for i in np.arange(0, len(t1)):
                    for j in np.arange(0, len(t2)):
                        tau = t1[i] - t2[j]
                        K[i, j] += amplitude * np.exp(-2.0*np.sin(np.pi*freq*tau)**2/harmonicity)
                param_index += 3
            elif cov_type == "noise":
                if data_or_test:
                    assert(len(t1) == len(t2))
                    if self.stationary_noise:
                        sigma_n = params[param_index]
                        K += np.identity(len(t1))*sigma_n
                    else:
                        K += self.noise_cov
                param_index += 1
            else:
                raise Exception("Unsupported covariance function type")
        return K

    def calc_cov_deriv(self, params):
        derivs = []
        param_index = 0
        for cov_type in self.cov_types:
            if cov_type == "square_exponential":
                inv_len_scale2 = params[param_index]
                sigma_f = params[param_index+1]
                K_l = np.zeros((self.n, self.n))
                K_sf = np.zeros((self.n, self.n))
                for i in np.arange(0, self.n):
                    for j in np.arange(i, self.n):
                        tau2 = (self.t[i] - self.t[j])**2
                        if self.params_to_learn[param_index]:
                            K_l[i, j] = -0.5 * sigma_f * tau2 * np.exp(-0.5 * inv_len_scale2 * tau2)
                            K_l[j, i] = K_l[i, j]
                        if self.params_to_learn[param_index + 1]:
                            K_sf[i, j] = np.exp(-0.5 * inv_len_scale2 * tau2)
                            K_sf[j, i] = K_sf[i, j]
                derivs.append(K_l)
                derivs.append(K_sf)
                param_index += 2
            elif cov_type == "periodic":
                amplitude = params[param_index]
                harmonicity = params[param_index+1]
                freq = params[param_index+2]
                K_amp = np.zeros((self.n, self.n))
                K_har = np.zeros((self.n, self.n))
                K_f = np.zeros((self.n, self.n))
                for i in np.arange(0, self.n):
                    for j in np.arange(i, self.n):
                        tau = self.t[i] - self.t[j]
                        sin2 = np.sin(np.pi*freq*tau)**2
                        if self.params_to_learn[param_index]:
                            K_amp[i, j] = np.exp(-2.0*sin2/harmonicity)
                            K_amp[j, i] = K_amp[i, j]
                        if self.params_to_learn[param_index + 1]:
                            K_har[i, j] = 2.0 * amplitude / (harmonicity**2) * sin2 * np.exp(-2.0*sin2/harmonicity)
                            K_har[j, i] = K_har[i, j]
                derivs.append(K_amp)
                derivs.append(K_har)
                derivs.append(K_f)
                param_index += 3
            elif cov_type == "noise":
                K_sn = np.zeros((self.n, self.n))
                if self.params_to_learn[param_index]:
                    K_sn = np.identity(self.n)
                derivs.append(K_sn)
                param_index += 1
            else:
                raise Exception("Unsupported type")
        return np.asarray(derivs)

    def obj_func(self, params, *args):
        print params
        K = self.calc_cov(params, self.t, self.t, True)
        L = la.cholesky(K)
        alpha = la.solve(L.T, la.solve(L, self.y))
        loglik = 0.5 * np.dot(self.y.T, alpha) + sum(np.log(np.diag(L))) + 0.5 * self.n * np.log(2.0 * np.pi)
        #sign, slogdet = la.slogdet(K)
        #loglik_true = 0.5 * np.dot(self.y.T, alpha) + 0.5*(slogdet) + 0.5 * self.n * np.log(2.0 * np.pi)
        #print  loglik_true, loglik, sign
        self.loglik = loglik
        print loglik
        return loglik

    def obj_func_grad(self, params, *args):
        K = self.calc_cov(params, self.t, self.t, True)
        K_inv = la.inv(K)
        alpha = np.dot(K_inv, self.y)
        cov_deriv = self.calc_cov_deriv(params)
        loglik_deriv = -0.5 * (np.tensordot(np.dot(cov_deriv, alpha), alpha, (1,0)))
        loglik_deriv += 0.5 * np.sum(K_inv * np.transpose(cov_deriv, (0, 2, 1)), axis=(1,2))
        #for i in np.arange(len(params)):
        #    loglik_deriv_true = -0.5 * np.dot(alpha, np.dot(cov_deriv[i], alpha)) + 0.5 * np.trace(np.dot(K_inv, cov_deriv[i]))
        #    if abs(loglik_deriv[i] - loglik_deriv_true) > 1e-10:
        #        print i, loglik_deriv_true, loglik_deriv[i]
        #        #assert(False)
        return loglik_deriv

    def optimize(self):
        #self.params = optimize.fmin_ncg(self.obj_func, self.params, fprime=self.obj_func_grad, args=())
        #params, _, _, _, _, _ , _, _ = optimize.fmin_bfgs(self.obj_func, self.params, fprime=None, args=())#self.obj_func_grad, args=())
        #params, _, _, _, _, _ , _, _ = optimize.fmin_bfgs(self.obj_func, self.params, fprime=self.obj_func_grad, args=())
        #(params, nfeval, rc) = optimize.fmin_tnc(self.obj_func, self.params, fprime=self.obj_func_grad, args=(), bounds=len(self.params)*[(1e-100, None)])
        #print self.name, "num iterations", nfeval, "return code: ", rc 
        (params, loglik, d) = optimize.fmin_l_bfgs_b(self.obj_func, self.params, fprime=self.obj_func_grad, args=(), bounds=len(self.params)*[(1e-10, None)])
        print self.name, "return code: ", d
       
        self.params = params

    def calc_log_lik_deriv(self, K_inv, alpha):
        cov_deriv = self.calc_cov_deriv(self.params)
        # loglik_deriv = 0.5(y.T K_inv dK/dtheta K_inv y - tr(K_inv dK/dtheta))
        loglik_deriv = 0.5 * (np.tensordot(np.dot(cov_deriv, alpha), alpha, (1,0)))
        loglik_deriv -= 0.5 * np.sum(K_inv * np.transpose(cov_deriv, (0, 2, 1)), axis=(1,2))
        return loglik_deriv

    def step(self):
        K = self.calc_cov(self.params, self.t, self.t, True)
        K_inv = la.inv(K)
        L = la.cholesky(K)
        alpha = np.dot(K_inv, self.y)
        loglik = -0.5 * np.dot(self.y.T, alpha) - sum(np.log(np.diag(L))) - 0.5 * self.n * np.log(2.0 * np.pi)
        new_momenta = self.inertia * self.momenta + self.learning_rates * self.calc_log_lik_deriv(K_inv, alpha)
        new_params = self.params + new_momenta
        return new_params, new_momenta, loglik

    def optimize2(self):
        self.loglik = None
        params_list = []
        momenta_list = []
        while True:
            old_loglik = self.loglik
            #learning_rates = learning_rates * 0.999
            params_list.append(self.params)
            momenta_list.append(self.momenta)
            params, momenta, self.loglik = self.step()
            self.params = params
            self.momenta = momenta
            print star, params_list[-1], self.loglik
            indices = np.where(params < 0)[0]
            if old_loglik != None and self.loglik < old_loglik - epsilon:
                print "New loglik smaller than previous -> decreasing learning rates"
                self.loglik = None
                self.learning_rates = self.learning_rates * 0.1
                self.inertia = self.inertia * 0.1
                params_list.pop()
                self.params = params_list.pop()
                momenta_list.pop()
                self.momenta = momenta_list.pop()
            elif len(indices) > 0:
                print "Invalid parameter(s)", indices, "-> decreasing learning rate(s)"
                self.loglik = None
                self.learning_rates[indices] = self.learning_rates[indices] * 0.1
                self.inertia[indices] = self.inertia[indices] * 0.1
                self.params = params_list.pop()
                self.momenta = momenta_list.pop()
            elif old_loglik != None and self.loglik >= old_loglik + epsilon:# and loglik < old_loglik + 1:
                #print "Loglik increasing too slow -> increasing learning rates"
                self.learning_rates = self.learning_rates * 2.0
                self.inertia = self.inertia * 2.0
            elif old_loglik != None and abs(self.loglik - old_loglik) < epsilon:
                break
        
        self.params = params_list[-1]
        


    def fit(self):
        K = self.calc_cov(self.params, self.t, self.t, True)
        L = la.cholesky(K)
        alpha = la.solve(L.T, la.solve(L, self.y))
        t_test = np.linspace(min(self.t), max(self.t), 500)
        K_test = self.calc_cov(self.params, t_test, self.t, False)
        f_mean = np.dot(K_test, alpha)
        v = la.solve(L, K_test.T)
        covar = self.calc_cov(self.params, t_test, t_test, False) - np.dot(v.T, v)
        var = np.diag(covar)
        loglik = -0.5 * np.dot(self.y.T, alpha) - sum(np.log(np.diag(L))) - 0.5 * self.n * np.log(2.0 * np.pi)
        
        fig = plt.gcf()
        fig.set_size_inches(18, 6)
        plt.plot(self.t, self.y, 'b+')
        plt.plot(t_test, f_mean, 'k-')
        plt.fill_between(t_test, f_mean + 2.0 * np.sqrt(var), f_mean - 2.0 * np.sqrt(var), alpha=0.1, facecolor='lightgray', interpolate=True)
    
        plt.savefig("GPR/"+star + '.png')
        plt.close()
    
        with FileLock("GPRLock"):
            with open("GPR/results.txt", "a") as output:
                output.write(star + " " + str(loglik) + ' ' + (' '.join(['%s' % (param) for param in self.params])) + "\n")
        
        

num_groups = 1
group_no = 0
if len(sys.argv) > 1:
    num_groups = int(sys.argv[1])
if len(sys.argv) > 2:
    group_no = int(sys.argv[2])

files = []

for root, dirs, dir_files in os.walk("cleaned"):
    for file in dir_files:
        if file[-4:] == ".dat":
            files.append(file)

modulo = len(files) % num_groups
group_size = len(files) / num_groups
if modulo > 0:
    group_size +=1

output = open("GPR/results.txt", 'w')
output.close()
output = open("GPR/all_results.txt", 'w')
output.close()

offset = 1979.3452

rot_periods = mw_utils.load_rot_periods()

for i in np.arange(0, len(files)):
    if i < group_no * group_size or i >= (group_no + 1) * group_size:
        continue
    file = files[i]
    star = file[:-4]
    star = star.upper()
    if (star[-3:] == '.CL'):
        star = star[0:-3]
    if (star[0:2] == 'HD'):
        star = star[2:]
    #if star != "218658":
    #    continue
    print star
    dat = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)
    t = dat[:,0]
    t /= 365.25
    t += offset
    duration = max(t) - min(t)
    
    y = dat[:,1]
    y -= np.mean(y)
    
    indices1 = np.random.choice(len(t), len(t)/2, replace=False, p=None)
    indices1 = np.sort(indices1)

    indices2 = np.random.choice(len(indices1), len(indices1)/2, replace=False, p=None)
    indices2 = np.sort(indices2)
    
    best_loglik = None

    #Test
    #params = optimize.fmin_bfgs(lambda params, *args: np.sin(params[0])*np.cos(params[1]), [1, 1], fprime=None, args=())
    #print "Optimum: ", params

    var = np.var(y)
    sigma_f = var/4#np.var(y) / 2
    #sigma_n = np.max(get_noise_var(t, y))
    sigma_n = np.max(mw_utils.get_seasonal_noise_var(t, y))
    #sigma_n = var/4#sigma_f
    if rot_periods.has_key(star):
        cov_types = ["square_exponential", "periodic", "noise"]
        amplitude = var/10
        harmonicity = duration
        rot_freq = 365.25/rot_periods[star]
        params = np.asarray([None, sigma_f, amplitude, harmonicity, rot_freq, sigma_n])
        learning_rates = np.asarray([duration*1e-10, var*1e-7, var*0.000001, 0.0, 0.0, 0.0])#var*1.0e-7])
        inertia = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        params_to_learn = [True, True, True, False, False, False]
    else:
        cov_types = ["square_exponential", "noise"]
        params = np.asarray([None, sigma_f, sigma_n])
        learning_rates = np.asarray([duration*0.0001, var*0.00001, 0.0])#var*1.0e-7])
        inertia = np.asarray([0.0, 0.0, 0.0])
        params_to_learn = [True, True, False]

    for len_scale in [duration/2]: # Try different initial conditions
        params[0] = 1.0/len_scale/len_scale
        
        gpr=GPR(star, cov_types, t[indices1][indices2], y[indices1][indices2], params, learning_rates, inertia, params_to_learn)
        gpr.optimize()

        gpr=GPR(star, cov_types, t[indices1], y[indices1], gpr.params, gpr.learning_rates, gpr.inertia, params_to_learn)
        gpr.optimize()

        gpr=GPR(star, cov_types, t, y, gpr.params, gpr.learning_rates, gpr.inertia, params_to_learn)
        #gpr=GPR(star, cov_types, t, y, params, learning_rates, inertia, params_to_learn)
        gpr.optimize()
    
        with FileLock("GPRLock"):
            with open("GPR/all_results.txt", "a") as output:
                output.write(star + " " + str(gpr.loglik) + ' ' + (' '.join(['%s' % (param) for param in gpr.params])) + "\n")    

        if best_loglik == None or gpr.loglik > best_loglik:
            best_params = gpr.params
            best_loglik = gpr.loglik
    
    
    gpr=GPR(star, cov_types, t, y, best_params, [], [], [])
    gpr.fit()
