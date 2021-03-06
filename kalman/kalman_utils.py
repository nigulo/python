
import numpy as np
import utils.plot as plot
import scipy.special as special
import numpy.linalg as la
from scipy.special import gamma
from scipy.special import kv
from scipy.linalg import expm

from component_exp_quad import component_exp_quad
from component_matern import component_matern
from component_periodic import component_periodic
from component_quasiperiodic import component_quasiperiodic
from component_linear_trend import component_linear_trend
from sampler import sampler

try:
    from scipy.linalg import solve_lyapunov as solve_continuous_lyapunov
except ImportError:  # pragma: no cover; github.com/scipy/scipy/pull/8082
    from scipy.linalg import solve_continuous_lyapunov

import kalman
from cov_exp_quad import cov_exp_quad
from cov_matern import cov_matern


def calc_cov_linear_trend(t, trend_var, c = 0.0):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = trend_var * (t[i] - c)*(t[j] - c)
            k[j, i] = k[i, j]
    return k

def calc_cov_matern(t, length_scale, sig_var, nu):
    k = np.zeros((len(t), len(t)))
    sqrt_two_nu_inv_l = np.sqrt(2.0*nu)/length_scale
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            if i == j:
                k[i, j] = sig_var
            else:
                tau = np.abs(t[i]-t[j])
                k[i, j] = sig_var*2.0**(1.0-nu)/gamma(nu)*(sqrt_two_nu_inv_l*tau)**nu*kv(nu, sqrt_two_nu_inv_l*tau)
            #test = sig_var*np.exp(-np.abs(t[i]-t[j])/length_scale)
            #assert(k[i, j] - test < 1e-15)
            k[j, i] = k[i, j]
    return k

def calc_cov_exp_quad(t, length_scale, sig_var):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 0.5/(length_scale*length_scale)
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*np.exp(-inv_l2*(t[i]-t[j])**2)
            k[j, i] = k[i, j]
    return k

def calc_cov_qp(t, f, length_scale, sig_var):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 0.5/(length_scale*length_scale)
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*np.exp(-inv_l2*(t[i]-t[j])**2)*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            #k[i, j] = sig_var*np.exp(-np.abs(t[i]-t[j])/length_scale)*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k

def calc_cov_p(t, f, sig_var):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k

class kalman_utils():
    
    def __init__(self, t, y, num_iterations = 100, condition_fn = None, initial_indices = None, max_iters_wo_progress = 10):
        self.t = t
        self.y = y
        self.cov_types = []
        self.cov_settings = dict()
        self.param_counts = dict()
        self.sampler = sampler(self.loglik_fn, condition_fn = condition_fn, initial_indices = initial_indices)
        self.num_iterations = num_iterations
        self.has_A = False
        self.delta_t = t[1:] - t[:-1]
        self.max_iters_wo_progress = max_iters_wo_progress

    def loglik_fn(self, param_values):
        A_shape = np.array([0, 0])
        F_shape = np.array([0, 0])
        L_shape = np.array([0, 0])
        H_shape = np.array([0])
        m_0_shape = np.array([0])
        P_0_shape = np.array([0, 0])
        Q_shape = np.array([0, 0])
    
        cov_data = []
        index = 0
        R = 0.0
        for cov_type in self.cov_types:
            F_is_A = False
            if cov_type == "white_noise":
                R = param_values[index]
            else:
                if cov_type == "linear_trend":
                    component = component_linear_trend(slope=param_values[index], intercept=param_values[index+1], t=self.t)
                    F_is_A = True
                elif cov_type == "periodic":
                    component = component_periodic(self.cov_settings[cov_type]["j_max"], sig_var=param_values[index], omega_0=param_values[index+1], ell=param_values[index+2])
                elif cov_type == "quasiperiodic":
                    component = component_quasiperiodic(self.cov_settings[cov_type]["j_max"], sig_var=param_values[index], omega_0=param_values[index+1], ellp=param_values[index+2], ellq=param_values[index+3])
                elif cov_type == "exp_quad":
                    component = component_exp_quad(sig_var=param_values[index], ell=param_values[index+1])
                elif cov_type == "matern":
                    component = component_matern(sig_var=param_values[index], ell=param_values[index+1], p=self.cov_settings[cov_type]["p"])
                else:           
                    assert(True==False)
                index += self.param_counts[cov_type]
                F, L, H, m_0, P_0, Q_c = component.get_params()
                if F_is_A:
                    A = F
                    A_shape += np.shape(A)[1:]
                else:
                    A = None
                    if self.has_A:
                        A_shape += np.shape(F)
                    else:
                        F_shape += np.shape(F)
                kf = kalman.kalman(t=self.t, y=self.y, F=F, L=L, H=H, R=0, m_0=m_0, P_0=P_0, Q_c=Q_c, noise_int_prec=100, F_is_A=self.has_A)
                kf.calc_Q()
                L_shape += np.shape(L)
                H_shape += np.shape(H)
                m_0_shape += np.shape(m_0)
                P_0_shape += np.shape(P_0)
                Q_shape += np.shape(kf.Q)[1:]
                cov_data.append((A, F, L, H, m_0, P_0, kf.Q))
    
        As = np.zeros((len(self.delta_t), A_shape[0], A_shape[1]))
        Fs = np.zeros(F_shape)
        Ls = np.zeros(L_shape)
        Hs = np.zeros(H_shape)
        m_0s = np.zeros(m_0_shape)
        P_0s = np.zeros(P_0_shape)
        Qs = np.zeros((len(self.delta_t), Q_shape[0], Q_shape[1]))
        
        A_index = np.array([0, 0])
        F_index = np.array([0, 0])
        L_index = np.array([0, 0])
        H_index = np.array([0])
        m_0_index = np.array([0])
        P_0_index = np.array([0, 0])
        Q_index = np.array([0, 0])
        
        for A, F, L, H, m_0, P_0, Q in cov_data:
            if self.has_A:
                if A is None:
                    A = np.zeros((len(self.delta_t), np.shape(F)[0], np.shape(F)[1]))
                    for i in np.arange(0, len(self.delta_t)):
                        A[i] = expm(F*self.delta_t[i])
                A_size = np.shape(A)[1:]
                for i in np.arange(0, np.shape(As)[0]):
                    #print np.shape(A[i])
                    #print np.shape(As[i])
                    As[i, A_index[0]:A_index[0]+A_size[0], A_index[1]:A_index[1]+A_size[1]] = A[i]
                A_index += A_size
                #print As[0]
            else:
                F_size = np.shape(F)
                Fs[F_index[0]:F_index[0]+F_size[0], F_index[1]:F_index[1]+F_size[1]] = F
                F_index += F_size
            L_size = np.shape(L)
            Ls[L_index[0]:L_index[0]+L_size[0], L_index[1]:L_index[1]+L_size[1]] = L
            L_index += L_size
            
            m_0_size = np.shape(m_0)
            m_0s[m_0_index[0]:m_0_index[0]+m_0_size[0]] = m_0
            m_0_index += m_0_size
    
            P_0_size = np.shape(P_0)
            P_0s[P_0_index[0]:P_0_index[0]+P_0_size[0], P_0_index[1]:P_0_index[1]+P_0_size[1]] = P_0
            P_0_index += P_0_size
    
            H_size = np.shape(H)
            Hs[H_index[0]:H_index[0]+H_size[0]] = H
            H_index += H_size
    
            Q_size = np.shape(Q)[1:]
            for i in np.arange(0, np.shape(Qs)[0]):
                Qs[i, Q_index[0]:Q_index[0]+Q_size[0], Q_index[1]:Q_index[1]+Q_size[1]] = Q[i]
            Q_index += Q_size
    
        if self.has_A:
            kf = kalman.kalman(t=self.t, y=self.y, F=As, L=Ls, H=Hs, R=R, m_0=m_0s, P_0=P_0s, Q_c=None, noise_int_prec=100, F_is_A=self.has_A, Q=Qs)
        else:
            kf = kalman.kalman(t=self.t, y=self.y, F=Fs, L=Ls, H=Hs, R=R, m_0=m_0s, P_0=P_0s, Q_c=None, noise_int_prec=100, F_is_A=self.has_A, Q=Qs)
        y_means, loglik = kf.filter()
        y_means_smooth = kf.smooth()
        return y_means_smooth, loglik

    def add_component(self, cov_type, param_values, settings = dict(), param_funcs = None):
        if cov_type == "linear_trend":
            assert(len(param_values) == 2)
            self.has_A = True
        elif cov_type == "periodic":
            assert(len(param_values) == 3)
        elif cov_type == "quasiperiodic":
            assert(len(param_values) == 4)
        elif cov_type == "exp_quad":
            assert(len(param_values) == 2)
        elif cov_type == "matern":
            assert(len(param_values) == 2)
        elif cov_type == "white_noise":
            assert(len(param_values) == 1)
        else:           
            raise Exception("Invalid parameter values")
        self.cov_types.append(cov_type)
        self.param_counts[cov_type] = len(param_values)
        self.sampler.add_parameter_values(param_values, param_funcs)
        self.cov_settings[cov_type] = settings

    '''
    Do the inference of optimal parameter values using the greedy search
    '''        
    def do_inference(self):
        self.sampler.init()
        self.results = []
        last_iteration = -1
        last_loglik = -float("inf")
        num_iters_wo_progress = 0
        while self.sampler.get_iteration() < self.num_iterations and num_iters_wo_progress < self.max_iters_wo_progress:
            if self.sampler.get_iteration() != last_iteration:
                print("Iteration", self.sampler.get_iteration())
                last_iteration = self.sampler.get_iteration()
                if last_iteration > 0:
                    param_modes, param_means, param_sigmas, y_means, loglik = self.sampler.get_results()
                    print(param_modes)
                    if loglik == last_loglik:
                        num_iters_wo_progress += 1
                    else:
                        last_loglik = loglik
                        num_iters_wo_progress = 0
            params_sample, loglik = self.sampler.sample()
            if loglik is not None:
                self.results.append([params_sample, loglik])
            #print "Sample", params_sample, loglik
        return self.sampler.get_results()

    '''
    Do filtering for given parameter values
    '''        
    def do_filter(self, params):
        y_means, loglik = self.loglik_fn(params)
        return y_means, loglik


    def plot(self, file_name_prefix=""):
        plots = []
        num_params = len(self.results[0][0])
        for i in np.arange(num_params):
            plots.append(plot.plot())
        for result in self.results:
            params = result[0]
            loglik = result[1]
            for i in np.arange(num_params):
                plots[i].plot(params[i], loglik, params="k+")
                
        for i in np.arange(num_params):
            plots[i].save(file_name_prefix + "param" + str(i) + ".png")
            plots[i].close()
        
