import numpy as np
import matplotlib.pyplot as plt

def calc_cov_qp(t, f, length_scale, sig_var):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 0.5/length_scale/length_scale
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*np.exp(-inv_l2*(t[i]-t[j])**2)*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k

def calc_cov_p(t, f, sig_var):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k


def calc_sel_fn_qp(t, f, length_scale, sig_var):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 0.5/length_scale/length_scale
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = np.exp(-inv_l2*(t[i]-t[j])**2)*(1.0 + 2.0*(np.cos(2 * np.pi*f*(t[i] - t[j]))))
            k[j, i] = k[i, j]
    return k

def calc_sel_fn_p(t, f, sig_var):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = 1.0 + 2.0*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k
    
def calc_g(t, sig_var, noise_var, k, a=1.0):
    sigma2 = sig_var + noise_var
    g_1 = np.zeros((len(t), len(t)))
    g_2 = np.zeros((len(t), len(t)))
    g_log = 0
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            if i == j:
                g_1[i, j] = a
                g_2[i, j] = a
            else:
                det = sigma2**2-k[i, j]**2
                g_1[i, j] = sigma2/det
                g_1[j, i] = g_1[i, j]
                g_2[i, j] = k[i, j]/det
                g_2[j, i] = g_2[i, j]
                g_log += np.log(det)
    return g_1, g_2, g_log
    
def calc_d2(t, y, sel_fn, normalize):
    d2 = 0.0
    norm = 0.0
    i = 0
    for ti in t:
        j = 0
        for tj in t:
            d2 += sel_fn[i, j] * (y[i] - y[j]) ** 2
            norm += sel_fn[i, j]
            j += 1
        i += 1
    if normalize:
        return d2 / norm / 2.0
    else:
        return d2

def calc_kalman(kalman_utils, t, y, sig_var, noise_var, t_coh, f, plot, coh_ind, f_ind):

    y_means, loglik = kalman_utils.do_filter([sig_var, 2.0*np.pi*f, 1.0, t_coh, noise_var])
    #y_means, loglik = kalman_utils.do_filter([sig_var, 2.0*np.pi*f, 500, noise_var])
    if plot:
        fig, (ax1) = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(6, 3)
        ax1.plot(t, y, 'b+')
        ax1.plot(t[1:], y_means, 'r--')
        fig.savefig('kalman_' + str(coh_ind) + '_' + str(f_ind) + '.png')
        plt.close(fig)
    return loglik

