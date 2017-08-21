import numpy as np
import matplotlib.pyplot as plt

def bayes_lin_reg(t, y, w):
    W = sum(w)
    wt_arr = w * t
    tau = sum(wt_arr) / W
    wy_arr = w * y

    yt = sum(wy_arr * (t - tau))
    Y = sum(wy_arr)
    tt = sum(wt_arr * (t - tau))

    sigma_alpha = 1.0/tt
    mu_alpha = yt * sigma_alpha

    sigma_beta = 1.0 / W
    mu_beta = Y * sigma_beta - mu_alpha * tau

    norm_term = sum(np.log(np.sqrt(w)) - np.log(np.sqrt(2.0*np.pi)))

    y_model = t * mu_alpha + mu_beta
    loglik = norm_term - 0.5 * sum(w * (y - y_model)**2)

    return ((mu_alpha, mu_beta), (sigma_alpha, sigma_beta), y_model, loglik)


def _test():
    time_range = 100.0
    n = 100
    t = np.random.uniform(0.0, time_range, n)
    #t = np.random.randint(time_range, size=n)+np.random.rand(n)
    t = np.sort(t)
    sigma = np.random.normal(0, 1, n)
    assert(np.all(sigma != 0))
    #y = np.ones(len(t))
    y = sigma + t/20
    #y = sigma
    w = np.ones(len(sigma))/sigma**2
    
    
    ((mu_alpha, mu_beta), (sigma_alpha, sigma_beta), y_model, loglik) = bayes_lin_reg(t, y, w)
    
    plt.scatter(t, y)
    t_model = np.linspace(min(t), max(t), 1000)
    y_model = t_model * mu_alpha + mu_beta
    plt.plot(t_model, y_model, 'k-')
    
    plt.show()
    
