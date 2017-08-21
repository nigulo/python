import numpy as np
import scipy
import scipy.misc

class BGLST2():

    def __init__(self, t, y, w):
        self.t = t
        self.y = y
        self.w = w
        self.wy_arr = w * y
        self.yy = sum(self.wy_arr * y)
        self.Y = sum(self.wy_arr)
        self.W = sum(w)
        self.two_pi_t = 2.0 * np.pi * t
        self.four_pi_t = 4.0 * np.pi * t
        self.norm_term = -sum(np.log(np.sqrt(w)*np.sqrt(2*np.pi)))

    def _linreg(self):
        W = sum(self.w)
        wt_arr = self.w * self.t
        tau = sum(wt_arr) / W
        wy_arr = self.w * self.y
    
        yt = sum(wy_arr * (self.t - tau))
        Y = sum(wy_arr)
        tt = sum(wt_arr * (self.t - tau))
    
        sigma_alpha = 1.0/tt
        mu_alpha = yt * sigma_alpha
    
        sigma_beta = 1.0 / W
        mu_beta = Y * sigma_beta - mu_alpha * tau
    
        y_model = self.t * mu_alpha + mu_beta
        loglik = self.norm_term - 0.5 * sum(self.w * (self.y - y_model)**2)
        return ((mu_alpha, mu_beta), (sigma_alpha, sigma_beta), y_model, loglik)

    def calc(self, freq):
        s_arr_1 = np.sin(self.two_pi_t * freq)
        c_arr_1 = np.cos(self.two_pi_t * freq)
        return self._calc(s_arr_1, c_arr_1)
        
    def _calc(self, s_arr_1, c_arr_1):
        s_2_arr = 2.0 * s_arr_1 * c_arr_1
        c_2_arr = c_arr_1 * c_arr_1 - s_arr_1 * s_arr_1
        tau = 0.5 * np.arctan(sum(self.w * s_2_arr)/sum(self.w * c_2_arr))
        cos_tau = np.cos(tau)
        sin_tau = np.sin(tau)
        s_arr = s_arr_1 * cos_tau - c_arr_1 * sin_tau
        c_arr = c_arr_1 * cos_tau + s_arr_1 * sin_tau
        #c_arr = np.cos(self.two_pi_t * freq - tau)
        #s_arr = np.sin(self.two_pi_t * freq - tau)
        wc_arr = self.w * c_arr
        ws_arr = self.w * s_arr
        c = sum(wc_arr)
        s = sum(ws_arr)
        cc = sum(wc_arr * c_arr)
        ss = sum(ws_arr * s_arr)
        ct = sum(wc_arr * self.t)
        st = sum(ws_arr * self.t)
        yc = sum(self.wy_arr * c_arr)
        ys = sum(self.wy_arr * s_arr)

        tau2 = ct / c
        
        wt_arr = self.w * (self.t - tau2)
        tt = sum(wt_arr * (self.t - tau2))
        T = sum(wt_arr)
        yt = sum(self.wy_arr * (self.t - tau2))
        ct = sum(wc_arr * (self.t - tau2))
        st = sum(ws_arr * (self.t - tau2))
        
        if ss == 0.0:
            assert(cc > 0)
            K = (ct**2/cc - tt)/2.0
            assert(K < 0)
            M = yt - yc * ct / cc
            N = ct * c / cc - T
            P = (c**2 / cc - self.W - N**2 / 2.0 / K)/2.0
            Q = -yc * c / cc + self.Y - M * N / 2.0 / K
            if P == 0.0:
                if Q != 0.0:
                    print "WARNING: Q=", Q
                assert(abs(Q) < 1e-8)
                log_prob = self.norm_term + np.log(2.0 * np.pi**2 / np.sqrt(-cc * K)) + (yc**2/2.0/cc - M**2/4.0/K - self.yy/2.0)
            else:
                assert(P < 0)
                log_prob = self.norm_term + np.log(2.0 * np.pi**2 / np.sqrt(cc * K * P)) + (yc**2/2.0/cc - M**2/4.0/K - Q**2/4.0/P - self.yy/2.0)
        elif cc == 0.0:
            assert(ss > 0)
            K = (st**2/ss - tt)/2.0
            assert(K < 0)
            M = yt - ys * st / ss
            N = st * s / ss - T
            P = (s**2 / ss - self.W - N**2 / 2.0 / K)/2.0
            Q = -ys * s / ss + self.Y - M * N / 2.0 / K
            if P == 0.0:
                if Q != 0.0:
                    print "WARNING: Q=", Q
                log_prob = self.norm_term + np.log(2.0 * np.pi**2 / np.sqrt(-ss * K)) + (ys**2/2.0/ss - M**2/4.0/K - self.yy/2.0)
            else:
                assert(P < 0)
                log_prob = self.norm_term + np.log(2.0 * np.pi**2 / np.sqrt(ss * K * P)) + (ys**2/2.0/ss - M**2/4.0/K - Q**2/4.0/P - self.yy/2.0)
        else:
            assert(cc > 0)
            assert(ss > 0)
            K = (ct**2/cc + st**2/ss - tt)/2.0
            assert(K < 0)
            M = yt - yc * ct / cc - ys * st / ss
            N = ct * c / cc + st * s / ss - T
            P = (c**2 / cc + s**2 / ss - self.W - N**2 / 2.0 / K)/2.0
            Q = -yc * c / cc - ys * s / ss + self.Y - M * N / 2.0 / K
            if P == 0.0:
                if Q != 0.0:
                    print "WARNING: Q=", Q
                log_prob = self.norm_term + np.log(2.0 * np.pi**2 / np.sqrt(-cc * ss * K)) + (yc**2/2.0/cc + ys**2/2.0/ss - M**2/4.0/K - self.yy/2.0)
            elif P > 0 and abs(P) < 1e-6:
                print "ERROR: P=", P
                if Q != 0.0:
                    print "WARNING: Q=", Q
                log_prob = self.norm_term + np.log(2.0 * np.pi**2 / np.sqrt(-cc * ss * K)) + (yc**2/2.0/cc + ys**2/2.0/ss - M**2/4.0/K - self.yy/2.0)
            else:
                if P > 0:
                    print "ERROR: P=", P
                assert(P < 0)
                log_prob = self.norm_term + np.log(2.0 * np.pi**2 / np.sqrt(cc * ss * K * P)) + (yc**2/2.0/cc + ys**2/2.0/ss - M**2/4.0/K - Q**2/4.0/P - self.yy/2.0)
        return (log_prob, (tau, tau2, ss, cc, ys, yc, st, ct, s, c, K, M, N, P, Q))

    def step(self, n):
        if n == 0 and all(self.sin_0 == 0):
            # zero frequency
            (_, _, _, loglik) = self._linreg()
            return loglik
        
        if n > 1:
            sin_n_delta = self.two_cos_delta * self.sin_n_1_delta - self.sin_n_2_delta
            cos_n_delta = self.two_cos_delta * self.cos_n_1_delta - self.cos_n_2_delta
        elif n == 1:
            sin_n_delta = np.sin(self.delta)
            cos_n_delta = np.cos(self.delta)
        else:# n == 0:
            sin_n_delta = np.zeros(len(self.t))
            cos_n_delta = np.ones(len(self.t))
        
        self.sin_n_2_delta = self.sin_n_1_delta
        self.cos_n_2_delta = self.cos_n_1_delta
        self.sin_n_1_delta = sin_n_delta
        self.cos_n_1_delta = cos_n_delta

        s_arr_1 = self.sin_0 * cos_n_delta + self.cos_0 * sin_n_delta
        c_arr_1 = self.cos_0 * cos_n_delta - self.sin_0 * sin_n_delta

        log_prob, _ =  self._calc(s_arr_1, c_arr_1)
        return log_prob


    def calc_all(self, freq_start, freq_end, count):
        self.freq_start = freq_start
        delta_freq = (freq_end - freq_start) / count
        self.delta = self.two_pi_t * delta_freq
        self.two_cos_delta = 2.0 * np.cos(self.delta)
        self.sin_n_1_delta = np.zeros(len(self.t))
        self.cos_n_1_delta = np.ones(len(self.t))
        
        self.sin_0 = np.sin(self.two_pi_t * freq_start)
        self.cos_0 = np.cos(self.two_pi_t * freq_start)

        freqs = np.zeros(count) 
        probs = np.zeros(count) 
        for n in np.arange(0, count):
            freqs[n] = freq_start + delta_freq * n
            probs[n] = self.step(n)
        probs -= scipy.misc.logsumexp(probs) + np.log(delta_freq)
        return (freqs, probs)

    def model(self, freq, t = None):
        if freq == 0.0:
            ((mu_alpha, mu_beta), (sigma_alpha, sigma_beta), y_model, loglik) = self._linreg()
            return (0.0, (0.0, 0.0, mu_alpha, mu_beta), (0.0, 0.0, sigma_alpha, sigma_beta), y_model, loglik)
            
        _, params = self.calc(freq)
        tau, tau2, ss, cc, ys, yc, st, ct, s, c, K, M, N, P, Q = params

        if P == 0.0:
            sigma_beta = 0.0
            mu_beta = 0.0
        else:
            sigma_beta = 1.0/P
            mu_beta = -Q/2.0*sigma_beta

        L = M + N * mu_beta
        
        sigma_alpha = 1.0/K
        mu_alpha = -L/2.0*sigma_alpha
        
        AC = -cc/2.0
        AS = -ss/2.0
        BC = yc - mu_alpha * ct - mu_beta * c        
        BS = ys - mu_alpha * st - mu_beta * s        

        if AC == 0.0:
            sigma_A = 0.0
            mu_A = 0.0
        else:
            sigma_A = 1.0/AC
            mu_A = -BC/2.0*sigma_A

        if AS == 0.0:
            sigma_B = 0.0
            mu_B = 0.0
        else:
            sigma_B = 1.0/AS
            mu_B = -BS/2.0*sigma_B
        
        y_model = np.cos(self.t * 2.0 * np.pi * freq - tau) * mu_A  + np.sin(self.t * 2.0 * np.pi * freq - tau) * mu_B + (self.t - tau2) * mu_alpha + mu_beta
        loglik = self.norm_term - 0.5 * sum(self.w * (self.y - y_model)**2)
        if t == None:
            t = self.t
        if np.any(t != self.t):
            y_model = np.cos(t * 2.0 * np.pi * freq - tau) * mu_A  + np.sin(t * 2.0 * np.pi * freq - tau) * mu_B + (t - tau2) * mu_alpha + mu_beta
        return (tau, tau2, (mu_A, mu_B, mu_alpha, mu_beta), (sigma_A, sigma_B, sigma_alpha, sigma_beta), y_model, loglik)
        
    def fit(self, tau, tau2, freq, A, B, alpha, beta, t = None):
        y_model = np.cos(self.t * 2.0 * np.pi * freq - tau) * A  + np.sin(self.t * 2.0 * np.pi * freq - tau) * B + (self.t - tau2) * alpha + beta
        loglik = self.norm_term - 0.5 * sum(self.w * (self.y - y_model)**2)
        if t == None:
            t = self.t
        if np.any(t != self.t):
            y_model = np.cos(t * 2.0 * np.pi * freq - tau) * A  + np.sin(t * 2.0 * np.pi * freq - tau) * B + (t - tau2) * alpha + beta
        return y_model, loglik
