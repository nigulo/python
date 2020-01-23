import numpy as np
from cov_func import cov_func

class cov_sq_exp(cov_func):
    
    def __init__(self, sig_var, length_scale, noise_var, dim_out, toeplitz = False, scales = None):
        super(cov_sq_exp, self).__init__(toeplitz=toeplitz)
        self.sig_var = sig_var
        self.length_scale = length_scale
        self.inv_length_scale_sq = 1.0/(length_scale*length_scale)
        self.inv_length_scale_qb = self.inv_length_scale_sq/length_scale
        self.noise_var = noise_var
        self.dim_out = dim_out
        self.scales = scales

    def calc_cov(self, x1, x2, data_or_test, calc_grad = False):
        if calc_grad:
            # We only calculate gradients in the constant noise case
            assert(np.isscalar(self.noise_var))
        K = np.zeros((np.shape(x1)[0]*self.dim_out, np.shape(x2)[0]*self.dim_out))
        assert(x1.shape[1] == x2.shape[1])
        scales = np.ones(x1.shape[1])
        if self.scales is not None:
            assert(len(self.scales) == x1.shape[1])
            for k in np.arange(len(self.scales)):
                if self.scales[k] is not None:
                    scales[k] = self.scales[k]
        # Gradients for sigvar, length_scale and noise_var
        I = np.identity(self.dim_out)
        if calc_grad:
            if self.scales is not None:
                K_grads = np.zeros((3+len(self.scales), np.shape(x1)[0]*self.dim_out, np.shape(x2)[0]*self.dim_out))
            else:
                K_grads = np.zeros((3, np.shape(x1)[0]*self.dim_out, np.shape(x2)[0]*self.dim_out))
        for i in np.arange(0, np.shape(x1)[0]):
            i_abs_1 = self.dim_out*i
            i_abs_2 = i_abs_1 + self.dim_out
            for j in np.arange(0, np.shape(x2)[0]):
                x_diff = (x1[i] - x2[j])*scales
                x_diff_sq = x_diff*x_diff
                x_diff_sq_sum = np.sum(x_diff_sq)
                exp_fact = np.exp(-0.5 * self.inv_length_scale_sq * x_diff_sq_sum)
                j_abs_1 = self.dim_out*j
                j_abs_2 = j_abs_1 + self.dim_out
                K[i_abs_1:i_abs_2, j_abs_1:j_abs_2] = I * exp_fact
                if calc_grad:
                    K_grads[0, i_abs_1:i_abs_2, j_abs_1:j_abs_2] = K[i_abs_1:i_abs_2, j_abs_1:j_abs_2]
                    K_grads[1, i_abs_1:i_abs_2, j_abs_1:j_abs_2] = K[i_abs_1:i_abs_2, j_abs_1:j_abs_2] * x_diff_sq_sum * self.inv_length_scale_qb
                    K_grads[1, i_abs_1:i_abs_2, j_abs_1:j_abs_2] *= self.sig_var
                    if self.scales is not None:
                        for k in np.arange(len(self.scales)):
                            if self.scales[k] is not None:
                                K_grads[3:3+k, i_abs_1:i_abs_2, j_abs_1:j_abs_2] = -K[i_abs_1:i_abs_2, j_abs_1:j_abs_2] * x_diff_sq[k] * scales[k]
                                K_grads[3:3+k, i_abs_1:i_abs_2, j_abs_1:j_abs_2] *= self.sig_var
                                
                K[i_abs_1:i_abs_2, j_abs_1:j_abs_2] *= self.sig_var
                
        if data_or_test:
            assert(np.size(x1) == np.size(x2))
            if (np.isscalar(self.noise_var)):
                K += np.identity(np.shape(x1)[0]*self.dim_out)*self.noise_var
            else:
                K += np.diag(self.noise_var)
            if calc_grad:
                K_grads[2, :, :] += np.identity(np.shape(x1)[0]*self.dim_out)
        if calc_grad:
            return K, K_grads
        else:
            return K

