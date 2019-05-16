import numpy as np
from cov_func import cov_func

class cov_sq_exp(cov_func):
    
    def __init__(self, sig_var, length_scale, noise_var, toeplitz = False):
        super(cov_sq_exp, self).__init__(toeplitz=toeplitz)
        self.sig_var = sig_var
        self.length_scale = length_scale
        self.inv_length_scale_sq = 1.0/(length_scale*length_scale)
        self.inv_length_scale_qb = self.inv_length_scale_sq/length_scale
        self.noise_var = noise_var

    def calc_cov(self, x1, x2, data_or_test, calc_grad = False):
        if calc_grad:
            # We only calculate gradients in the constant noise case
            assert(np.isscalar(self.noise_var))
        K = np.zeros((np.size(x1), np.size(x2)))
        assert(x1.shape[1] == x2.shape[1])
        dim = x1.shape[1]
        # Gradients for sigvar, length_scale and noise_var
        I = np.identity(dim)
        if calc_grad:
            K_grads = np.zeros((3, np.size(x1), np.size(x2)))
        for i in np.arange(0, np.shape(x1)[0]):
            i_abs_1 = dim*i
            i_abs_2 = i_abs_1 + dim
            for j in np.arange(0, np.shape(x2)[0]):
                x_diff = x1[i] - x2[j]
                x_diff_sq = np.dot(x_diff, x_diff)
                exp_fact = np.exp(-0.5 * self.inv_length_scale_sq * x_diff_sq)
                j_abs_1 = dim*j
                j_abs_2 = j_abs_1 + dim
                K[i_abs_1:i_abs_2, j_abs_1:j_abs_2] = I * exp_fact
                if calc_grad:
                    K_grads[0, i_abs_1:i_abs_2, j_abs_1:j_abs_2] = K[i_abs_1:i_abs_2, j_abs_1:j_abs_2]
                    K_grads[1, i_abs_1:i_abs_2, j_abs_1:j_abs_2] = K[i_abs_1:i_abs_2, j_abs_1:j_abs_2] * x_diff_sq * self.inv_length_scale_qb
                    K_grads[1, i_abs_1:i_abs_2, j_abs_1:j_abs_2] *= self.sig_var
                K[i_abs_1:i_abs_2, j_abs_1:j_abs_2] *= self.sig_var
                
        if data_or_test:
            assert(np.size(x1) == np.size(x2))
            if (np.isscalar(self.noise_var)):
                K += np.identity(np.size(x1))*self.noise_var
            else:
                K += np.diag(self.noise_var)
            if calc_grad:
                K_grads[2, :, :] += np.identity(np.size(x1))
        if calc_grad:
            return K, K_grads
        else:
            return K

