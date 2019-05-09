import numpy as np
from cov_func import cov_func

class cov_div_free(cov_func):
    
    def __init__(self, sig_var, length_scale, noise_var, toeplitz = False):
        super(cov_div_free, self).__init__(toeplitz=toeplitz)
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
        if calc_grad:
            K_grads = np.zeros((3, np.size(x1), np.size(x2)))
        for i in np.arange(0, np.shape(x1)[0]):
            for j in np.arange(0, np.shape(x2)[0]):
                x_diff = x1[i] - x2[j]
                x_diff_sq = np.dot(x_diff, x_diff)
                for i1 in np.arange(0, dim):
                    i_abs = dim*i + i1
                    for j1 in np.arange(0, dim):
                        j_abs = dim*j + j1
                        #TODO: Simplify for diagonal elements
                        K[i_abs, j_abs] = x_diff[i1] * x_diff[j1] * self.inv_length_scale_sq
                        if calc_grad:
                            K_grads[1, i_abs, j_abs] = -2.0 * x_diff[i1] * x_diff[j1] * self.inv_length_scale_qb
                        if (i1 == j1):
                            K[i_abs, j_abs] += (dim - 1.) - x_diff_sq * self.inv_length_scale_sq
                            if calc_grad:
                                K_grads[1, i_abs, j_abs] += 2.0 * x_diff_sq * self.inv_length_scale_qb
                        exp_fact = np.exp(-0.5 * self.inv_length_scale_sq * x_diff_sq)
                        if calc_grad:
                            K_grads[0, i_abs, j_abs] = K[i_abs, j_abs] * exp_fact
                            K_grads[1, i_abs, j_abs] += K[i_abs, j_abs] * x_diff_sq * self.inv_length_scale_qb
                            K_grads[1, i_abs, j_abs] *= self.sig_var * exp_fact
                        K[i_abs, j_abs] *= self.sig_var * exp_fact
                
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
