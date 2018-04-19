import numpy as np
import scipy.misc as misc

class sampler():
    
    def __init__(self, loglik_fn, greedy = True, condition_fn = None, initial_indices = None):
        self.params_values = []
        self.indices = []
        self.current_param = 0
        self.iteration = 0
        self.loglik_fn = loglik_fn
        self.max_loglik = None
        self.greedy = greedy
        self.logliks = []
        self.condition_fn = condition_fn
        self.initial_indices = initial_indices

    def add_parameter_values(self, param_values):
        self.params_values += param_values
        for i in np.arange(0, len(param_values)):
            self.logliks.append(np.zeros(len(param_values[i])))
            if self.greedy:
                if self.initial_indices is not None and self.initial_indices[i] is not None:
                    self.indices.append(self.initial_indices[i])
                else:
                    self.indices.append(np.random.randint(len(param_values[i])))
            else:
                self.indices.append(0)
                
    def init(self):
        self.indices = np.asarray(self.indices)
        if self.greedy:
            self.params_order = np.random.choice(len(self.params_values), size=len(self.params_values), replace=False)
            self.indices[self.params_order[self.current_param]] = 0
        else:
            self.params_order = np.arange(0, len(self.params_values))
        
    def sample(self):
        params_sample = []
        for i in np.arange(0, len(self.params_values)):
            params_sample.append(self.params_values[i][self.indices[i]])

        loglik = None
        if self.max_loglik is None or self.condition_fn is None or self.condition_fn(params_sample):
            y_means, loglik = self.loglik_fn(params_sample)
            self.logliks[self.params_order[self.current_param]][self.indices[self.params_order[self.current_param]]] = loglik
            if (self.max_loglik is None or loglik > self.max_loglik):
                self.max_loglik = loglik
                self.best_indices = np.array(self.indices)
                self.best_y_mean = y_means
        #else:
        #    print "Skipping ", params_sample

        if self.greedy:
            i_s = [self.current_param]
        else:
            i_s = np.arange(0, len(self.params_values))

        done = True
        for i in i_s:
            index = self.indices[self.params_order[i]]
            index += 1
            if index >= len(self.params_values[self.params_order[i]]):
                if self.greedy:
                    self.indices = np.array(self.best_indices)
                else:
                    self.indices[self.params_order[i]] = 0
            else:
                self.indices[self.params_order[i]] = index
                done = False
                break
        if done:
            if self.greedy:
                self.current_param += 1
                if self.current_param >= len(self.params_values):
                    self.current_param = 0
                    self.iteration += 1
                self.indices[self.params_order[self.current_param]] = 0
            else:
                self.iteration += 1

        return params_sample, loglik
        
    def get_iteration(self):
        return self.iteration

    def get_results(self):
        
        params_mode = []
        params_mean = []
        params_sigma = []
        for i in np.arange(0, len(self.params_values)):
            if len(self.params_values[i]) > 1:
                if self.best_indices[i] == 0 :
                    print "WARNING, optimal value for parameter " + str(i) + " at the lower boundary of the grid"
                elif self.best_indices[i] == len(self.params_values[i]) - 1:
                    print "WARNING, optimal value for parameter " + str(i) + " at the upper boundary of the grid"
            param_value = self.params_values[i][self.best_indices[i]]
            params_mode.append(param_value)
            log_probs = np.array(self.logliks[i])
            log_probs -= misc.logsumexp(log_probs)
            probs = np.exp(log_probs)
            probs /= sum(probs)
            #mean = np.exp(scipy.misc.logsumexp(np.log(freqs_m)+probs_m))
            #sigma = np.sqrt(np.exp(scipy.misc.logsumexp(2*np.log(freqs_m-best_freq) + probs_m)))
            mean = sum(self.params_values[i]*probs)
            sigma = np.sqrt(sum((self.params_values[i]-param_value)**2 * probs))
            params_mean.append(mean)
            params_sigma.append(sigma)
            if len(self.params_values[i]) > 1:
                if sigma < max(self.params_values[i][1:] - self.params_values[i][:-1]):
                    print "WARNING, grid for parameter " + str(i) + " too sparse"
        return params_mode, params_mean, params_sigma, self.best_y_mean, self.max_loglik