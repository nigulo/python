import numpy as np

class sampler():
    
    def __init__(self, loglik_fn, greedy = True):
        self.params_values = []
        self.indices = []
        self.current_param = 0
        self.iteration = 0
        self.loglik_fn = loglik_fn
        self.max_loglik = None
        self.greedy = greedy

    def add_parameter_values(self, param_values):
        self.params_values += param_values
        for i in np.arange(0, len(param_values)):
            if self.greedy:
                self.indices.append(np.random.randint(len(param_values[i])))
            else:
                self.indices.append(0)
    
    def init(self):
        if self.greedy:
            self.params_order = np.random.choice(len(self.params_values), size=len(self.params_values), replace=False)
            self.indices[self.params_order[self.current_param]] = 0
        else:
            self.params_order = np.arange(0, len(self.params_values))
        
    def sample(self):
        params_sample = []
        for i in np.arange(0, len(self.params_values)):
            params_sample.append(self.params_values[i][self.indices[i]])

        y_means, loglik = self.loglik_fn(params_sample)
        if self.max_loglik is None or loglik > self.max_loglik:
            self.max_loglik = loglik
            self.best_indices = self.indices
            self.best_y_mean = y_means

        if self.greedy is None:
            i_s = [self.current_param]
        else:
            i_s = np.arange(0, len(self.params_values))

        done = True
        for i in i_s:
            index = self.indices[self.params_order[i]]
            index += 1
            if index >= len(self.params_values[self.params_order[i]]):
                if self.greedy:
                    self.indices = self.best_indices
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

    def get_best_sample(self):
        params_sample = []
        for i in np.arange(0, len(self.params_values)):
            params_sample.append(self.params_values[i][self.best_indices[i]])
        return params_sample, self.best_y_mean