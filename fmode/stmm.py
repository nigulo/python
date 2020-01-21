'''
Student-t mixture model
Based on Lo 2009 - Statistical Methods for High Throughput Genomics
'''
import numpy as np
import numpy.linalg as la
import scipy.special as special
import sys
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

'''
Some notations used in documentation:
    D: dimensionality of data
    N: number of datapoints
    G: number of clusters    
'''
class stmm:
    
    '''
        nu: degrees of freedom of Student-t distribution
        X: D x N matrix of data (each datapoint is a column vector)
        num_clusters: number of clusters
    '''
    def __init__(self, nu, X, num_clusters = None):
        self.nu = nu
        self.X = X
        if num_clusters is not None:
            self.set_num_clusters(num_clusters)
            
        D = self.X.shape[0] # number of dimensions
        self.const_fact = -.5*np.log(np.pi*self.nu)*D + special.gammaln(.5*(self.nu+D)) - special.gammaln(.5*self.nu)
            
        
    '''
        num_clusters: number of clusters
        cluster_weights: list or array of length num_clusters
        cluster_means: D x G matrix
        cluster_covs: D x D x G tensor
    '''
    def set_params(self, num_clusters, cluster_weights=None, cluster_means=None, cluster_covs=None):
        self.G = num_clusters

        D = self.X.shape[0] # number of dimensions
        N = self.X.shape[1] # number of data points

        if cluster_weights is None:
            self.w = np.ones(self.G)/self.G
        else:
            assert(self.G == len(cluster_weights))
            self.w = np.array(cluster_weights)

        if cluster_means is None:
            r = np.random.choice(N, size=self.G)
            self.m = self.X[:,r] # initialise the centers to random datapoints
        else:
            assert(self.G == cluster_means.shape[1])
            self.m = np.array(cluster_means)
            
        if cluster_covs is None:
            s2 = np.mean(np.diag(np.cov(self.X.T)))
            self.S = np.tile((s2*np.eye(D))[:, :, np.newaxis], (1, 1, self.G))
        else:
            assert(self.G == cluster_covs.shape[2])
            self.S = np.array(cluster_covs)
    
    def normalize(self, probs):
        p = probs + sys.float_info.epsilon # in case all probabilities are zero
        return p / np.sum(p, axis=0)
        
    '''
    Calculates cluster responsibilities
    Input:
        log_probs: N x G matrix of log-probabilities of belonging to each cluster per data point (returned by calc_log_probs)
    Returns:
        G x N matrix of cluster responsibilities for each data point 
    '''
    def calc_responsibilities(self, log_probs):
        log_probs = log_probs.T
    
        p = np.exp(log_probs)
        p = self.normalize(p)
    
        return p

    '''
    Log-likelihood of each data point under a Student-t Mixture Model
    Returns:
        log_probs: N x G matrix of log-probabilities of belonging to each cluster per data point
        logliks: array of size N containing log likelihoods of each datapoint
        u: N x G matrix of weights for normal-gamma compound parametrization for t-distribution
    '''
    def calc_log_probs(self):
        D = self.X.shape[0] # number of dimensions
        N = self.X.shape[1] # number of data points
        
        log_probs = np.zeros((N, self.G))
        # weights for normal-gamma compound parametrization for t-distribution
        u = np.zeros((N, self.G))
        logliks = np.zeros(N)
        for i in np.arange(0, self.G):
            inv_Si = la.inv(self.S[:, :, i])
            sign, logdet_Si=la.slogdet(self.S[:, :, i])
            assert(sign > 0) # We must be dealing with positive definite matrix
            logdet_Si = sign*logdet_Si
            mi = self.m[:, i]
            for n in np.arange(0, N):
                v = self.X[:, n] - mi
                vSv = np.dot(v.T, np.dot(inv_Si, v))
                log_probs[n, i] = -.5*(self.nu + D)*np.log(1.+ vSv/self.nu) - .5*logdet_Si + np.log(self.w[i]) + self.const_fact
                u[n, i] = (self.nu + D)/(self.nu + vSv)
        for n in np.arange(0, N):
            logliks[n] = special.logsumexp(log_probs[n, :], b = np.ones(self.G))
        return log_probs, logliks, u
    
    '''
    Fit a mixture of Gaussians to the data X using EM
    
    Inputs:
        num_iter: number of EM iterations
    
    Outputs:
        w: learned mixture coefficients (array of size G)
        m: learned cluster means (D x G matrix)
        S: learned cluster covariances (D x D x G tensor)
        loglik: log-likelihood of data
        z: G x N matrix of responsibilities (mixture assignment probabilties)
    '''
    def calc(self, num_iter=100):
    
        D = self.X.shape[0] # number of dimensions
        N = self.X.shape[1] # number of data points
        
        loglik = np.zeros(num_iter)
        for i in np.arange(num_iter):
            log_probs = np.zeros((N, self.G))
            # E-step:
                    
            log_probs, logliks, u = self.calc_log_probs()
            loglik[i] = np.sum(logliks)
            
            z = self.calc_responsibilities(log_probs) # responsibilities
            #print(u)
            zu = z.T*u
            zu_norm = self.normalize(zu)
            
            # M-step:
            Ng = np.sum(z, axis=1)
            for j in np.arange(self.G): # now get the new parameters for each component
                tmp = (self.X - np.tile(self.m[:, j][:, np.newaxis], (1, N)))*np.tile(np.sqrt(zu[:, j].T), (D, 1))
                S_new = np.dot(tmp, tmp.T)/Ng[j]
                det = la.det(S_new)
                #print("det", det)
                if det > 1e-5: # don't accept too low determinant
                    self.S[:, :, j] = S_new
                else:
                    print("Discarding change in S due to too low determinant")
            self.m = np.dot(self.X, zu_norm)
            self.w = Ng / N
            #print(self.m)
        return self.w, self.m, self.S, loglik[-1], z


def main():

    if len(sys.argv) > 1:
        file_name = sys.argv[1]

    max_num_clusters = 5
    if len(sys.argv) > 2:
        max_num_clusters = int(sys.argv[2])
        
    assert(max_num_clusters > 0 and max_num_clusters <= 5) # No more colors defined
    cluster_colors = ['blue', 'red', 'green', 'peru', 'purple']
    
    X = np.loadtxt(file_name, usecols=(0,1), skiprows=1)
    
    X = X.T;
    
    logliks = np.zeros(max_num_clusters)
    bics = np.zeros(max_num_clusters)
    ms = []
    Ss = []
    zs = []
    stmm_ = stmm(nu=1, X=X)
    for G in np.arange(0, max_num_clusters):
        stmm_.set_params(G+1)        
        w, m, S, loglik, z = stmm_.calc()
        logliks[G] = loglik
        num_params = np.prod(np.shape(w))+np.prod(np.shape(m))+np.prod(np.shape(S))
        bic = np.log(np.prod(np.shape(X))) * num_params - 2.*loglik
        bics[G] = bic
        ms.append(m)
        Ss.append(S)
        zs.append(z)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(12, 6)
    ax.plot(np.arange(max_num_clusters), bics, 'ko-')
    ax.set_xlabel('Number of mixture components')
    ax.set_ylabel('BIC')
    ax.set_title('Model Selection')
    fig.savefig("BIC.png")    


    ###########################################################################
    # Select the optimal model
    min_G = np.argmin(bics) # select the number of mixture components which minimizes the BIC 
    min_bic = bics[min_G]
    max_loglik = logliks[min_G]
    opt_m = ms[min_G]
    opt_S = Ss[min_G]
    opt_z = zs[min_G]
    
    print(bics)
    print('Data likelihood and BIC: %f %f\n' % (max_loglik, min_bic))
       
    ###########################################################################
    # Plot the results for optimal model
    fig = plt.figure(figsize=(6.5, 6.5))
    fig.add_axes([0.12,0.12,0.8,0.8])
    [ax] = fig.axes
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Clusters')
    
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    
    opt_m = opt_m.T
    opt_S = opt_S.T
    X = X.T

    #print(opt_z)
    point_clusters = np.argmax(opt_z, axis=0)
    #cluster_points = get_cluster_points(m, S, X)
    #print("cluster_points", cluster_points.keys())
    
    for cluster_no in np.arange(opt_m.shape[0]):
        m = opt_m[cluster_no]
        s = opt_S[cluster_no]
        print("Cluster " + str(cluster_no) + "mean: " + str(m))
        print("Cluster " + str(cluster_no) + "covariance: " + str(s))
        if m.shape > 2:
            continue

        # Plot cluster centers
        if m.shape == 1:
            #ax.plot(m, m[1], cluster_colors[cluster_no][0]+'+', markersize=20, markeredgewidth=2)
            # Plot 2-sigma ranges representing Gaussians approximation
            ax.axvspan(m - 2.* np.sqrt(s), m + 2.* np.sqrt(s), alpha=0.5, color=cluster_colors[cluster_no])
        elif m.shape == 2:
            ax.plot(m[0], m[1], cluster_colors[cluster_no][0]+'+', markersize=20, markeredgewidth=2)
            w, v = la.eig(s)
            
            cos = v[0,0]
            sin = v[1,0]
            angle = np.arccos(cos)
            if sin < 0.0:
                angle = -angle
            
            
            # Plot cluster ellipses representing Gaussian approximation
            e = Ellipse(xy=m, width=2*np.sqrt(w[0]), height=2*np.sqrt(w[1]), angle=angle*180/np.pi, linestyle=None, linewidth=0)
            ax.add_artist(e)
            e.set_alpha(0.25)
            e.set_facecolor(cluster_colors[cluster_no])

        # Regression coefficients computed directly using the covariance of the cluster
        #slope = s[0,1]/s[0,0]
        #intercept = m[1] - slope*m[0]

        #print(point_clusters)
        #print(point_clusters == cluster_no)
        points = X[point_clusters == cluster_no, :]
        if len(points > 0): # Due to hard selection some clusters might be empty

            # Plot points belonging to the cluster
            ax.plot(points[:,0], points[:,1], cluster_colors[cluster_no][0]+'.')
            
            # Regression coefficients calculated independently (not used anywhere)
            #slope2, intercept2, r_value, p_value, std_err = stats.linregress(points[:,0], points[:,1])
            
            # Double-check that both lead to the same results
            #np.testing.assert_almost_equal(slope, slope2, 1)
            #np.testing.assert_almost_equal(intercept, intercept2, 1)
            
            # Plot regression lines
            #x_test = np.linspace(np.min(points[:,0]), np.max(points[:,0]))
            #ax.plot(x_test, x_test * slope + intercept, color=cluster_colors[cluster_no][0], linestyle='-', linewidth=1)
            #ax.plot(x_test, x_test * slope2 + intercept2, color=cluster_colors[cluster_no][0], linestyle='--', linewidth=1)
        
        
    ###############################################################################
    
    
    fig.savefig("clusters.png")
    plt.close(fig)
    
if __name__ == "__main__":
    main()
