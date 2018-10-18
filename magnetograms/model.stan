data {
    int<lower=1> N;
    vector[2] x[N];
    vector[2] y[N];
    real noise_var;
}

transformed data {
    vector[N*2] y_flat;
    for (i in 1:N) {
        y_flat[2*(i-1)+1] = y[i][1];
        y_flat[2*(i-1)+2] = y[i][2];
    }
    
}

parameters {
    real<lower=0> inv_length_scale;
    real<lower=0> sig_var;
    real m;
    vector[N*2] z;
    simplex[2] theta[N];
}

transformed parameters {
    real<lower=0> length_scale;
    vector[N] mu;
    length_scale = inv(inv_length_scale);
    for (i in 1:(N)) mu[i] = m;
}

model {
    matrix[N*2, N*2] Sigma;
    matrix[N, N] L;
    vector[2] x_diff;
    real x_diff_sq;

    for (i in seq(1, 2*(N-1), by=2)) {
        for (j in seq(i+2, 2*N, by=2)) {
            x_diff = x[i]-x[j];
            x_diff_sq = dot_product(x_diff, x_diff);
            for (i1 in 0:1) {
                int i_abs = 2*i + i1;
                for (j1 in 0:1) {
                    int j_abs = 2*j + j1;
                    Sigma[i_abs, j_abs] = (x_diff[i1+1]*x_diff[j1+1]);
                    if (i1 == j1) {
                        Sigma[i_abs, j_abs] += 1-x_diff_sq;
                    }
                    Sigma[i_abs, j_abs] *= inv_length_scale * inv_length_scale;
                    Sigma[i_abs, j_abs] *= sig_var * exp(-0.5*inv_length_scale * inv_length_scale * x_diff_sq);
                    Sigma[j_abs, i_abs] = Sigma[i_abs, j_abs];
                }
            }
            j += 1;
        }
        i += 1;
    }
    
    for (k in 1:N*2)
        Sigma[k, k] = sig_var + noise_var; // + jitter    
    
    L = cholesky_decompose(Sigma);
    
    sig_var ~ normal(0, 1);
    inv_length_scale ~ normal(0, 1.0/3);
    m ~ normal(0, 1);

    real ps[2];    
    for (n in 1:N){
        ps[1] = log(theta[1])+multi_normal_cholesky_lpdf(y_flat[n] | mu, L); //increment log probability of the gaussian
        ps[2] = log(theta[2])+multi_normal_cholesky_lpdf(-y[n] | mu, L); //increment log probability of the gaussian
        target += log_sum_exp(ps);
    }
     
    z ~ multi_normal_cholesky(mu, L);
    y_flat = z;
}

generated_quantities {
}
