data {
    int<lower=1> N;
    real x[N];
    vector[N] y;
    vector[N] noise_var;
    real var_y;
    real sig_var_prior_var;
    real inv_length_scale_prior_var;
    real inv_length_scale_max;
}

transformed data {
    real duration;
    real mean_x;
    real range;
    real mean_y;
    duration = max(x) - min(x);
    mean_x = (max(x) + min(x)) / 2;
    range = max(y) - min(y);
    mean_y = mean(y);
}

parameters {
    real<lower=0, upper=inv_length_scale_max> inv_length_scale;
    real<lower=0> sig_var;
    real<lower=0> trend_var;
    real m;
}

transformed parameters {
    real<lower=0> length_scale;
    vector[N] mu;
    length_scale = inv(inv_length_scale);
    for (i in 1:(N)) mu[i] = m;
}

model {
    matrix[N, N] Sigma;
    matrix[N, N] L;

    for (i in 1:(N-1)) {
        for (j in (i+1):N) {
            Sigma[i, j] = sig_var * exp(-0.5*inv_length_scale * inv_length_scale * pow(x[i] - x[j],2)) + trend_var * x[i] * x[j];
            Sigma[j, i] = Sigma[i, j];
        }
    }
    
    for (k in 1:N)
        Sigma[k, k] = sig_var + trend_var * x[k] * x[k] + noise_var[k]; // + jitter    
    
    L = cholesky_decompose(Sigma);
    
    sig_var ~ normal(0, sig_var_prior_var);
    inv_length_scale ~ normal(0, inv_length_scale_prior_var);
    trend_var ~ normal(0, var_y/duration/duration);
    m ~ normal(mean_y, sqrt(var_y));
    y ~ multi_normal_cholesky(mu, L);
}
