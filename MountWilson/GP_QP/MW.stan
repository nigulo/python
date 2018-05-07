data {
    int<lower=1> N;
    real x[N];
    vector[N] y;
    vector[N] noise_var;
    real var_y;
    real var_seasonal_means;
    real prior_freq_mean;
    real prior_freq_std;
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
    real<lower=0> inv_length_scale;
    //real<lower=0> inv_length_scale2;
    real<lower=0> sig_var;
    real<lower=0> freq;
    real<lower=0> trend_var;
    real m;
}

transformed parameters {
    real<lower=0> length_scale;
    //real<lower=0> length_scale2;
    vector[N] mu;
    //length_scale = inv(inv_length_scale);
    length_scale = inv(inv_length_scale * freq);
    //length_scale2 = inv(inv_length_scale2);
    for (i in 1:(N)) mu[i] = m;
}

model {
    matrix[N, N] Sigma;
    matrix[N, N] L;

    for (i in 1:(N-1)) {
        for (j in (i+1):N) {
            Sigma[i, j] = sig_var * exp(-0.5*inv_length_scale * inv_length_scale / 4.0 * freq * freq * pow(x[i] - x[j],2)) * cos(2.0*pi()*freq*(x[i] - x[j])) + trend_var * x[i] * x[j];
            //Sigma[i, j] = sig_var * exp(-0.5*inv_length_scale * inv_length_scale * pow(x[i] - x[j],2)) * cos(2.0*pi()*freq*(x[i] - x[j])) + trend_var * x[i] * x[j];
            //Sigma[i, j] = sig_var * exp(-0.5*inv_length_scale * inv_length_scale * pow(x[i] - x[j],2)) * exp(-2.0*sin(pi()*freq*(x[i] - x[j]))^2*inv_length_scale2*inv_length_scale2) + trend_var * x[i] * x[j];
            //Sigma[i, j] = sig_var * exp(-0.5*inv_length_scale * inv_length_scale * freq * freq * pow(x[i] - x[j],2)) * exp(-2.0*sin(pi()*freq*(x[i] - x[j]))^2*inv_length_scale2*inv_length_scale2) + trend_var * x[i] * x[j];
            Sigma[j, i] = Sigma[i, j];
        }
    }
    
    for (k in 1:N)
        Sigma[k, k] = sig_var + trend_var * x[k] * x[k] + noise_var[k]; // + jitter    
    
    L = cholesky_decompose(Sigma);
    
    freq ~ normal(prior_freq_mean, prior_freq_std);
    sig_var ~ normal(var_seasonal_means, var_seasonal_means);
    //inv_length_scale ~ normal(0, freq/3.0);
    inv_length_scale ~ beta(1, 3);
    //inv_length_scale ~ normal(0, 0.5/3);
    //inv_length_scale2 ~ normal(0, 0.5/3);
    trend_var ~ normal(0, var_y/duration/duration);
    m ~ normal(mean_y, sqrt(var_y));
    y ~ multi_normal_cholesky(mu, L);
}
