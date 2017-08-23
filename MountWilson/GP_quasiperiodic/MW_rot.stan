functions {
    real generalized_inverse_gaussian_lpdf(real x, int p, real a, real b) {
        return p * 0.5 * log(a / b) - log(2 * modified_bessel_second_kind(p, sqrt(a * b))) + (p - 1) * log(x) - (a * x + b / x) * 0.5;
    }
    
#    real inverse_gaussian_lpdf(real x, real mu, real lambda) {
#        return generalized_inverse_gaussian_lpdf(x | -0.5, lambda/mu/mu, lambda);
#    }
} 

data {
    int<lower=1> N;
    real x[N];
    vector[N] y;
    vector[N] noise_var_prop;
    real var_y;
    real var_seasonal_means;
    real rot_freq;
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
    real<lower=0> sig_var;
    real<lower=0> noise_var;
    real<lower=0> freq;
    real<lower=0> rot_amplitude;
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
            Sigma[i, j] = sig_var * exp(-0.5 * inv_length_scale * inv_length_scale * pow(x[i] - x[j],2)) * cos(2*pi()*freq*(x[i] - x[j])) + rot_amplitude * cos(2*pi()*rot_freq*(x[i] - x[j])) + trend_var * x[i] * x[j];
            Sigma[j, i] = Sigma[i, j];
        }
    }
    
    for (k in 1:N)
        Sigma[k, k] = sig_var + rot_amplitude + trend_var * x[k] * x[k] + noise_var*noise_var_prop[k]; // + jitter    
    
    L = cholesky_decompose(Sigma);
    
    #freq ~ student_t(4, 0, 0.5);
    #freq ~ cauchy(0, 0.5);
    freq ~ normal(0, 0.167);
    sig_var ~ normal(0, var_seasonal_means);
    noise_var ~ normal(1, 1.0/6);
    #length_scale ~ inv_gamma(3, 4.0/(freq+ 1.0/duration));
    #inv_length_scale ~ normal(0, freq/3.0);
    inv_length_scale ~ beta(1.0, 1.0);
    rot_amplitude ~ normal(0, mean(noise_var_prop)/6);
    trend_var ~ normal(0, var_y/duration/duration);
    m ~ normal(mean_y, sqrt(var_y));
    y ~ multi_normal_cholesky(mu, L);
}
