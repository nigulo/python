data {
    int<lower=1> N;
    real x[N];
    vector[N] y;
    vector[N] noise_var;
    real var_y;
    real var_seasonal_means;
    real freq_mu;
    real freq_sigma;
    int<lower=1> M;
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
    //real<lower=1> length_scale;
    real<lower=0> inv_length_scale;
    real<lower=0> sig_var;
    real<lower=1.5/duration, upper=0.5> freq;
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
    matrix[M, M] Sigma_mm;
    matrix[N, M] Sigma_nm;
    matrix[M, N] Sigma_mn;
    matrix[M, N] Sigma_mn_inv;
    vector[N] Inv_noise_var;
    //matrix[N, N] Inv_noise_var_n;
    matrix[N, N] Omega;
    //matrix[N, N] Omega_test;
    real val;

    for (i in 1:(N-1)) {
        for (j in (i+1):N) {
            val = sig_var * exp(-0.5*inv_length_scale * inv_length_scale * freq * freq * pow(x[i] - x[j],2)) * cos(2*pi()*freq*(x[i] - x[j])) + trend_var * x[i] * x[j];
            //val = sig_var * exp(-0.5 * pow((x[i] - x[j])*inv_length_scale,2)) * cos(2*pi()*freq*(x[i] - x[j])) + trend_var * x[i] * x[j];
            //val = sig_var * exp(-0.5 * pow((x[i] - x[j])/length_scale*freq,2)) * cos(2*pi()*freq*(x[i] - x[j])) + trend_var * x[i] * x[j];
            if (i <= M && j <= M) {
                Sigma_mm[i, j] = val;
                Sigma_mm[j, i] = val;
                Sigma_nm[i, j] = val;
                Sigma_nm[j, i] = val;
                Sigma_mn[i, j] = val;
                Sigma_mn[j, i] = val;
            } else if (i <= M) {
                Sigma_nm[j, i] = val;
                Sigma_mn[i, j] = val;
            }
        }
    }
    
    Inv_noise_var = rep_vector(1, N)./noise_var;
    //Inv_noise_var_n = diag_matrix(Inv_noise_var);
    
    for (k in 1:M) {
        val = sig_var + trend_var * x[k] * x[k];// + noise_var[k]; // + jitter    
        Sigma_mm[k, k] = val;
        Sigma_nm[k, k] = val;
        Sigma_mn[k, k] = val;
    }

    //Sigma_mn = Sigma_nm';
    for (i in 1:M) {
        for (j in 1:N) {
            Sigma_mn_inv[i, j] = Sigma_mn[i, j] * Inv_noise_var[j];
        }
    }
   
    Omega = Sigma_nm*inverse_spd(Sigma_mm + Sigma_mn_inv*Sigma_nm)*Sigma_mn_inv;
    for (i in 1:N) {
        Omega[i, i] = 1.0 - Omega[i, i];
    }
    for (i in 1:N) {
        for (j in 1:N) {
            // This is needed because we messed up the signs of the off-diagonal elements in previous loop
            if (i == j) {
                Omega[i, j] = Inv_noise_var[i] * Omega[i, j];
            } else {
                Omega[i, j] = -Inv_noise_var[i] * Omega[i, j];
            }
        }
    }
    
    //Omega_test = Inv_noise_var_n-Inv_noise_var_n*Sigma_nm*inverse_spd(Sigma_mm + Sigma_mn*Inv_noise_var_n*Sigma_nm)*Sigma_mn*Inv_noise_var_n;
    //Omega = Omega_test;
    //print(max(Omega - Omega_test));
    
    //Omega = inverse_spd(diag_matrix(noise_var)+Sigma_nm*inverse(Sigma_mm)*Sigma_mn);
    //print(max((Omega_test - inverse_spd(diag_matrix(noise_var)+Sigma_nm*inverse(Sigma_mm)*Sigma_mn))./max(Omega)));
    
    //Omega = (Omega + Omega')/2;
    
    for (i in 1:(N-1)) {
        for (j in (i+1):N) {
            val = Omega[i,j];
            Omega[i,j] = (Omega[i,j]+Omega[j,i])*0.5;
            Omega[j,i] = Omega[i,j];
        }
    }
    
    //if (min(eigenvalues_sym(Omega)) < 0) {
    //    print("JAMAAAAA");
    //}
    
    
    //freq ~ student_t(4, 0, 0.5);
    //freq ~ cauchy(0, 0.5);
    //freq ~ normal(0, 0.167);
    freq ~ normal(freq_mu, freq_sigma);
    sig_var ~ normal(0, var_seasonal_means);
    //length_scale ~ cauchy(1.0/(freq+ 1.0/duration), duration);
    //inv_length_scale ~ normal(0, 0.5/3.0);
    inv_length_scale ~ beta(2, 5);
    trend_var ~ normal(0, var_y/duration/duration);
    m ~ normal(mean_y, sqrt(var_y));
    y ~ multi_normal_prec(mu, Omega);
}
