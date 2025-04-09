/** Linear DA model fit to cell counts only
 * The net growth rate can be time dependent.
 */

functions {    
    // solution to linear ODE model
    vector inhom_lin_model(real t, real t0, real u, vector rho, vector eta, vector x0) {
        int d = num_elements(x0);
        vector[d] At = (rho - eta) * (1-exp(-u*(t-t0))) / u + eta * (t-t0);

        return exp(At) .* x0;
    }
    
    vector lin_model(real t, real t0, vector rho, vector x0) {
        int d = num_elements(x0);
        vector[d] At = rho * (t-t0);

        return exp(At) .* x0;
    }
}

data {
    int<lower=0> C; // number of clusters
    int<lower=0> Nc; // number of mice for count data

    int N; // number of unique time points
    vector[N] T; // unique time points
    
    vector[Nc] TotalCounts; // total cell counts
    array[Nc] int<lower=1, upper=N> Idxc; // indices of sampling times for count data  

    real T0; // initial time
    
    // simulation times
    int Nsim;
    vector[Nsim] Tsim;
        
    int<lower=0, upper=1> u_nonzero; // if u == 0, we have the time-homogeneous model
    int<lower=0, upper=1> eta_nonzero; // allow for a background turnover rate
}

parameters {
    ordered[C] rho; // decaying turnover
    real<lower=0> u; // decay rate of rho
    vector[C] eta; // background turnover
    
    vector[C] logX0; // initial state at time T0
    
    real<lower=0> sigma_c; // measurement error for cell counts 
}

transformed parameters {
    vector<lower=0>[C] X0 = exp(logX0);
}
    
model {
    // solve model at required times
    matrix[C, N] Xs;
    for ( i in 1:N ) {
        if ( u_nonzero == 1 ) {
            Xs[:,i] = inhom_lin_model(T[i], T0, u, rho, eta, X0);
        } else {
            Xs[:,i] = lin_model(T[i], T0, rho, X0);
        }
    }
    
    for ( i in 1:Nc ) {
        real X = sum(Xs[:, Idxc[i]]);
        TotalCounts[i] ~ lognormal(log(X), sigma_c);
    }
    
    // prior
    
    sigma_c ~ exponential(1.0);
    rho ~ normal(0, 1);
    u ~ normal(0,1);
    eta ~ normal(0,1);
    logX0 ~ normal(0, 10);
}

generated quantities {
    matrix[C, Nsim] X;
    matrix[C, Nsim] freqs;
    vector[Nsim] counts;
    vector[Nsim] counts_sim; // also simulate cell counts for intermediate time points
    
    vector[Nc] log_lik;
       
    // trajectories
    for ( i in 1:Nsim ) {
        if ( u_nonzero == 1 ) {
            X[:, i] = inhom_lin_model(Tsim[i], T0, u, rho, eta, X0);
        } else {
            X[:, i] = lin_model(Tsim[i], T0, rho, X0);
        }
        real Xi = sum(X[:,i]);
        counts[i] = Xi;
        counts_sim[i] = lognormal_rng(log(Xi), sigma_c);
        freqs[:, i] = X[:,i] / Xi;            
    }
    
    // log-likelihood of count data
    for ( i in 1:Nc ) {
        real Xi = (
            u_nonzero == 1 ?
            sum(inhom_lin_model(T[Idxc[i]], T0, u, rho, eta, X0)) :
            sum(lin_model(T[Idxc[i]], T0, rho, X0))
        );    
        // compute log-lik
        log_lik[i] = lognormal_lpdf(TotalCounts[i] | log(Xi), sigma_c); 
    }
}