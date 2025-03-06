/** Linear DA model with a prior on the differentiation matrix.
 * The net growth rate is time dependent.
 *
 * We assume that net growt rates rho are cluster-specific,
 * but decay at a global rate u. So, rho_t = rho_0 e^{-ut} + rho_inf (1-e^{-ut})
 * If Q is the differentation matrix, we then get the ODE
 * dX/dt = (Q + rho_t)X and hence 
 * X(t) = exp(\int_0^t (Q + rho_s) ds) X_0 = exp(Qt + (rho_0 - rho_inf) (1 - e^{-ut})/u + rho_inf t) X_0
 * 
 * FIXME: the matrix exponential solution IS NOT VALID FOR NON-AUTONOMOUS SYSTEMS.
 * We'll have to integrate the ODEs in these cases.
 */

functions {
    // dirichlet-multinomial model
#include functions.stan    
    
    // solution to linear ODE model
    array[] vector indep_inhom_lin_model(array[] real t, real t0, real u, vector rho, vector eta, vector x0) {
        int d = num_elements(x0), n = num_elements(t);
        array[n] vector[d] x;
        for ( i in 1:n ) {
            vector[d] At = (rho-eta) * (1-exp(-u*(t[i]-t0))) / u + eta*(t[i]-t0);
            x[i] = exp(At) .* x0;
        }
        return x;
    }
    
    array[] vector inhom_lin_model_naive_sol(array[] real t, real t0, real u, vector rho, vector eta, matrix Q, vector x0) {
        int d = num_elements(x0), n = num_elements(t);
        array[n] vector[d] x;
        for ( i in 1:n ) {
            matrix[d,d] At = add_diag(Q * (t[i]-t0), (rho-eta) * (1-exp(-u*(t[i]-t0))) / u + eta*(t[i]-t0));
            x[i] = matrix_exp(At) * x0;
        }
        return x;
    }

    vector inhom_lin_model_vf(real t, vector x, real t0, real u, vector rho, vector eta, matrix Q) {
        int d = num_elements(x);
        vector[d] rhot = (rho - eta) * exp(-u*(t-t0)) + eta;
        return rhot .* x + Q * x;
    }

    array[] vector inhom_lin_model(array[] real t, real t0, real u, vector rho, vector eta, matrix Q, vector x0) {
        int d = num_elements(x0), n = num_elements(t);
        array[n] vector[d] x = ode_rk45(inhom_lin_model_vf, x0, t0-1e-3, t, t0, u, rho, eta, Q);
        return x;
    }

    vector inhom_lin_rate(real t, real t0, real u, vector rho, vector eta, matrix Q, vector x) {
        return (rho - eta) * exp(-u*(t-t0)) + eta + (Q * x) ./ x;
    }

    array[] vector sigmoid_lin_model_naive_sol(array[] real t, real t0, real tau, vector rho, vector eta, matrix Q, vector x0) {
        int d = num_elements(x0), n = num_elements(t);
        array[n] vector[d] x; 
        real v = 2.0; // how fast do we switch from rho to eta?
        for ( i in 1:n ) {
            real ht = (log1p_exp(v * (t0-tau)) - log1p_exp(v * (t[i]-tau))) / v;
            matrix[d,d] At = add_diag(Q * (t[i]-t0), (rho-eta) * ht + rho * (t[i]-t0));
            x[i] = matrix_exp(At) * x0;
        }
        return x;
    }

    vector sigmoid_lin_model_vf(real t, vector x, real t0, real tau, vector rho, vector eta, matrix Q) {
        real v = 2.0;
        int d = num_elements(x);
        vector[d] rhot = ((rho-eta) * (1 - inv_logit(v * (t-tau))) + eta);
        return rhot .* x + Q * x;
    }

    array[] vector sigmoid_lin_model(array[] real t, real t0, real tau, vector rho, vector eta, matrix Q, vector x0) {
        int d = num_elements(x0), n = num_elements(t);
        array[n] vector[d] x = ode_rk45(sigmoid_lin_model_vf, x0, t0-1e-3, t, t0, tau, rho, eta, Q);
        return x;
    }    

    array[] vector indep_sigmoid_lin_model(array[] real t, real t0, real tau, vector rho, vector eta, vector x0) {
        int d = num_elements(x0), n = num_elements(t);
        array[n] vector[d] x;
        real v = 2.0; // how fast do we switch from rho to eta?
        for ( i in 1:n ) {
            real ht = (log1p_exp(v * (t0-tau)) - log1p_exp(v * (t[i]-tau))) / v;
            vector[d] At = (rho-eta) * ht + rho * (t[i]-t0);
            x[i] = exp(At) .* x0;
        }
        return x;
    }

    vector sigmoid_lin_rate(real t, real t0, real tau, vector rho, vector eta, matrix Q, vector x) {
        real v = 2.0;
        return (rho - eta) * (1 - inv_logit(v * (t-tau))) + eta + (Q * x) ./ x;
    }
    
    array[] vector lin_model(array[] real t, real t0, vector rho, matrix Q, vector x0) {
        int d = num_elements(x0), n = num_elements(t);
        array[n] vector[d] x;
        for ( i in 1:n ) {
            matrix[d,d] At = add_diag(Q * (t[i]-t0), rho * (t[i]-t0));
            x[i] = matrix_exp(At) * x0;
        }
        return x;
    }

    array[] vector indep_lin_model(array[] real t, real t0, vector rho, vector x0) {
        int d = num_elements(x0), n = num_elements(t);
        array[n] vector[d] x;
        for ( i in 1:n ) {
            vector[d] At = rho * (t[i]-t0);
            x[i] = exp(At) .* x0;
        }
        return x;
    }
    
    vector lin_rate(real t, real t0, vector rho, matrix Q, vector x) {
        return rho + (Q * x) ./ x;
    }
       
    array[] vector inhom_diff_inhom_lin_model_naive_sol(array[] real t, real t0, real u, vector rho, vector eta, real w, matrix Q, vector x0) {
        /* both Q and rho are functions of time */
        int d = num_elements(x0), n = num_elements(t);
        array[n] vector[d] x;
        for ( i in 1:n ) {
            matrix[d,d] At = add_diag(Q * (1 - exp(-w*(t[i]-t0))) / w, (rho-eta) * (1-exp(-u*(t[i]-t0))) / u + eta*(t[i]-t0));
            x[i] = matrix_exp(At) * x0;
        }
        return x;
    }

    vector inhom_diff_inhom_lin_rate(real t, real t0, real u, vector rho, vector eta, real w, matrix Q, vector x) {
        return (rho - eta) * exp(-u*(t-t0)) + eta + (Q * x) ./ x * exp(-w*(t-t0));
    }

    array[] vector inhom_diff_lin_model_naive_sol(array[] real t, real t0, vector rho, real w, matrix Q, vector x0) {
        /* only Q is a function of time */
        int d = num_elements(x0), n = num_elements(t);
        array[n] vector[d] x;
        for ( i in 1:n ) {
            matrix[d,d] At = add_diag(Q * (1 - exp(-w*(t[i]-t0))) / w, rho*(t[i]-t0));
            x[i] = matrix_exp(At) * x0;
        }
        return x;
    }

    vector inhom_diff_lin_rate(real t, real t0, vector rho, real w, matrix Q, vector x) {
        return rho + (Q * x) ./ x * exp(-w*(t-t0));
    }

    
    array[] vector model_dispatch(int dyn_model, int indep, array[] real t, real t0, real u, real tau, vector rho, vector eta, real w, matrix Q, vector x0) {
        if ( dyn_model == 0 ) {
            if ( indep == 0 ) return lin_model(t, t0, rho, Q, x0);
            return indep_lin_model(t, t0, rho, x0);
        } else if ( dyn_model == 1 ) {
            if ( indep == 0 ) return inhom_lin_model(t, t0, u, rho, eta, Q, x0);
            return indep_inhom_lin_model(t, t0, u, rho, eta, x0);
        } else if ( dyn_model == 2 ) {
            if ( indep == 0 ) return sigmoid_lin_model(t, t0, tau, rho, eta, Q, x0);
            return indep_sigmoid_lin_model(t, t0, tau, rho, eta, x0);
        } else if ( dyn_model == 3 ) {
            if ( indep == 0 ) return inhom_diff_lin_model_naive_sol(t, t0, rho, w, Q, x0); // WARNING! using the naive solution
            return indep_lin_model(t, t0, rho, x0); // w is meaningless in this case.        
        } else if ( dyn_model == 4 ) {
            if ( indep == 0 ) return inhom_diff_inhom_lin_model_naive_sol(t, t0, u, rho, eta, w, Q, x0); // WARNING! using the naive solution
            return indep_inhom_lin_model(t, t0, u, rho, eta, x0); // w is meaningless in this case.
        } else {
            reject("invalid dyn_model code");
        }
    }
    
    vector rate_dispatch(int dyn_model, real t, real t0, real u, real tau, vector rho, vector eta, real w, matrix Q, vector x) {
        // don't worry about independence as the Q matrix is 0 in that case.
        if ( dyn_model == 0 ) {
            return lin_rate(t, t0, rho, Q, x);
        } else if ( dyn_model == 1 ) {
            return inhom_lin_rate(t, t0, u, rho, eta, Q, x);
        } else if ( dyn_model == 2 ) {
            return sigmoid_lin_rate(t, t0, tau, rho, eta, Q, x);
        } else if ( dyn_model == 3 ) {
            return inhom_diff_lin_rate(t, t0, rho, w, Q, x);
        } else if ( dyn_model == 4 ) {
            return inhom_diff_inhom_lin_rate(t, t0, u, rho, eta, w, Q, x);
        } else {
            reject("invalid dyn_model code");
        }
    }

    real long_term_growth_rate(int dyn_model, vector rho, vector eta, matrix Q) {
        int d = num_elements(eta);
        matrix[d,d] A = rep_matrix(0.0, d, d);
        if ( dyn_model == 0 ) {
            A = add_diag(Q, rho);
        } else if ( dyn_model == 1 ) {
            A = add_diag(Q, eta);
        } else if ( dyn_model == 2 ) {
            A = add_diag(Q, eta);
        } else if ( dyn_model == 3 ) {
            return max(rho);
        } else if ( dyn_model == 4 ) {
            return max(eta);
        }
        complex_vector[d] eigs = eigenvalues(A);
        vector[d] re_eigs = get_real(eigs);
        return max(re_eigs);
    }
}

data {
    int<lower=0> Nf; // number of mice for freq data
    int<lower=0> C; // number of clusters
    int<lower=0> Nc; // number of mice for count data

    int N; // number of unique time points
    vector[N] T; // unique time points
    
    array[C, Nf] int ClusFreq; // frequency per cluster
    vector[Nc] TotalCounts; // total cell counts
    array[Nf] int<lower=1, upper=N> Idxf; // indices of sampling times for freq data
    array[Nc] int<lower=1, upper=N> Idxc; // indices of sampling times for count data  

    real T0; // initial time
    
    // simulation times
    int Nsim;
    vector[Nsim] Tsim;
    
    // prior on the differentiation matrix
    int<upper=C*(C-1)> Nd; // number of differentiation arrows
    array[2, Nd] int<lower=1, upper=C> IdxDiff; // indices of non-zero differentiation pathways
    
    // covariates for turn-over rate
    int Ncovs; 
    matrix[C, Ncovs] Covariates;
    
    // options
    int<lower=0, upper=1> obs_model_freq; // specify observation model
    int<lower=0, upper=1> eta_nonzero; // allow for a background turnover rate
    int<lower=0, upper=4> dyn_model; // 0 = autonomous, 1 = exp decay, 2 = sigmoid, 3 = exp decay for Q and rho
    int<lower=0, upper=1> signed_diff; // 0 = Q matrix elements all positive, 1 = negative elements mean inverse flow
    
    // manual leave-one-out
    array[Nf] int<lower=0, upper=1> InclFreq;
    
    // prior parameters
    real tau_loc; // mean of the tau parameter in the sigmoid model
    real<lower=0> tau_scale;
    real u_loc;
    real<lower=0> u_scale;
    real w_loc;
    real<lower=0> w_scale;
    real<lower=0> delta_loc_prior;

    // penalty for a long-term positive growth rate
    real growth_rate_penalty;
}


transformed data {
    int<lower=0, upper=Nf> NumInclFreq = num_eq(InclFreq, 1);
    int<lower=0, upper=Nf> NumExclFreq = num_eq(InclFreq, 0);
    
    // get indices of left in and left out units
    array[NumInclFreq] int<lower=1, upper=Nf> IdxInclFreq = indices_eq(InclFreq, 1);
    array[NumExclFreq] int<lower=1, upper=Nf> IdxExclFreq = indices_eq(InclFreq, 0);
    
    // infer if model has independent populations from number of pathways
    int indep = (Nd == 0);
}

parameters {
    vector<multiplier=0.1>[C] rho; // decaying turnover
    real<multiplier=0.1> rho_loc; // hyper-parameters for rho
    real<lower=0> rho_scale;
    vector[Ncovs] rho_weight; // weight of covariate
    
    real<lower=0> u; // decay rate for rho
    real tau; // switch point for rho
    real<lower=0> w; // decay rate for Q
    
    vector<multiplier=0.01>[C] eta_vec; // background turnover
    real<multiplier=0.01> eta_loc; // hyper-parameters for rho
    real<lower=0> eta_scale;
    vector[Ncovs] eta_weight; // weight of covariate
    
    vector<lower=(signed_diff ? -2.5 : 0.0), upper=2.5>[Nd] delta_raw; // differentiation
    real<lower=0> delta_loc; // hyper-parameter for delta

    real logY0;
    simplex[C] p0;
    
    real<lower=0> sigma_c; // measurement error for cell counts 
    real<lower=0> phi_inv_f; // inverse dispersion parameter for freqs
}

transformed parameters {
    vector[C] eta = (eta_nonzero ? eta_vec : rep_vector(0.0, C));
    matrix[C,C] Q = rep_matrix(0.0, C, C);
    vector<lower=0>[C] X0 = exp(logY0) * p0;
    if ( signed_diff == 0 ) { // simple case: just assign delta_raw to the indexed matrix element
        for ( n in 1:Nd ) {
            Q[IdxDiff[1, n], IdxDiff[2, n]] = delta_raw[n];
        }
    } else { // assign negative elements to the negated transpose of the matrix
        for ( n in 1:Nd ) {
            Q[IdxDiff[1, n], IdxDiff[2, n]] += (delta_raw[n] > 0) * delta_raw[n];
            Q[IdxDiff[2, n], IdxDiff[1, n]] -= (delta_raw[n] < 0) * delta_raw[n];
        }   
    }
    Q = add_diag(Q, -transpose(ones_row_vector(C) * Q));

    real rho_inf =  long_term_growth_rate(dyn_model, rho, eta, Q);
}
    
model {
    // solve model at required times
    matrix[C, N] ps;
    vector[N] Ys;
    array[N] vector[C] X = model_dispatch(dyn_model, indep, to_array_1d(T), T0, u, tau, rho, eta, w, Q, X0);
    for ( i in 1:N ) {
        Ys[i] = sum(X[i]);
        ps[:,i] = X[i] / Ys[i];
    }
    
    for ( i in 1:Nf ) {
        if ( InclFreq[i] == 0 ) continue; // skip left-out index
        
        if ( obs_model_freq == 0 ) {
            ClusFreq[:, i] ~ multinomial(ps[:,Idxf[i]]);
        } else {
            ClusFreq[:, i] ~ dirichlet_multinomial(ps[:,Idxf[i]] / phi_inv_f);
        }
    }    
    
    for ( i in 1:Nc ) {
        TotalCounts[i] ~ lognormal(log(Ys[Idxc[i]]), sigma_c);
    }
    
    // prior
    
    sigma_c ~ exponential(1.0);
    phi_inv_f ~ exponential(1e3);
    
    rho ~ normal(rho_loc + Covariates * rho_weight, rho_scale);
    rho_loc ~ normal(0,1);
    rho_scale ~ normal(0,1);
    rho_weight ~ normal(0,1);
    
    u ~ normal(u_loc, u_scale);
    w ~ normal(w_loc, w_scale);
    tau ~ normal(tau_loc, tau_scale);

    
    if ( eta_nonzero && (dyn_model != 0) ) {
        eta_vec ~ normal(eta_loc + Covariates * eta_weight, eta_scale);
    } else {
        eta_vec ~ normal(0, 1);
    }
    eta_loc ~ normal(0, 1);
    eta_scale ~ normal(0, 1);
    eta_weight ~ normal(0, 1);
    
    if ( Nd > 1 ) {
        if ( signed_diff == 0 ) {
            delta_raw ~ exponential(1/delta_loc); // L1
        } else {
            delta_raw ~ double_exponential(0, delta_loc); // L1
        }
    } else if ( Nd == 1 ) {
        if ( signed_diff == 0 ) {
            delta_raw ~ exponential(delta_loc_prior);
        } else {
            delta_raw ~ double_exponential(0, 1/delta_loc_prior);
        }
    }
    delta_loc ~ exponential(delta_loc_prior);
    
    //logX0 ~ normal(0, 10);
    logY0 ~ normal(0, 10); // TESTING

    // penalize an exponentially growing population
    target += -(growth_rate_penalty * fmax(rho_inf, 0.0))^2;
}

generated quantities {
    matrix[C, Nsim] X;
    matrix[C, Nsim] freqs;
    matrix[C, Nsim] rates; // for plottling the time-dependent decay rates rho(t)
    matrix[C, Nf] freqs_sim; // freqs only simulated at observation times
    vector[Nsim] counts;
    vector[Nsim] counts_sim; // also simulate cell counts for intermediate time points
    
    vector[Nf + Nc] log_lik; // concatenated lls for freq data and count data

    // trajectories
    {
        array[Nsim] vector[C] Xsim = model_dispatch(dyn_model, indep, to_array_1d(Tsim), T0, u, tau, rho, eta, w, Q, X0);
        for ( i in 1:Nsim ) {
            real Yi;
            X[:, i] = Xsim[i];
            rates[:, i] = rate_dispatch(dyn_model, Tsim[i], T0, u, tau, rho, eta, w, Q, Xsim[i]);
            Yi = sum(Xsim[i]);
            counts[i] = Yi;
            counts_sim[i] = lognormal_rng(log(Yi), sigma_c);
            freqs[:, i] = Xsim[i] / counts[i];            
        }
    }
    
    // simulate freq data (and log-likelihoods)
    {
        array[N] vector[C] Xsim = model_dispatch(dyn_model, indep, to_array_1d(T), T0, u, tau, rho, eta, w, Q, X0);
        for ( i in 1:Nf ) {
            vector[C] Xi = Xsim[Idxf[i]];
            vector[C] p = Xi / sum(Xi);
            int K = sum(ClusFreq[:,i]);
            array[C] int Y;
            if ( obs_model_freq == 0 ) 
                Y = multinomial_rng(p, K);
            else {
                Y = dirichlet_multinomial_rng(p / phi_inv_f, K);
            }
            for ( j in 1:C ) {
                freqs_sim[j,i] = Y[j] * inv(K);
            }
            // compute log-lik (importantly, also for excluded data)
            if ( obs_model_freq == 0 ) {
                log_lik[i] = multinomial_lpmf(ClusFreq[:,i] | p);
            } else {
                log_lik[i] = dirichlet_multinomial_lpmf(ClusFreq[:,i] | p / phi_inv_f);
            }
        }
        // log-likelihood of count data (simulated at many more intermediate points above)
        for ( i in 1:Nc ) {
            real Yi = sum(Xsim[Idxc[i]]);
            // compute log-lik
            log_lik[Nf + i] = lognormal_lpdf(TotalCounts[i] | log(Yi), sigma_c); 
        }
    }
}