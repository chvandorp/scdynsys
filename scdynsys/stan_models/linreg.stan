/* simple model for estimating growth / decay rates of T cells in the lung.
 * This is used for the CD90.1 transfer experiment.
 */

data {
    int N;
    vector[N] x, y;
    int Nhat;
    vector[Nhat] xhat;
}

parameters {
    real a, b;
    real<lower=0> sigma;
}

model {
    y ~ normal(a + b * x, sigma);
}

generated quantities {
    vector[Nhat] yhat = a + b * xhat;
    real loglik = normal_lpdf(y | a + b * x, sigma);
}