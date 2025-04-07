from scdynsys.vae import dyn
import torch


class TestDyn:
    def test_hom_dyn_model(self):
        ts = torch.tensor([0.0, 1.0, 2.0])
        rho = torch.tensor([-0.5, -0.1])
        x0 = torch.tensor([2.0, 1.0])
        logweights, logyhat = dyn.dynamic_model_hom(ts, rho, x0.log())
        
        assert logweights.shape == (3, 2), "weights have incorrect shape"
        
        assert logyhat.shape == (3,), "counts have incorrect shape"
        
        ## TODO: validate results
        
        
    def test_lin_iterpolation(self):
        ts = torch.arange(0, 1, 0.01)

        ## knots at time points
        knots = torch.tensor([0, 0.2, 0.25, 0.7, 1.0])

        ## values for each of the clusters
        vals = torch.tensor([
            [1.0, 2.0, 2.1, 1.5, 0.7],
            [1.4, 1.6, 2.3, 2.5, 2.8],
            [2.4, 2.1, 1.3, 0.2, 0.9],
        ])
        
        # let all weights sum to one
        vals /= vals.sum(axis=0)

        ## test that interpolated values also add to 1.0
        xs = dyn.lin_interpolate(ts, knots, vals)

        sxs = xs.sum(axis=0)

        assert torch.all(torch.isclose(sxs, torch.ones_like(sxs))), "all values should sum to 1"
        
        
    def test_max_growthrate(self):
        # test with diagonal matrix
        eta = torch.tensor([0.1, 0.2, 0.3])
        Q = torch.zeros(eta.shape[0], eta.shape[0])

        max_growth = dyn.max_growthrate_diff(eta, Q)

        assert torch.isclose(max_growth, eta.max(), atol=1e-6), "max growth rate should be equal to max(eta)"

        # test with non-diagonal matrix
        Qoffdiag = 0.1*torch.randn(4,5).exp()
        Q = dyn.build_Q_mat(Qoffdiag)
        eta = torch.zeros(5)

        max_growth = dyn.max_growthrate_diff(eta, Q)

        assert torch.isclose(max_growth, torch.tensor(0.0), atol=1e-6), "max growth rate should be equal to 0.0"

        # test shape with batched input
        eta = torch.randn(2, 3, 5)
        Qoffdiag = 0.1*torch.randn(2, 3, 4, 5).exp()
        Q = dyn.build_Q_mat(Qoffdiag)
        max_growth = dyn.max_growthrate_diff(eta, Q)

        assert max_growth.shape == (2, 3), "max growth rate should have shape (2, 3)"

    def test_dyn_model_batch_dims(self):
        """
        Test that the dynamic model can handle batch dimensions,
        both for inference and prediction.
        """

        def gen_params(batch_dim, d):
            Qoffdiag = torch.zeros((*batch_dim, d-1,d))
            Qoffdiag[...,0,0] = 0.1
            Qoffdiag[...,1,2] = 0.05

            rho = -0.1 * torch.ones((*batch_dim, d,))
            rho[...,0] = -0.2
            rho[...,2] = -0.05

            logX0 = torch.zeros((*batch_dim, d,))
            logX0[...,2] = -1.0

            return Qoffdiag, rho, logX0

        n = 10
        time = torch.arange(0,n,1)
        d = 3

        batch_dim = (5,2) ## in this case, we should add a dimension for time.
        Qoffdiag, rho, logX0 = gen_params(batch_dim, d)
        logFt, logYt = dyn.dynamic_model_diff_hom(time, rho, logX0, Qoffdiag)

        assert logFt.shape == (*batch_dim, n, d), "logFt has incorrect shape"
        assert logYt.shape == (*batch_dim, n), "logYt has incorrect shape"

        batch_dim = (5,1) ## in this case, we should use the second batch dimension for time.
        Qoffdiag, rho, logX0 = gen_params(batch_dim, d)
        logFt, logYt = dyn.dynamic_model_diff_hom(time, rho, logX0, Qoffdiag)

        assert logFt.shape == (batch_dim[0], n, d), "logFt has incorrect shape"
        assert logYt.shape == (batch_dim[0], n), "logYt has incorrect shape"

        batch_dim = (5,) ## in this case, we should again add a dimension for time.
        Qoffdiag, rho, logX0 = gen_params(batch_dim, d)
        logFt, logYt = dyn.dynamic_model_diff_hom(time, rho, logX0, Qoffdiag)

        assert logFt.shape == (*batch_dim, n, d), "logFt has incorrect shape"
        assert logYt.shape == (*batch_dim, n), "logYt has incorrect shape"


# TODO: add tests for other models, add tests for Predictive, and SVI!

