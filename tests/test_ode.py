from scdynsys.vae import dyn
import torch


class TestODE:
    def test_ode_solver(self):
        rho = torch.tensor([-0.5, 0.2, -0.1])
        eta = torch.tensor([-0.1, -0.3, -0.5])
        u = torch.tensor(0.1)
        Qoffdiag = torch.tensor([[0.05, 0.08, 0.1], [0.02, 0.1, 0.03]])
        time = torch.arange(0.0, 50.0, 1)
        logX0 = torch.tensor([1.0, 2.0, 1.5]).log()

        logFt, logYt = dyn.dynamic_model_diff_ode(time, rho, eta, u, logX0, Qoffdiag)

        # solve same model with scipy

        from scipy.integrate import solve_ivp
        import numpy as np

        def vf_dynamic_model_diff_np(t, x, rho_minus_eta, u, Q_plus_eta):
            dxdt = Q_plus_eta @ x + rho_minus_eta * np.exp(-u*t) * x
            return dxdt

        def dynamic_model_diff_np(time, rho, eta, u, logX0, Q):
            X0 = np.exp(logX0)
            args = (rho - eta, u, Q + np.diag(eta))
            t_span = time[0], time[-1]
            sol = solve_ivp(vf_dynamic_model_diff_np, t_span, X0, t_eval=time, args=args, atol=1e-9, rtol=1e-6)
            Xt = sol.y.T
            logXt = np.log(Xt)
            logYt = np.log(np.sum(Xt, axis=1))
            logFt = logXt - np.expand_dims(logYt,-1)
            return logFt, logYt
        
        Q = dyn.build_Q_mat(Qoffdiag)
        logFt_np, logYt_np = dynamic_model_diff_np(time.numpy(), rho.numpy(), eta.numpy(), u.item(), logX0.numpy(), Q.numpy())


        assert torch.allclose(logFt, torch.tensor(logFt_np, dtype=torch.float32), atol=1e-5), "logFt does not match"
        assert torch.allclose(logYt, torch.tensor(logYt_np, dtype=torch.float32), atol=1e-5), "logYt does not match"





