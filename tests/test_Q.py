from scdynsys.vae import dyn
import torch


class TestQ:
    def test_build_Q(self):
        n = 7 # dim
        b = (3,1) # batch dims
        i = (1,0) # test batch
        Qoffdiag = torch.randn(*b, n-1, n)
        Q = dyn.build_Q_mat(Qoffdiag)
        
        # test that the cols sum to zero
        assert all(torch.isclose(Q[i].sum(axis=0), torch.zeros(n), atol=1e-5)), "Q is not a stochastic matrix"
        
        # test that triu and tril are identical
        
        TU1 = Qoffdiag[i].triu(diagonal=1)
        TU2 = Q[i].triu(diagonal=1)[:-1,:]

        assert all(torch.isclose(TU1, TU2, atol=1e-5).flatten()), "upper triangular mismatch"


        TL1 = Qoffdiag[i].tril(diagonal=0)
        TL2 = Q[i].tril(diagonal=-1)[1:,:]

        assert all(torch.isclose(TL1, TL2, atol=1e-5).flatten()), "lower triangular mismatch"


    def test_remove_diagonal(self):
        n = 7
        b = (3,1)
        Qoffdiag = torch.randn(*b, n-1, n).exp()
        Q = dyn.build_Q_mat(Qoffdiag)

        Qoffdiag2 = dyn.remove_diagonal(Q)

        assert Qoffdiag.shape == Qoffdiag2.shape, "shape mismatch"

        assert all(torch.isclose(Qoffdiag, Qoffdiag2, atol=1e-5).flatten()), "diagonal removal failed"

