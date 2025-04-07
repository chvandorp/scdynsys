from scdynsys import identifiability as ident
import numpy as np

class TestIdentifiability:
    def test_perturb_Q(self):
        d = 5
        # construct a random Q matrix
        Q = np.exp(np.random.randn(d,d))
        idx = np.diag_indices(d)
        Q[idx] = 0.0
        Q[idx] = -np.sum(Q, axis=0)

        # perturb an element
        i, j = 1, 3
        num = 11
        pQs = ident.perturb_Q(Q, 0.5, num, i,j)

        # test that the cols sum to 0
        for pQ in pQs:        
            assert np.all(np.isclose(np.sum(pQ, axis=0), np.zeros(d))), "perturbed Q matrix is not stochastic"
