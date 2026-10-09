import unittest
import numpy as np
from task2_regression.regression import fit_sparse, predict_sparse


class RegressionTests(unittest.TestCase):
    def test_recovers_nonlinear_signal_out_of_sample(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(1000,53))
        y = x[:,6]**2+x[:,7]
        model = fit_sparse(x[:800],y[:800],2)
        self.assertLess(np.max(np.abs(predict_sparse(model,x[800:])-y[800:])),1e-10)

if __name__ == "__main__":
    unittest.main()
