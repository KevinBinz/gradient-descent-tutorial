import numpy as np
from numpy import linalg as la
from numpy import testing
from lib.linear_regression import LinearRegression

class OrdinaryLeastSquares():
    def __init__(self, data, num_iter, verbosity):
        self.task = LinearRegression(data, num_iter, verbosity)

    def fit(self):
        x, y, m, theta = self.task.reset()
        sym_inv = la.inv(np.dot(x.T, x))
        pseudo_inverse = np.dot(sym_inv, x.T)
        pseudo_inverse_canonical = la.pinv(x)
        #testing.assert_allclose(pseudo_inverse, pseudo_inverse_canonical, rtol=1e-5)
        theta = np.dot(pseudo_inverse, y)
        return theta, self.task.compute_loss_directly(x, y, theta)