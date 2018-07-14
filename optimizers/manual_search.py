import numpy as np
from tasks.linear_regression import LinearRegression

class RandomSearch():
    def __init__(self, data, num_iter, verbosity, param_range):
        self.task = LinearRegression(data, num_iter, verbosity)
        self.range = param_range

    def fit(self):
        x, y, _ = self.task.get_initial_data()
        m, n = x.shape[0], x.shape[1]
        min_theta = np.zeros((n,1))
        min_loss = 1000000
        for i in range(self.task.num_iters):
            theta = np.random.rand(n,1)
            loss = self.task.compute_loss_directly(x, y, theta)
            if(loss < min_loss):
                min_loss = loss
                min_theta = theta

            self.task.save_result(i, min_theta, loss)

        loss = self.task.compute_loss_directly(x, y, min_theta)
        return min_theta, loss


class GridSearch():
    def __init__(self, data, num_iter, verbosity, param_range):
        self.task = LinearRegression(data, num_iter, verbosity)
        self.range = param_range

    def fit(self):
        x, y, _ = self.task.get_initial_data()
        m, n = x.shape[0], x.shape[1]
        min_theta = np.zeros((n,1))
        min_loss = 1000000
        nx, ny = (3, 2)
        iter = 0
        for x in np.linspace(0, 1, nx):
            for y in np.linspace(0, 1, ny):
                iter = iter + 1
                loss = self.task.compute_loss_directly(x, y, theta)
                if(loss < min_loss):
                    min_loss = loss
                    min_theta = 2

            self.task.save_result(iter, min_theta, loss)

        loss = self.task.compute_loss_directly(x, y, min_theta)
        return min_theta, loss