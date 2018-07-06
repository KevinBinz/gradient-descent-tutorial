import random
from lib.linear_regression import LinearRegression

class RandomSearch(LinearRegression):
    def __init__(self, data, num_iter, verbosity, param_range):
        super(RandomSearch, self).__init__(data, num_iter, verbosity)
        self.range = param_range

    def run(self):
        x, y, n, blah = self.get_initial_data()
        min_theta = [0.0, 0.0]
        min_loss = 1000000
        for i in range(self.num_iters):
            b = random.uniform(self.range[0], self.range[1])
            m = random.uniform(self.range[2], self.range[3])
            theta = [b, m]
            loss = self.compute_loss_directly(x, y, theta)
            if(loss < min_loss):
                min_loss = loss
                min_theta = theta

            self.save_result(i, min_theta, loss)

        loss = self.compute_loss_directly(x, y, min_theta)
        return min_theta, loss

