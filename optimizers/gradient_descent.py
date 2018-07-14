import numpy as np
from tasks.linear_regression import LinearRegression


class GradientDescent():
    def __init__(self, data, num_iter, learning_rate, verbosity):
        self.task = LinearRegression(data, num_iter, verbosity)
        self.learning_rate = learning_rate

    def fit(self):
        x, y, theta = self.task.reset()
        m, n = x.shape[0], x.shape[1]
        loss = 0
        for iter in range(0, self.task.num_iters):
            loss, gradient = 0, np.zeros((n, 1))
            for i in range(0, m):
                prediction = 0
                for j in range(0, n):
                    prediction = prediction + theta[j] * x[i][j]
                error = y[i] - prediction
                loss = loss + (1/m) * error ** 2
                for j in range(0, n):
                    gradient[j] = gradient[j] - (2/m) * x[i][j] * error
            for k in range(0, n):
                theta[k] = theta[k] - self.learning_rate * gradient[k]
            self.task.save_result(iter, theta, loss)
        return theta, loss

class GradientDescent_Vectorized():
    def __init__(self, data, num_iter, learning_rate, verbosity):
        self.task = LinearRegression(data, num_iter, verbosity)
        self.learning_rate = learning_rate

    def fit(self):
        x, y, theta = self.task.reset()
        m, n = x.shape[0], x.shape[1]
        loss = 0

        for iter in range(0, self.task.num_iters):
            prediction = np.dot(x, theta)                                   # (m,2)*(2,1) -> (m,1)
            error = prediction - y                                          # (m,1)-(m,1) -> (m,1)
            loss = np.sum(np.square(error)) / len(error)                    # (m,1) -> (1,1)
            gradient = (2/m) * np.dot(x.T, error)                           # (2,m)*(m,1) -> (2,1)
            theta = theta - (self.learning_rate * gradient)                 # (2,1) -> (2,1)
            self.task.save_result(iter, theta, loss)
        return theta, loss


