import numpy as np
from lib.linear_regression import LinearRegression


class GradientDescent2D():
    def __init__(self, data, num_iter, learning_rate, verbosity):
        self.task = LinearRegression(data, num_iter, verbosity)
        self.learning_rate = learning_rate

    def fit(self):
        x, y, m, theta = self.task.reset()
        for iter in range(0, self.task.num_iters):
            loss, gradient = 0, [0, 0]
            for i in range(0, m):
                prediction = theta[1] * x[i][1] + theta[0] * x[i][0]
                error = y[i] - prediction
                loss = loss - (1/m) * error ** 2
                gradient[0] = gradient[0] - (2/m) * x[i][0] * error
                gradient[1] = gradient[1] - (2/m) * x[i][1] * error
            theta[0] = theta[0] - self.learning_rate * gradient[0]
            theta[1] = theta[1] - self.learning_rate * gradient[1]
            self.task.save_result(iter, theta, loss)
        return theta, self.task.compute_loss_directly(x, y, theta)

class GradientDescent2D_PartiallyVectorized():
    def __init__(self, data, num_iter, learning_rate, verbosity):
        self.task = LinearRegression(data, num_iter, verbosity)
        self.learning_rate = learning_rate

    def fit(self):
        x, y, m, theta = self.task.reset()

        for iter in range(0, self.task.num_iters):
            loss, gradient = 0, 0
            for i in range(0, m):
                prediction = np.dot(x[i,:], theta)
                error = y[i] - prediction
                loss = loss + (1/m) * error ** 2
                gradient = gradient - (2/m) * x[i,1] * error
            theta = theta - self.learning_rate * gradient
            self.task.save_result(iter, theta, loss)
        return theta, self.task.compute_loss_directly(x, y, theta)

class GradientDescent2D_Vectorized():
    def __init__(self, data, num_iter, learning_rate, verbosity):
        self.task = LinearRegression(data, num_iter, verbosity)
        self.learning_rate = learning_rate

    def fit(self):
        x, y, m, theta = self.task.reset()

        for iter in range(0, self.task.num_iters):
            prediction = np.dot(x, theta)                                   # (m,2)*(2,1) -> (m,1)
            error = prediction - y                                          # (m,1)-(m,1) -> (m,1)
            loss = np.sum(np.square(error)) / len(error)                    # (m,1) -> (1,1)
            gradient = (2/m) * np.dot(x.T, error)                           # (2,m)*(m,1) -> (2,1)
            theta = theta - (self.learning_rate * gradient)                 # (2,1) -> (2,1)
            self.task.save_result(iter, theta, loss)
        return theta, self.task.compute_loss_directly(x, y, theta)


