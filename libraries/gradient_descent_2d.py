import numpy as np
from libraries.ols import OLS


class GradientDescent2D(OLS):
    def __init__(self, data, num_iter, learning_rate, verbosity):
        super(GradientDescent2D, self).__init__(data, num_iter, verbosity)
        self.learning_rate = learning_rate

    def gradient_descent_v1(self):
        x, y, m, theta = self.reset()
        for iter in range(0, self.num_iters):
            loss, gradient = 0, [0, 0]
            for i in range(0, m):
                prediction = theta[1] * x[i][1] + theta[0] * x[i][0]
                error = y[i] - prediction
                loss = loss - (1/m) * error ** 2
                gradient[0] = gradient[0] - (2/m) * x[i][0] * error
                gradient[1] = gradient[1] - (2/m) * x[i][1] * error
            theta[0] = theta[0] - self.learning_rate * gradient[0]
            theta[1] = theta[1] - self.learning_rate * gradient[1]
            self.save_result(iter, theta, loss)

    def gradient_descent_v2(self):
        x, y, m, theta = self.reset()

        for iter in range(0, self.num_iters):
            loss, gradient = 0, np.zeros(shape=2)
            for i in range(0, m):
                prediction = np.dot(x[i,:], theta)
                error = y[i] - prediction
                loss = loss + (1/m) * error ** 2
                gradient = gradient - (2/m) * np.dot(x[i,:], error)
            theta = theta - self.learning_rate * gradient
            self.save_result(iter, theta, loss)

    def gradient_descent_v3(self):
        x, y, m, theta = self.reset()

        for iter in range(0, self.num_iters):
            prediction = np.dot(x, theta)                                   # (m,2)*(2,1) -> (m,1)
            error = prediction - y                                          # (m,1)-(m,1) -> (m,1)
            loss = np.sum(np.square(error)) / len(error)                    # (m,1) -> (1,1)
            gradient = (2/m) * np.dot(x.T, error)                           # (2,m)*(m,1) -> (2,1)
            theta = theta - (self.learning_rate * gradient)                 # (2,1) -> (2,1)

            self.save_result(iter, theta, loss)