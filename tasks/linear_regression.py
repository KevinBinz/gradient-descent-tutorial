import numpy as np

class LinearRegression():
    def __init__(self, data, num_iter, verbosity):
        self.data = data
        self.num_iters = num_iter
        self.print_mod = verbosity
        self.clear_data()

    def clear_data(self):
        self.loss_history = dict()
        self.theta_history = dict()

    def reset(self):
        self.clear_data()
        x, y, theta = self.get_initial_data()
        return x, y, theta

    def get_theta(self, iter):
        return self.theta_history[iter]

    def get_loss(self, iter):
        return self.loss_history[iter]

    def save_result(self, iter, theta, loss):
        self.loss_history[iter] = loss
        self.theta_history[iter] = theta
        if(self.print_mod <= 0) :
            return
        if(iter % self.print_mod == 0):
            self.print_status(iter, theta, loss)

    def print_status(self, num_iter, theta, loss):
        print("After {} iterations theta = {}, loss = {}".format(num_iter, theta.flatten(), loss))

    def compute_error(self, x, y, theta):
        hypothesis = np.dot(x, theta)
        return (hypothesis - y)

    def compute_loss(self, error):
        return np.sum(np.square(error)) / len(error)

    def compute_loss_directly(self, x, y, theta):
        return self.compute_loss(self.compute_error(x, y, theta))

    def get_initial_data(self):
        num_rows = self.data.shape[0]
        num_features = self.data.shape[1] - 1
        x = np.ones((num_rows, num_features+1))
        y = np.ones((num_rows,1))
        theta = np.zeros((num_features+1, 1))

        x[:, 1:num_features+1] = self.data[:, 0:num_features]
        y[:, 0] = self.data[:, -1]

        return x, y, theta

    # def get_loss_surface_from_grid(self, X, Y):
    #     Z = np.zeros(X.shape)
    #     x, y, n, theta = self.get_initial_data()
    #     for i in range(len(X)):
    #         for j in range(len(Y)):
    #             theta = np.array([X[i][j], Y[i][j]])
    #             Z[i][j] = self.compute_loss(self.compute_error(x, y, theta))
    #     return Z
    #
    # def get_loss_surface_from_vec(self, X, Y):
    #     Z = np.zeros(X.shape)
    #     x, y, n, theta = self.get_initial_data()
    #     for i in range(len(X)):
    #         theta = np.array([X[i], Y[i]])
    #         Z[i] = self.compute_loss(self.compute_error(x, y, theta))
    #     return Z
    # def get_historical_domain(self, curr_iter):
    #     if(curr_iter == 0):
    #         return None, None
    #
    #     b = np.zeros(shape=curr_iter-1)
    #     m = np.zeros(shape=curr_iter-1)
    #     for iter in range(0, curr_iter-1):
    #         b_now, m_now = self.get_theta(iter)
    #         b[iter] = b_now
    #         m[iter] = m_now
    #     return b, m