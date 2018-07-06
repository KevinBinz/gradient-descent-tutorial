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
        x, y, m, theta = self.get_initial_data()
        return x, y, m, theta

    def get_theta(self, iter):
        b = self.theta_history[iter][0]
        m = self.theta_history[iter][1]
        return b, m

    def get_loss(self, iter):
        return self.loss_history[iter]

    def get_historical_domain(self, curr_iter):
        if(curr_iter == 0):
            return None, None

        b = np.zeros(shape=curr_iter-1)
        m = np.zeros(shape=curr_iter-1)
        for iter in range(0, curr_iter-1):
            b_now, m_now = self.get_theta(iter)
            b[iter] = b_now
            m[iter] = m_now
        return b, m

    def save_result(self, iter, theta, loss):
        self.loss_history[iter] = loss
        self.theta_history[iter] = theta
        self.print_result(iter, theta, loss)

    def print_result(self, iter, theta, loss):
        if(self.print_mod <= 0):
            return
        if(iter % self.print_mod == 0):
            print("After {0} iterations b = {1}, m = {2}, loss = {3}".format(iter, theta[0], theta[1], loss))

    def compute_error(self, x, y, theta):
        hypothesis = np.dot(x, theta)
        return (hypothesis - y)

    def compute_loss_directly(self, x, y, theta):
        return self.compute_loss(self.compute_error(x, y, theta))

    def compute_loss(self, error):
        return np.sum(np.square(error)) / len(error)

    def get_initial_data(self):
        m = len(self.data)
        x = np.zeros(shape=(m, 2))
        y = np.zeros(shape=(m,1))
        theta = np.zeros(shape=(2,1))

        for i in range(0, m):
            x[i][0] = 1
            x[i][1] = self.data[i, 0]
            y[i] = self.data[i, 1]
        return x, y, m, theta

    def get_loss_surface_from_grid(self, X, Y):
        Z = np.zeros(X.shape)
        x, y, n, theta = self.get_initial_data()
        for i in range(len(X)):
            for j in range(len(Y)):
                theta = np.array([X[i][j], Y[i][j]])
                Z[i][j] = self.compute_loss(self.compute_error(x, y, theta))
        return Z

    def get_loss_surface_from_vec(self, X, Y):
        Z = np.zeros(X.shape)
        x, y, n, theta = self.get_initial_data()
        for i in range(len(X)):
            theta = np.array([X[i], Y[i]])
            Z[i] = self.compute_loss(self.compute_error(x, y, theta))
        return Z