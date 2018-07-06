import time
import numpy as np

from lib.gradient_descent_2d import GradientDescent2D, GradientDescent2D_Vectorized
from lib.manual_search import RandomSearch, GridSearch
from lib.least_squares import OrdinaryLeastSquares

def display_correct_result():
   print("\n=== Correct Parameters: b = {}, m = {}, loss = {} ===".format((2/3), (1/2), 0.055555555555))

def display_solution(final_params, final_loss, start_time, print_elapsed_time):
    print("Solution: b = {}, m = {}, loss = {}".format(final_params[0], final_params[1], final_loss))
    if print_elapsed_time:
        print("Elapsed Time: {0}".format(time.time() - start_time))

if __name__ == "__main__":
    verbosity = 300  # -1 turns off printing
                     # N to display results from every N runs.
    num_iter = 601
    learning_rate = 0.05

    data = np.array(np.genfromtxt("data/data.csv", delimiter=','))
    display_correct_result()

    print("=== Ordinary Least Squares ===")
    ols = OrdinaryLeastSquares(data, num_iter, verbosity)
    start_time = time.time()
    final_params, final_loss = ols.fit()
    display_solution(final_params, final_loss, start_time, print_elapsed_time=False)

    print("=== Random Search ===")
    rs = RandomSearch(data, num_iter, verbosity, param_range=[0.0, 1.0, 0.0, 1.0])
    start_time = time.time()
    final_params, final_loss = rs.fit()
    display_solution(final_params, final_loss, start_time, print_elapsed_time=False)

    print("=== Grid Search ===")
    rs = GridSearch(data, num_iter, verbosity, param_range=[0.0, 1.0, 0.0, 1.0])
    start_time = time.time()
    final_params, final_loss = rs.fit()
    display_solution(final_params, final_loss, start_time, print_elapsed_time=False)

    print("=== Vanilla Gradient Descent ===")
    gd = GradientDescent2D(data, num_iter, learning_rate, verbosity)
    start_time = time.time()
    final_params = gd.fit()
    display_solution(final_params, final_loss, start_time, print_elapsed_time=False)

    print("=== Vectorized Gradient Descent ===")
    gd_vec = GradientDescent2D_Vectorized(data, num_iter, learning_rate, verbosity)
    start_time = time.time()
    final_params, final_loss = gd_vec.fit()
    display_solution(final_params, final_loss, start_time, print_elapsed_time=False)


    # save_mod = -1  # Turns off file creation. Set to e.g., 100 to save one file per 100 runs.
    # save_path = "rs_scatter_gif/"''
    # z_angle = 0
    # ov = MatplotLibVisualizer(random_iterations, param_range, data_range=[0, 4, 0, 3],
    #   grid_space_between_points=0.005, z_angle)
    # #ov.create_scattergif(rs, save_mod, save_path)
    #
    # save_path = "surface_compare_gif/"
    # ov.compare_scatter_to_surface(rs, save_path)
    #
    # save_path = "gd_scatter_gif/"
    # ov = MatplotLibVisualizer(gd_iterations, param_range, data_range, grid_space_between_points)
    # ov.create_scattergif(gdv, save_mod, save_path)
