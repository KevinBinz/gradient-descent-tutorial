import time
import numpy as np

from optimizers.gradient_descent_2d import GradientDescent2D, GradientDescent2D_Vectorized
from optimizers.gradient_descent import GradientDescent, GradientDescent_Vectorized
from optimizers.manual_search import RandomSearch, GridSearch
from optimizers.least_squares import OrdinaryLeastSquares

def get_metadata(fn):
    learning_rates = {
        "data/more_columns.csv": 0.0005,
        "data/data.csv": 0.05,
        "data/more_rows.csv": 0.005
    }
    num_columns = {
        "data/more_columns.csv": 2,
        "data/data.csv": 1,
        "data/more_rows.csv": 1
    }
    return learning_rates[fn], num_columns[fn]

def display_solution(final_params, final_loss, start_time, print_elapsed_time):
    print("Solution: theta = {}, loss = {}".format(final_params.flatten(), final_loss))
    if print_elapsed_time:
        print("Elapsed Time: {0}".format(time.time() - start_time))

def deploy_optimizer(string, optimizer):
    print("=== {} ===".format(string))
    start_time = time.time()
    final_params, final_loss = optimizer.fit()
    display_solution(final_params, final_loss, start_time, print_elapsed_time=False)

if __name__ == "__main__":
    verbosity = 300    # N to display results from every N runs (-1 turns off printing)
    num_iter = 901

    fns = ["data/data.csv", "data/more_rows.csv", "data/more_columns.csv"]
    for fn in fns:
        learning_rate, num_columns = get_metadata(fn)
        data = np.array(np.genfromtxt(fn, delimiter=','))
        print()
        print(fn)

        deploy_optimizer("Ordinary Least Squares", OrdinaryLeastSquares(data, num_iter, verbosity))
        deploy_optimizer("Random Search", RandomSearch(data, num_iter, verbosity, param_range=[0.0, 1.0, 0.0, 1.0]))
        deploy_optimizer("Gradient Descent", GradientDescent(data, num_iter, learning_rate, verbosity))
        deploy_optimizer("Gradient Descent (Vectorized)", GradientDescent_Vectorized(data, num_iter, learning_rate, verbosity))
        if num_columns == 1:
            deploy_optimizer("2D Gradient Descent", GradientDescent(data, num_iter, learning_rate, verbosity))
            deploy_optimizer("2D Gradient Descent (Vectorized)", GradientDescent2D_Vectorized(data, num_iter, learning_rate, verbosity))




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
