import time
import numpy as np

from libraries.gradient_descent_2d import GradientDescent2D
from libraries.random_search import RandomSearch

def display_correct_result():
   print("\nCorrect Result: b = {0}, m = {1}, error = {2}".format((2/3), (1/2), 0.055555555555))

if __name__ == "__main__":
    verbosity = 50  # -1 turns off printing
                    # N to display results from every N runs.
    num_iter = 601
    learning_rate = 0.05

    data = np.array(np.genfromtxt("data/data.csv", delimiter=','))
    display_correct_result()

    rs = RandomSearch(data, num_iter, verbosity, param_range=[0.0, 1.0, 0.0, 1.0])
    gd = GradientDescent2D(data, num_iter, learning_rate, verbosity)

    print("=== Random Search ===")
    start_time = time.time()
    rs.run()
    print("Elapsed Time: {0}".format(time.time() - start_time))

    print("=== Vanilla Gradient Descent ===")
    start_time = time.time()
    gd.gradient_descent_v1()
    print("Elapsed Time: {0}".format(time.time() - start_time))

    # print("=== Partially Vectorized Gradient Descent ===")
    # start_time = time.time()
    # gd.gradient_descent_v2()
    # print("Elapsed Time: {0}".format(time.time() - start_time))

    print("=== Vectorized Gradient Descent ===")
    start_time = time.time()
    gd.gradient_descent_v3()
    print("Elapsed Time: {0}".format(time.time() - start_time))


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
