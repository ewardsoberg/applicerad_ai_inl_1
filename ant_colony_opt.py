import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sko.ACA import ACA_TSP
from util import get_points_for_algorithms, get_distance_matrix_for_algorithm
"""
Use city_long_lat.csv to create 120 points of long and lat.
Makes a distance matrix in shape (120, 120) out of the calculated distances between two cities.
Uses scikit learn Ant Colony Algorithm to do the Travelling salesperson problem.
Plot the result where the x and y points is real long and lat from cities.
Result of three different max_iter [100, 250, 500] is stored in result_plots.
"""

points_coordinate = get_points_for_algorithms()
distance_matrix = get_distance_matrix_for_algorithm()
num_points = len(points_coordinate)
print(num_points)


def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# %% Do ACA


aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=10, max_iter=50,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()

# %% Plot
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
#plt.savefig('ACO_pop50_iter500.png')
plt.show()
