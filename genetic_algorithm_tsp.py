import datetime
from util import get_points_for_algorithms, get_distance_matrix_for_algorithm
import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA_TSP

"""
Use city_long_lat.csv to create 120 points of long and lat.
Makes a distance matrix in shape (120, 120) out of the calculated distances between two cities.
Uses scikit learn GeneticAlgorithm TSP to do the Travelling salesperson problem.
Plot the result where the x and y points is real long and lat from cities.
"""


points_coordinate = get_points_for_algorithms()
distance_matrix = get_distance_matrix_for_algorithm()
num_points = len(points_coordinate)


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# %% do GA

ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=0.001)


start_time = datetime.datetime.now()
best_points, best_distance = ga_tsp.run()
print('on {max_iter} iterations, {prob_mut} mutations, costs {time_costs}s'
      .format(max_iter=ga_tsp.max_iter, prob_mut=ga_tsp.prob_mut,
              time_costs=(datetime.datetime.now() - start_time).total_seconds()))


# %% plot
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()