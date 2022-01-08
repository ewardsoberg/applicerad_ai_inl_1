import datetime
from util import get_points_for_algorithms, get_distance_matrix_for_algorithm
import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA_TSP
import multiprocessing as mp
import time
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
    '''
    The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

def ga_tsp_single():
    for max_iter in (100, 500, 1000):
        for prob_mut in (0.001, 0.01, 0.05):
            ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=0.001)

            start_time = datetime.datetime.now()
            best_points, best_distance = ga_tsp.run()

            print('on {max_iter} iterations, {prob_mut} mutations, costs {time_costs}s'
                  .format(max_iter=ga_tsp.max_iter, prob_mut=ga_tsp.prob_mut,
                          time_costs=(datetime.datetime.now() - start_time).total_seconds()))

def ga_tsp_multi():
    for max_iter in (100, 500, 1000):
        for prob_mut in (0.001, 0.01, 0.05):
            ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=max_iter, prob_mut=prob_mut)

            start_time = datetime.datetime.now()
            best_points, best_distance = ga_tsp.run()

            print('multiprocessing on {max_iter} iterations, {prob_mut} mutations, costs {time_costs}s'
                  .format(max_iter=ga_tsp.max_iter, prob_mut=ga_tsp.prob_mut,
                          time_costs=(datetime.datetime.now() - start_time).total_seconds()))

def main():

    n_cores = mp.cpu_count()
    print(f'Number of cores: {n_cores}')

    start_time = datetime.datetime.now()

    processes = [mp.Process(target=ga_tsp_multi) for i in range(n_cores)]

    # Starta varje process
    [p.start() for p in processes]
    # Sammanställ processer
    [p.join() for p in processes]
    # Plocka ut resultat

    print(f' multi time {datetime.datetime.now()-start_time}')



if __name__ == "__main__":
    
    mp.freeze_support()
    main()
