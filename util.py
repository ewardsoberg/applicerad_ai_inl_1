import csv
import numpy as np


def get_distance_matrix_for_algorithm():
    distances = []
    with open('csv_files/city_distances.csv', 'r') as cd:
        reader2 = csv.reader(cd)
        for row in reader2:
            distances.append(row[2])
    distances.pop(0)
    distances = np.array([float(x) for x in distances]).reshape(120, 120)
    return distances


def get_points_for_algorithms():
    points = []
    with open('csv_files/city_long_lat.csv', 'r') as c:
        reader1 = csv.reader(c)
        for row in reader1:
            points.append([row[1], row[2]])
    points.pop(0)
    points = [[float(x) for x in y] for y in points]
    return np.array(points)
