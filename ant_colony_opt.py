import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sko.ACA import ACA_TSP

points = []
distances = []
with open('csv_files/city_long_lat.csv', 'r') as c:
    reader1 = csv.reader(c)
    for row in reader1:
        points.append([row[1], row[2]])
with open('csv_files/city_distances.csv', 'r') as cd:
    reader2 = csv.reader(cd)
    for row in reader2:
        distances.append(row[2])
distances.pop(0)
points.pop(0)
distances = np.array([float(x) for x in distances]).reshape(120, 120)
points = [[float(x) for x in y] for y in points]
num_points = len(points)
points_coordinate = np.array(points)
distance_matrix = distances


def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# %% Do ACA


aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=50, max_iter=500,
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
