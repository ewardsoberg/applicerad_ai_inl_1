import csv
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import spatial
from sko.ACA import ACA_TSP
import pickle
import os


def save_pickle(file, filename: str):
    with open(filename, "wb") as output_file:
        pickle.dump(file, output_file)


def load_pickle(filename):
    try:
        with open(filename, "rb") as input_file:
            return pickle.load(input_file)
    except:
        print('no such file')


def get_csv_data(for_random=False):
    if for_random:
        filename = 'data_for_random.pkl'
    else:
        filename='data_for_ACO.pkl'
    if os.path.isfile(filename):
        data=load_pickle(filename)
    else:
        data = []
        with open('distances.csv', mode='r') as distances:
            reader = csv.reader(distances)
            for row in reader:
                data.append(row)

        df_all = pd.DataFrame(data[1:], columns=['Id', 'Start', 'Target', 'Distance'])
        df = df_all[['Start', 'Target', 'Distance']]

        cities = list(set(df['Start']))
        cities.sort()
        # print(cities)
        if for_random:
            data = {}
            values = {}
            for city in cities:
                print(city)
                for _, row in df.iterrows():
                    if row['Start'] == city:
                        values[row['Target']] = row['Distance']
                data[city] = values
                values = {}
            save_pickle(data, filename)
        else:
            dataset = []
            dataset_corpus = []
            for city in cities:
                print(city)
                for _, row in df.iterrows():
                    if row['Start'] == city:
                        dataset_corpus.append(float(row['Distance']))
                dataset.append(dataset_corpus)
                dataset_corpus = []
            data = [dataset, cities]
            save_pickle(data, filename)
    return data


def travel(data):
    start = 'Aberdeen'
    route = [('Aberdeen', 0)]
    distance_traveled = 0
    avalible_destinations = [key for key, _ in data.items()]
    avalible_destinations.remove(start)
    for steps in range(len(data) - 1):
        next_destination = random.choice(avalible_destinations)
        distance = data[start][next_destination]
        distance_traveled += int(distance)
        avalible_destinations.remove(next_destination)
        route.append((next_destination, distance))
        start = next_destination
    return [distance_traveled, route]


def run_random_travel(data, rounds: int=10000):
    result = []
    for i in range(rounds):
        result.append(travel(data))
    result_sorted = sorted(result)
    return result_sorted


def print_distance(result):
    for way in result[0][1]:
        print(f'{way[0]}, {way[1]}km')
    print('')
    print(f'Distance traveled: {result[0][0]}')


def plot_result(result):

    res=[x[0] for x in result]

    res1=res[:30]
    x = [x for x in range(len(res1))]
    res1=list(map(lambda x: x, res1))
    ax = plt.axes(projection='3d')

    plt.figure(figsize=(20,8))
    plt.bar(x, res1, width=0.2)
    plt.show()

    res2=sorted(res)
    high=res2[-1]
    low=res2[0]
    print(f'high: {high}km')
    print(f'low: {low}km')
    print(f'iterations: {len(res)}')


def ant_colony(dataset):
    num_points = 120
    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points

    #print(points_coordinate)

    #distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    distance_matrix = dataset

    #print(distance_matrix)


    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                  size_pop=50, max_iter=200,
                  distance_matrix=distance_matrix)

    best_x, best_y = aca.run()

    # %% Plot
    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    plt.show()


def highscore(result):
    filename = 'highscores.pkl'
    res = [x[0] for x in result]
    res2 = sorted(res)
    low = res2[0]
    if os.path.isfile(filename):
        highscores = load_pickle(filename)
    else:
        highscores = []

    highscores.append(low)
    print(sorted(highscores))
    save_pickle(highscores, filename)


random_travel = False
if random_travel:
    dataset_all = get_csv_data(for_random=random_travel)
    result = run_random_travel(dataset_all)
    print_distance(result)
    plot_result(result)
    highscore(result)
else:
    dataset_all = get_csv_data()
    dataset = dataset_all[0]
    ant_colony(dataset)
