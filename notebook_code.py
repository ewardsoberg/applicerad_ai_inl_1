import csv
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from sko.ACA import ACA_TSP
import pickle
import os
from scrape_and_save import turn_correct_data_to_correct_shape, scrape_correct_data_from_website, save_data_to_csv
import plotly.graph_objects as go


def save_pickle(file, filename: str):
    with open(filename, "wb") as output_file:
        pickle.dump(file, output_file)


def load_pickle(filename):
    try:
        with open(filename, "rb") as input_file:
            return pickle.load(input_file)
    except:
        print('no such file')


def get_csv():
    '''
    scrape data from:
    https://www.infoplease.com/world/geography/major-cities-latitude-longitude-and-corresponding-time-zones
    as: 'city_distances.csv' and returns data,
    or if 'city_distances.csv' exists:
    returns data
    :return: data from 'city_distances.csv'
    '''
    data = []
    filename2 = 'city_distances.csv'
    if not os.path.isfile(filename2):
        url = "https://www.infoplease.com/world/geography/major-cities-latitude-longitude-and-corresponding-time-zones"
        headers = ['start city', 'stop city', 'distance']
        save_data_to_csv(headers, turn_correct_data_to_correct_shape(scrape_correct_data_from_website(url), True))
    with open('city_distances.csv', mode='r') as distances:
        reader = csv.reader(distances)
        for row in reader:
            data.append(row)
    return data


def get_csv_data_for_random():
    '''
    If no saved .pkl, parse csv for Random Travel
    :return: parsed data
    '''
    filename = 'csv_data_for_random.pkl'
    if os.path.isfile(filename):
        result_data = load_pickle(filename)
    else:
        data = get_csv()

        df_all = pd.DataFrame(data[1:], columns=['Start', 'Target', 'Distance'])
        df = df_all[['Start', 'Target', 'Distance']]
        cities = list(dict.fromkeys(df['Start']))
        result_data = {}
        values = {}
        for city in cities:
            print(city)
            for _, row in df.iterrows():
                if row['Start'] == city:
                    values[row['Target']] = row['Distance']
            result_data[city] = values
            values = {}
        save_pickle(result_data, filename)
    return result_data


def get_csv_data_for_aco():
    '''
    if no saved .pkl, parse csv for Ant Colony Opt
    :return: parsed data
    '''
    filename = 'csv_data_for_ACO.pkl'
    if os.path.isfile(filename):
        result_data = load_pickle(filename)
    else:
        data = get_csv()

        dataset = []
        dataset_corpus = []
        df_all = pd.DataFrame(data[1:], columns=['Start', 'Target', 'Distance'])
        df = df_all[['Start', 'Target', 'Distance']]
        cities = list(dict.fromkeys(df['Start']))
        for city in cities:
            print(city)
            for _, row in df.iterrows():
                if row['Start'] == city:
                    dataset_corpus.append(np.float32(row['Distance']))
            dataset.append(dataset_corpus)
            dataset_corpus = []
        result_data = [dataset, cities]
        save_pickle(result_data, filename)
    return result_data


def travel(data1):
    '''
    Random route between cities and total distance
    :param data1: get_csv_data_for_random()
    :return: [total_distance, route]
    '''

    start = 'Aberdeen'
    route = [('Aberdeen', 0)]
    distance_traveled = 0
    available_destinations = [key for key, _ in data1.items()]
    available_destinations.remove(start)
    for steps in range(len(data1) - 1):
        next_destination = random.choice(available_destinations)
        distance = data1[start][next_destination]
        distance_traveled += float(distance)
        available_destinations.remove(next_destination)
        route.append((next_destination, distance))
        start = next_destination
    return [distance_traveled, route]


def run_random_travel(rounds=10000):
    '''
    Does x amount of random routes between coordinates where x is rounds
    :param rounds: int
    :return: Total distances sorted
    '''

    data = get_csv_data_for_random()
    results = []
    for i in range(rounds):
        results.append(travel(data))
    result_sorted = sorted(results)
    return result_sorted


def print_route(results):
    data = get_csv()

    df_all = pd.DataFrame(data[1:], columns=['Start', 'Target', 'Distance'])
    df = df_all[['Start', 'Target', 'Distance']]
    cities = list(dict.fromkeys(df['Start']))
    long_lat = get_long_lat()

    long_lat_mapping = {}
    for i in range(len(cities)):
        long_lat_mapping[cities[i]] = long_lat[i]

    coordinates = [long_lat_mapping[way[0]] for way in results[0][1]]
    coordinates = np.array(coordinates)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(coordinates[:, 0], coordinates[:, 1], 'o-r')
    plt.show()
    return coordinates


def get_long_lat():
    '''
    returns saved longitudes and latitudes for cities from .pkl or if first time,
    scrapes longitudes and latitudes for cities from:
    https://www.infoplease.com/world/geography/major-cities-latitude-longitude-and-corresponding-time-zones

    :return: long_lat_only
    '''

    filename = 'long_lat_only.pkl'
    if os.path.isfile(filename):
        long_lat_only = load_pickle(filename)
    else:
        url = "https://www.infoplease.com/world/geography/major-cities-latitude-longitude-and-corresponding-time-zones"
        long_lat_only = turn_correct_data_to_correct_shape(scrape_correct_data_from_website(url), False)
        save_pickle(long_lat_only, filename)
    return long_lat_only


def ant_colony(plot='map', size_pop=50, max_iter=100):
    '''
    Performs Ant Colony Optimization on TSP.

    :param plot: default: 'map', Optional: 'scatter'
    :param size_pop: int
    :param max_iter: int
    '''

    assert type(max_iter) == int
    assert type(size_pop) == int
    assert plot == 'map' or 'scatter'

    distance_matrix = np.array(get_csv_data_for_aco()[0])
    long_lat = get_long_lat()
    num_points = len(distance_matrix)
    points_coordinate = np.array(long_lat)

    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                  size_pop=size_pop, max_iter=max_iter,
                  distance_matrix=distance_matrix)

    best_x, best_y = aca.run()

    if plot == 'scatter':
        fig, ax = plt.subplots(1, 2)
        best_points_ = np.concatenate([best_x, [best_x[0]]])
        best_points_coordinate = points_coordinate[best_points_, :]
        ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
        res = aca.y_best_history
        res_df = pd.DataFrame(res).cummin()
        res_df.plot(ax=ax[1])
        plt.show()
        high_score_aco(res)

    elif plot == 'map':
        best_points_ = np.concatenate([best_x, [best_x[0]]])
        best_points_coordinate = points_coordinate[best_points_, :]
        plot_map_super(best_points_coordinate)
        res = aca.y_best_history
        high_score_aco(res)


def high_score_aco(results):
    '''
    prints and saves high score from Ant Colony Optimization
    :param results: res
    :return:
    '''
    filename = 'high_scores_ACO.pkl'
    res2 = sorted(results)
    low = res2[0]
    if os.path.isfile(filename):
        high_scores = load_pickle(filename)
    else:
        high_scores = []
    high_scores.append(low)
    save_pickle(high_scores, filename)
    print(f'Best distance: {low}')


def high_score_random(results):
    '''
    prints and saves high score from run_random_travel()
    :param results: results from run_random_travel()
    '''
    filename = 'high_scores_random.pkl'
    results = [x[0] for x in results]
    res2 = sorted(results)
    low = res2[0]
    if os.path.isfile(filename):
        high_scores = load_pickle(filename)
    else:
        high_scores = []
    high_scores.append(low)
    save_pickle(high_scores, filename)
    print(f'Best distance: {low}')


def print_high_scores():
    '''
    prints best scores from saved .pkl
    '''
    if os.path.isfile('high_scores_random.pkl'):
        rand = load_pickle('high_scores_random.pkl')
        print('Random')
        print(sorted(rand))
    else:
        print('No saved high scores for Random')

    if os.path.isfile('high_scores_ACO.pkl'):
        aco = load_pickle('high_scores_ACO.pkl')
        print('ACO')
        print(sorted(aco))
    else:
        print('No saved high scores for ACO')


def plot_map_super(results):
    '''
    Plots result on a map.
    :param results: coordinates
    '''
    city_data = []
    city_coord_data = get_long_lat()
    city_names = get_csv_data_for_aco()[1]
    for i, city in enumerate(city_names):
        city_data.append([
            city,
            city_coord_data[i][0],
            city_coord_data[i][1]])
    df_cities = pd.DataFrame(city_data, columns=['city', 'latitude', 'longitude'])

    paths = []
    for i in range(len(results)-1):
        paths.append([
            results[i][0],
            results[i][1],
            results[i+1][0],
            results[i+1][1]
        ])
    df_paths = pd.DataFrame(paths, columns=['start_lat', 'start_long', 'end_lat', 'end_long'])
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=df_cities['longitude'],
        lat=df_cities['latitude'],
        hoverinfo='text',
        text=df_cities['city'],
        mode='text',
        marker=dict(
            size=2,
            color='rgb(255, 0, 0)',
            line=dict(
                width=3,
                color='rgba(68, 68, 68, 0)'
            )
        )))

    for i in range(len(df_paths)):
        fig.add_trace(
            go.Scattergeo(
                lon=[df_paths['start_long'][i], df_paths['end_long'][i]],
                lat=[df_paths['start_lat'][i], df_paths['end_lat'][i]],
                mode='lines',
                line=dict(width=1, color='red'),
            )
        )

    fig.update_layout(
        title_text='TSP',
        showlegend=False,
        geo=dict(
            scope='world',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
        ),
    )

    fig.show()



result = run_random_travel(rounds=10000)
high_score_random(result)
coordinates_random = print_route(result)
plot_map_super(coordinates_random)


ant_colony()
print_high_scores()


