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
from scrape_and_save import turn_correct_data_to_correct_shape, scrape_correct_data_from_website, save_data_to_csv


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
    data = []
    filename2 = 'city_distances.csv'
    if not os.path.isfile(filename2):
        URL = "https://www.infoplease.com/world/geography/major-cities-latitude-longitude-and-corresponding-time-zones"
        headers = ['start city', 'stop city', 'distance']
        save_data_to_csv(headers, turn_correct_data_to_correct_shape(scrape_correct_data_from_website(URL), True))
    with open('city_distances.csv', mode='r') as distances:
        reader = csv.reader(distances)
        for row in reader:
            data.append(row)
    return data

def get_csv_data_for_random():
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
        save_pickle(data, filename)
    return result_data


def get_csv_data_for_ACO():
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


'''def get_csv_data(for_random=False):
    if for_random:
        filename = 'data_for_random.pkl'
    else:
        filename='long_lat_calculated.pkl'

    if os.path.isfile(filename):
        data = load_pickle(filename)
    else:
        data = []
        with open('city_distances.csv', mode='r') as distances:
            reader = csv.reader(distances)
            for row in reader:
                data.append(row)

        df_all = pd.DataFrame(data[1:], columns=['Start', 'Target', 'Distance'])
        df = df_all[['Start', 'Target', 'Distance']]

        #cities = list(df['Start'])
        cities = list(dict.fromkeys(df['Start']))
        #df.sort_values(by='Start')
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
                        #if row['Distance']=='0.0':
                            #print(row['Start'], _/120)
                        dataset_corpus.append(np.float32(row['Distance']))
                dataset.append(dataset_corpus)
                dataset_corpus = []
            data = [dataset, cities]
            save_pickle(data, filename)
    return data
'''

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

def travel_with_index(data):
    start = 'Aberdeen'
    route = [0]
    distance_traveled = 0
    avalible_destinations = [key for key in range(119)]
    for steps in range(len(data) - 1):
        next_destination = random.choice(avalible_destinations)
        distance = data[start].iloc(next_destination)
        distance_traveled += int(distance)
        avalible_destinations.remove(next_destination)
        route.append((next_destination, distance))
        start = next_destination
    return [distance_traveled, route]

def run_random_travel(rounds: int=10000):
    data = get_csv_data_for_random()
    result = []
    for i in range(rounds):
        result.append(travel(data))
    result_sorted = sorted(result)
    return result_sorted


def print_route(result):
    data = get_csv()

    df_all = pd.DataFrame(data[1:], columns=['Start', 'Target', 'Distance'])
    df = df_all[['Start', 'Target', 'Distance']]
    cities = list(dict.fromkeys(df['Start']))
    long_lat = get_long_lat()

    facit_dict={}
    for i in range(len(cities)):
        facit_dict[cities[i]] = long_lat[i]

    '''for way in result[0][1]:
        print(f'{way[0]}, {way[1]}km')
    print('')
    print(f'Distance traveled: {result[0][0]}')
    '''

    coordinates = [facit_dict[way[0]] for way in result[0][1]]
    coordinates = np.array(coordinates)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(coordinates[:, 0], coordinates[:, 1], 'o-r')
    plt.show()
    return coordinates

def plot_result(result, verb=False):

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
    if verb:
        print(f'high: {high}km')
        print(f'low: {low}km')
        print(f'iterations: {len(res)}')


def get_long_lat():
    filename = 'long_lat_only.pkl'
    if os.path.isfile(filename):
        data = load_pickle(filename)
    else:
        URL = "https://www.infoplease.com/world/geography/major-cities-latitude-longitude-and-corresponding-time-zones"
        data = turn_correct_data_to_correct_shape(scrape_correct_data_from_website(URL), False)
        save_pickle(data, filename)
    return data


def ant_colony():
    dataset_all = get_csv_data_for_ACO()
    dataset = np.array(dataset_all[0])
    long_lat = get_long_lat()
    num_points = len(dataset)
    points_coordinate = np.array(long_lat)

    distance_matrix = np.array(dataset)

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
    res=aca.y_best_history
    res_df=pd.DataFrame(res).cummin()
    res_df.plot(ax=ax[1])
    plt.show()
    highscore_ACO(res)

def highscore_ACO(result):
    filename = 'highscores_ACO.pkl'
    res2 = sorted(result)
    low = res2[0]
    if os.path.isfile(filename):
        highscores = load_pickle(filename)
    else:
        highscores = []
    highscores.append(low)
    save_pickle(highscores, filename)

def highscore_random(result):
    filename = 'highscores_random.pkl'
    res2 = sorted(result)
    low = res2[0]
    if os.path.isfile(filename):
        highscores = load_pickle(filename)
    else:
        highscores = []
    highscores.append(low)
    save_pickle(highscores, filename)


def print_highscores():
    rand = load_pickle('highscores_random.pkl')
    aco = load_pickle('highscores_ACO.pkl')
    print('Random')
    print(sorted(rand))
    print('ACO')
    print(sorted(aco))




result = run_random_travel(rounds=100000)
coordinates_random = print_route(result)

#plot_result(result)
result_for_highscore = [x[0] for x in result]
highscore_random(result_for_highscore)
ant_colony()
print_highscores()

