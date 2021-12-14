import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from itertools import product
URL = "https://www.infoplease.com/world/geography/major-cities-latitude-longitude-and-corresponding-time-zones"
headers = ['start city', 'stop city', 'distance']


def scrape_correct_data_from_website(url: str):
    """Function to scrape a very specific data from a specific URL.
    Looking for the text in the div called 'td' with valign='top' and makes a list with lists out of the
     six columns for every city."""
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    what_i_need = [find.get_text() for find in soup.find_all({'td': 'top'})]
    raw_data = [what_i_need[j: j + 6] for j in range(0, len(what_i_need), 6)]
    return raw_data


def calc_distance(start: list, stop: list):
    """Calculate distance between a start and stop given long and lat."""
    R = 6373.0
    lat1 = radians(start[1])
    lon1 = radians(start[2])
    lat2 = radians(stop[1])
    lon2 = radians(stop[2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def turn_correct_data_to_correct_shape(data: list):
    """Function that sort out cities, long and lat from data.
    Depending on if the long or lat is 'S', 'N', 'W', 'E'
    it makes it to positive or negative float and removes the letter."""
    cities_long_lat = ([d[0].split(',')[0], (d[1] + '.' + d[2]), d[3] + '.' + d[4]] for d in data[:-1])
    new_cities = []
    for city in cities_long_lat:
        if 'S' in city[1]:
            city[1] = city[1].replace('S', '')
            city[1] = np.negative(float(city[1]))
        elif 'N' in city[1]:
            city[1] = city[1].replace('N', '')
            city[1] = np.positive(float(city[1]))
        if 'W' in city[2]:
            city[2] = city[2].replace('W', '')
            city[2] = np.negative(float(city[2]))
        elif 'E' in city[2]:
            city[2] = city[2].replace('E', '')
            city[2] = np.positive(float(city[2]))
        new_cities.append(city)
    cities_distance = [[data[0][0], data[1][0],
                        calc_distance(data[0],
                        data[1])]for data in product(new_cities, repeat=2)]
    return cities_distance


def save_data_to_csv(header: list, data: list):
    """Take headers, data and save it to a csv file called city_distance"""
    with open('city_distances.csv', 'w', newline="") as c:
        csv.writer(c).writerow(header)
        csv.writer(c).writerows(data)

def get_data_from_csv():
    data = []
    return data
# Example of a great pythonic function call, see below.

#save_data_to_csv(headers, turn_correct_data_to_correct_shape(scrape_correct_data_from_website(URL)))

