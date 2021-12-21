from math import cos, sin, e, atan
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm


coordinates_list = [x for x in product(range(-5, 5), repeat=2)]
result_list_1 = []
result_list_2 = []


def kontinuerlig_optimering_21(coord: tuple):
    """Calculate f(x, y) = 1/1 + |x| + |y|(cos(x) + sin(y))2
    given two points in a tuple. Returns the points and result.
    """
    result = 1/(1+abs(coord[0])+abs(coord[1]))*(cos(coord[0])+sin(coord[1]))**2
    return coord, result


def kontinuerligt_optimering_31(coord: tuple):
    """Calculate f(x, y) = e-0.05(x2+y2)(arctan(x) − arctan(y) + e−(x2+y2)cos2(x)sin2(y))
     given two points in a tuple. Returns the points and result"""
    result = e ** ((coord[0] ** 2 + coord[1] ** 2) * -0.05) * (atan(coord[0]) - atan(coord[1]) + (e ** (-(3 ** 2 + 2 ** 2)) * cos(coord[0]) ** 2 * sin(coord[1]) ** 2))
    return coord, result


for coord in coordinates_list:
    result_list_1.append(kontinuerlig_optimering_21(coord))
    result_list_2.append(kontinuerligt_optimering_31(coord))


def plot_result(result: list):
    """3D plot for the result.
    Result: List of results
    """

    Xs = [x[0][0] for x in result]
    Ys = [y[0][1] for y in result]
    Zs = [z[1] for z in result]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(10))
    fig.tight_layout()
    plt.show()


def get_highest_result(results: list):
    """Sort result list and return the first element in that list, aka the highest result because reverse=True"""
    results.sort(key=lambda y: y[1], reverse=True)
    return results[0]


def get_lowest_result(results: list):
    """Sort result list and return the first element in that list, aka the lowest result"""
    results.sort(key=lambda y: y[1], reverse=False)
    return results[0]


plot_result(result_list_1)
plot_result(result_list_2)

