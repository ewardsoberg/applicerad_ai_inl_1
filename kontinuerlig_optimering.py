from math import cos, sin
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm


coordinates_list = [x for x in product(range(-5, 5), repeat=2)]
result_list = []


def kontinuerlig_optimering_21(coord: tuple):
    """Calculate f(x, y) = 1/1 + |x| + |y|(cos(x) + sin(y))2
    given two points in a tuple. Returns the points and result.
    """
    result = 1/(1+abs(coord[0])+abs(coord[1]))*(cos(coord[0])+sin(coord[1]))**2
    return coord, result


for coord in coordinates_list:
    result_list.append(kontinuerlig_optimering_21(coord))


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
    """Sort result list and return the first element in that list, aka the highest result"""
    results.sort(key=lambda y: y[1], reverse=True)
    return results[0]


plot_result(result_list)
print(f'Highest result is: {get_highest_result(result_list)}')
