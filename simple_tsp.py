from sys import maxsize
import numpy as np
v = 3


def travelling_salesman_function(graph, s):
    vertex = []
    for i in range(v):
        if i != s:
            vertex.append(i)

    min_path = maxsize
    while True:
        current_cost = 0
        k = s
        for i in range(len(vertex)):
            current_cost += graph[k][vertex[i]]
            k = vertex[i]
        current_cost += graph[k][s]
        min_path = min(min_path, current_cost)

        if not next_perm(vertex):
            break
    return min_path


def next_perm(l):
    n = len(l)
    i = n - 2

    while i >= 0 and l[i] > l[i + 1]:
        i -= 1

    if i == -1:
        return False

    j = i + 1
    while j < n and l[j] > l[i]:
        j += 1

    j -= 1

    l[i], l[j] = l[j], l[i]
    left = i + 1
    right = n - 1

    while left < right:
        l[left], l[right] = l[right], l[left]
        left += 1
        right -= 1
    return True


dest = [['a', 'a', 0], ['a', 'b', 1], ['a', 'c', 3], ['b', 'a', 1], ['b', 'b', 0], ['b', 'c', 7],
        ['c', 'a', 3], ['c', 'b', 7], ['c', 'c', 0]]

graph = np.array([des[2] for des in dest]).reshape(3, 3)

print(graph)
s = 0
res = travelling_salesman_function(graph, s)
print(res)


