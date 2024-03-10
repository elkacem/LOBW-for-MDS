import networkx as nx
import random
import math
import numpy as np
import pandas as pd
from mat4py import loadmat
import glob
from functions import RandomPopulation, fitness, filtering, repair, getOrderSolution, local_search

def main():
    # load data from second dataset
    path = (r"datasets/data 2/problem11.mat")
    data = loadmat(path)
    evar = "E" + path[23:25]
    x = data[evar]
    A = np.array(x)
    G = nx.from_numpy_matrix(A)
    G.remove_edges_from(nx.selfloop_edges(G))

    # load data from first dataset
    # df = pd.read_table(path, delimiter=" ")
    # head = list(df.columns)
    # G = nx.from_pandas_edgelist(df - 1, head[0], head[1])

    # we put the runs
    for i in range(10):
        population = RandomPopulation(G)
        for whale_popu in population:
            repair(G, whale_popu)
            filtering(G, whale_popu)

        X = getOrderSolution(G, population)

        Xbest = X[0].copy()

        max_Iter = 200
        t = 1

        count_iter = 1
        while t <= max_Iter:

            a = 2 - t * (2 / max_Iter)

            for whale in X:
                i = whale.copy()
                index = X.index(whale)

                A = 2 * a * np.random.random() - a
                C = 2 * np.random.random()
                p = np.random.random()
                l = -1 + 2 * np.random.random()

                if p < 0.5:
                    if abs(A) < l:
                        for d in range(len(i)):
                            Dx = abs(C * Xbest[d] - i[d])
                            i[d] = Xbest[d] - A * Dx
                    elif abs(A) >= l:
                        for d in range(len(i)):
                            k = random.randint(0, len(X) - 1)
                            Dx = abs(C * X[k][d] - i[d])
                            i[d] = X[k][d] - A * Dx

                elif p >= 0.5:
                    for d in range(len(i)):
                        dist = abs(Xbest[d] - i[d])
                        i[d] = dist * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[d]

                for j in range(len(i)):
                    if i[j] > 1:
                        i[j] = 1
                    elif i[j] < 0:
                        i[j] = 0
                    elif 0 < i[j] < 1:
                        rf = random.uniform(0, 1)
                        if rf < 0.6:
                            i[j] = 1
                        else:
                            i[j] = 0

                repair(G, i)
                filtering(G, i)

                if fitness(G, i) > fitness(G, whale):
                    X[index] = i
                else:
                    rans = random.uniform(0, 1)
                    if rans < 0.5:
                        X[index] = i
                    else:
                        neighborL = []
                        for iteration in range(2):
                            neighbor = local_search(G, whale, population)
                            neighborL.append(neighbor)
                        neighborL = sorted(neighborL, key=lambda x: -fitness(G, x))
                        X[index] = neighborL[0]

            X = getOrderSolution(G, X)

            if fitness(G, X[0]) > fitness(G, Xbest):
                Xbest = X[0].copy()
                count_iter = t

            t = t + 1

if __name__ == '__main__':
    main()