import numpy as np
import random


def RandomPopulation(G):
    Graflength = len(G.nodes())
    ReducedPopulation = []
    for i in range(100):
        solution = np.random.choice([0, 1], size=(Graflength,), p=[.5 / 4, 3.5 / 4])
        solution = solution.tolist()
        ReducedPopulation.append(solution)
    return ReducedPopulation

def fitness(G, sol):
    V = len(G.nodes())
    vc = []
    numberOfNodes = 0
    for index, value in enumerate(sol):
        if value == 1:
            numberOfNodes += 1
            vc.append(index)
            # if nodes start from 1 we add 1 to index
            vc.append(list(G.neighbors(index)))
    flat_vc = []
    for i in vc:
        if isinstance(i, list):
            flat_vc.extend(i)
        else:
            flat_vc.append(i)
    flat_vc = list(dict.fromkeys(flat_vc))
    if numberOfNodes > 0:
        f = (len(flat_vc) / V) + (1 / (V * numberOfNodes))
    if numberOfNodes == 0:
        numberOfNodes = len(G.nodes())
        f = (len(flat_vc) / V) + (1 / (V * numberOfNodes))

    return f

def filtering(G, sol):
    bestFitness = fitness(G, sol)
    # if bestFitness < 1:
    #     return None
    solvar = sol
    # best = sol
    for index, value in enumerate(sol):
        if value == 1:
            solvar[index] = 0
            fitOfSolvar = fitness(G, solvar)
            if fitOfSolvar > bestFitness:
                bestFitness = fitOfSolvar
                sol[index] = 0
            else:
                solvar[index] = 1
    return bestFitness, sol

def nonCovered(G, sol):
    neighborhoods = {v: {v} | set(G[v]) for v in G if sol[v] == 1}
    list_of_values = set()
    for key, val in neighborhoods.items():
        list_of_values.update(val)
    Noncovered = list(set(G.nodes) - set(list_of_values))
    return Noncovered

def sumW(G, x, Noncovered):
    nodesN = [node for node in list(G.neighbors(x)) if node in Noncovered]
    nodesW = sum(nodesN)
    return nodesW

def repair(G, sol):
    Noncovered = nonCovered(G, sol)

    ph = 0.7
    p = np.random.uniform(low=0.0, high=1.0)

    if p < ph:
        while (Noncovered):
            maxCandidate = max(Noncovered, key=lambda x: (sumW(G, x, Noncovered)))

            sol[maxCandidate] = 1
            Noncovered = nonCovered(G, sol)

    else:
        while (Noncovered):
            maxCandidate = random.choice(Noncovered)

            sol[maxCandidate] = 1
            Noncovered = nonCovered(G, sol)

    return sol

def getOrderSolution(G, solutions):

    popu = sorted(solutions, key=lambda x: -fitness(G, x))

    return popu

def complement(pop):
    # complement operation
    for i in range(len(pop)):
        pop[i] = 1 - pop[i]
    return pop


def intersection(population1, population2):
    # set intersection operation
    pop = [0] * len(population1)
    for i in range(len(population1)):
        if (population1[i] * population2[i] == 1):
            pop[i] = 1

    return pop


def sum_(population1, population2):
    # set sum_ operation
    pop = [0] * len(population1)
    for i in range(len(population1)):
        if (population1[i] == 1 or population2[i] == 1):
            pop[i] = 1

    return pop


def difference(population1, population2):
    # set difference operation
    pop = population1
    population2 = complement(population2)
    pop = intersection(pop, population2)
    return pop


def local_search(G, whale, population):
    rwhale = random.choices([a for a in population if a != whale], k=2)
    whale1 = rwhale[0].copy()
    whale2 = rwhale[1].copy()

    diff = difference(whale1, whale2)
    neighbor1 = sum_(whale, diff)
    neighbor2 = difference(whale, diff)

    repair(G, neighbor1)
    repair(G, neighbor2)
    filtering(G, neighbor1)
    filtering(G, neighbor2)

    # fitness0 = fitness(G, whale)
    fitness1 = fitness(G, neighbor1)
    fitness2 = fitness(G, neighbor2)

    if (fitness1 >= fitness2):
        # print('im here 1')
        return neighbor1
    if (fitness2 > fitness1):
        # print('im here 2')
        return neighbor2

