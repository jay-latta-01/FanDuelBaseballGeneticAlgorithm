'''
This file contains a function that can produce several fantasy baseball lineups
optimized for average points using a novel genetic algorithm enhanced with steepest
ascent. The function takes an input file path, population size, mutation rate,
and exit condition for number of iterations without improvement as parameters.
The function outputs a dataframe with selected lineups, an array of their
fitness values, and execution time of the algorithm.
'''

import itertools as it
import math
import time
import pandas as pd
import numpy as np
import random

def run(popSize, bitwiseMut, nImp, input_file, output_file=None):
    # Import and process data
    players = pd.read_csv(input_file)
    names = np.array(players['name'].tolist())
    # Assign OR parameters
    positions = players['position'].tolist()
    cost = players['cost']
    value = players['points']
    lineupClass = ['P', 'C/1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'Util']
    nLineup = len(lineupClass)
    budget = 35000
    # Assign players to their respective position categories
    posCat = {}
    posCat['P'] = []
    posCat['C/1B'] = []
    posCat['2B'] = []
    posCat['3B'] = []
    posCat['SS'] = []
    posCat['OF'] = []
    posCat['Util'] = []
    for i in range(len(players)):
        if 'P' in positions[i]:
            posCat['P'].append(i)
        else:
            posCat['Util'].append(i)
            if ('C' in positions[i]) or ('1B' in positions[i]):
                posCat['C/1B'].append(i)
            if '2B' in positions[i]:
                posCat['2B'].append(i)
            if '3B' in positions[i]:
                posCat['3B'].append(i)
            if 'SS' in positions[i]:
                posCat['SS'].append(i)
            if 'OF' in positions[i]:
                posCat['OF'].append(i)

    # Ensure population does not contain any duplicate lineups, regardless of position
    def keepSetUnique(population):
        populationSet = population[:, :nLineup].copy()
        for i in range(populationSet.shape[0]):
            sol = populationSet[i, :]
            populationSet[i, :] = np.sort(sol)
        populationSet, keepInd = np.unique(populationSet, axis=0, return_index=True)
        population = population[keepInd, :]
        return population

    # Calculate the "unfitness" of a solution, meaning how over budget it is
    def calcUnfit(lineup):
        totalCost = sum(cost[player] for player in lineup)
        unfitness = max(0, totalCost - budget)
        return unfitness

    # Calculate objective function (fitness) value of each solution
    def calcObj(lineup):
        unfitnessFactor = 1000  # factor by which each unit of unfitness is penalized
        totalValue = sum(value[player] for player in lineup)
        totalValue = totalValue - calcUnfit(lineup) * unfitnessFactor
        return totalValue

    # Perform uniform crossover operation on 2 parent solutions to produce 2 children
    def uniCross(p1, p2):
        # In case of possible duplication
        if random.random() > 0.5:
            temp = p1
            p2 = p1
            p1 = temp
        childArr = np.zeros((2, nLineup + 1))
        crossInd1 = np.random.randint(0, 2, nLineup)
        # Check for possibility of within-solution duplication
        totalP = np.concatenate((p1, p2))
        uniqueP, uniqueCounts = np.unique(totalP, return_counts=True)
        duplicatedP = uniqueP[uniqueCounts > 1]
        # Ensures no within-solution duplication can occur
        if len(uniqueP) < len(totalP):
            flipIndices1 = np.where(np.in1d(p1, duplicatedP))[0]
            flipIndices2 = np.where(np.in1d(p2, duplicatedP))[0]
            flipIndices = np.concatenate((flipIndices1, flipIndices2))
            crossInd1[flipIndices] = 1
        crossInd2 = np.subtract(np.ones(nLineup), crossInd1)
        # Create children and return
        child1 = np.add(np.multiply(crossInd1, p1), np.multiply(crossInd2, p2))
        child2 = np.add(np.multiply(crossInd2, p1), np.multiply(crossInd1, p2))
        fitness1 = calcObj(child1)
        fitness2 = calcObj(child2)
        child1 = np.append(child1, fitness1)
        childArr[0, :] = child1
        child2 = np.append(child2, fitness2)
        childArr[1, :] = child2
        return childArr

    # Runs steepest ascent local search heuristic with an add/drop neighborhood structure
    def steepestAscent(sln, slnFit):
        # Get current solution
        localOptSln = sln.copy()
        localOpt = slnFit.copy()
        while True:  # Until a local optimum is reached
            # Iterate through all solutions in the add/drop neighborhood of current best
            nbd_ad = np.zeros((1, nLineup))
            nbd_obj = np.zeros((1))
            for i in range(nLineup):
                interNbdSol = localOptSln.copy()
                exchanges = np.array(posCat[lineupClass[i]])
                numExchanges = len(exchanges)
                exchanges = exchanges[list(exchanges[i] not in interNbdSol for i in range(numExchanges))]
                for j in exchanges:
                    nextNbdSol = interNbdSol.copy()
                    nextNbdSol[i] = j
                    nbd_ad = np.append(nbd_ad, [nextNbdSol], axis=0)
                    nbd_obj = np.append(nbd_obj, calcObj(nextNbdSol))
            # Get best neighborhood solution
            sortedObjInd = np.flip(np.argsort(nbd_obj))
            nbd_ad = nbd_ad[sortedObjInd, :]
            nbd_obj = nbd_obj[sortedObjInd]
            # Update the best solution overall if better solution is found by neighborhood search;
            # Else, we've reached a local optimum
            if nbd_obj[0] > localOpt:
                localOptSln = nbd_ad[0, :]
                localOpt = nbd_obj[0]
            else:
                break

        return localOptSln, localOpt

    # Carries out elitist replacement strategy
    def elitist(concatSln):
        candidatePopulation = np.concatenate(concatSln, axis=0)
        candidatePopulation = keepSetUnique(candidatePopulation)
        candidateFitness = candidatePopulation[:, nLineup]
        candidateSortOrder = np.flip(np.argsort(candidateFitness))
        candidatePopulation = candidatePopulation[candidateSortOrder, :]
        solutions = candidatePopulation[:popSize, :nLineup]
        fitness = candidatePopulation[:popSize, nLineup]
        return solutions, fitness

    # Create popSize unique lineups, regardless of position
    solutions = np.zeros((popSize, nLineup))
    setSolutions = [[]]
    fitness = np.zeros(popSize)
    for i in range(popSize):
        while True:
            solution = np.repeat(-1, nLineup)
            for posInd in range(len(lineupClass)):
                while True:
                    player = random.choice(posCat[lineupClass[posInd]])
                    if player not in solution:
                        solution[posInd] = player
                        break
            solutionSet = set(solution.tolist())
            if solutionSet not in setSolutions:
                solutions[i, :] = solution
                fitness[i] = calcObj(solution)
                setSolutions.append(solutionSet)
                break
    # Sort solutions by fitness and assign current optimal
    sortFitOrder = np.flip(np.argsort(fitness))
    solutions = solutions[sortFitOrder, :]
    fitness = fitness[sortFitOrder]
    optFit = np.max(fitness)
    optSol = solutions[np.argmax(fitness), :]
    # Define genetic algorithm parameters
    nIm = 0
    startTime = time.time()
    while nIm < nImp:  # Until the algorithm reaches nImp iterations without improvement
        # Create mating pool using roulette wheel scheme
        popFitness = sum(fitness)
        relFitness = np.divide(fitness, np.repeat(popFitness, popSize))
        for ind in range(1, popSize):
            relFitness[ind] = relFitness[ind] + relFitness[ind - 1]
        mateRand = np.random.rand(popSize)
        matingPool = np.zeros(popSize)
        for ind in range(popSize):
            matingPool[ind] = np.min((relFitness > mateRand[ind]).nonzero())
        # Get unique members of mating pool
        uniqueMP = np.unique(matingPool)
        numUnique = len(uniqueMP)
        mateChildren = [np.repeat(-10000000000, 10)]
        if numUnique > 1:
            # Choose sets of parents for crossover
            possibleParents = np.array(list(it.combinations(uniqueMP, 2)))
            numPossibleParents = np.shape(possibleParents)[0]
            numParentsChosen = min(math.floor(popSize / 2), numPossibleParents)
            parentCombsChosen = np.random.choice(numPossibleParents, numParentsChosen, replace=False)
            # Create crossover children
            mateChildren = np.zeros((2 * numParentsChosen, nLineup + 1))
            for i in range(numParentsChosen):
                mateChildren[(2 * i):(2 * i + 2), :] = uniCross(
                    solutions[int(possibleParents[parentCombsChosen[i], 0]), :],
                    solutions[int(possibleParents[parentCombsChosen[i], 1]), :])
        # Create children from mutation
        mutChildren = np.empty((0, nLineup + 1))
        for i in range(popSize):
            mutSol = solutions[i, :].copy()
            for j in range(nLineup):
                if random.random() > bitwiseMut:
                    continue
                pos = lineupClass[j]
                possReplacements = np.array(posCat[pos])
                numPossibleReplacements = len(possReplacements)
                possReplacements = possReplacements[
                    list(possReplacements[i] not in mutSol for i in range(numPossibleReplacements))]
                if len(possReplacements) > 0:
                    replacement = np.random.choice(possReplacements, 1)
                    mutSol[j] = replacement
            mutFitness = calcObj(mutSol)
            mutSol = np.array([np.append(mutSol, mutFitness)])
            mutChildren = np.append(mutChildren, mutSol, axis=0)
        # Combine parents and children and produce the next population using a fully elitist strategy
        fitness = fitness[..., np.newaxis]
        solutions = np.append(solutions, fitness, axis=1)
        solutions, fitness = elitist((solutions, mateChildren, mutChildren))
        # Update current best solution if this population improved it
        if fitness[0] > optFit:
            optSol = solutions[0, :]
            optFit = fitness[0]
            nIm = 0
        else:
            nIm += 1
    # Run steepest ascent on all solutions in final population
    localSearchSln = np.zeros((popSize, nLineup + 1))
    for i in range(popSize):
        localSln, localFit = steepestAscent(solutions[i, :], fitness[i])
        localSearchSln[i, :] = np.append(localSln, localFit)
    fitness = fitness[..., np.newaxis]
    solutions = np.append(solutions, fitness, axis=1)
    # Carry out elitist strategy on final population solutions and their
    # local optima to get final lineups
    solutions, fitness = elitist((solutions, localSearchSln))
    algoTime = time.time() - startTime
    exportDict = {'Position': lineupClass}
    for i in range(popSize):
        exp_lineup = names[solutions[i, :].astype('int32')]
        exportDict['L' + str(i + 1)] = exp_lineup
    exp_df = pd.DataFrame.from_dict(exportDict)
    if output_file:
        exp_df.to_csv(output_file, index=False)
    return exp_df, fitness, algoTime
