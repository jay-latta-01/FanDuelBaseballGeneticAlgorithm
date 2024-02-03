'''
Use this file to run the algorithm after setting your population size, mutation rate,
number of iterations without improvement before exit, input file, and output file
(optional). You can test the algorithm using different parameter values to see which
yield the minimum average deviation from the true optimal or best found solution.
Additionally, you can set your popSize to give you the number of optimized lineups
you want if you are using this to play FanDuel fantasy baseball and export them to
a csv file using the output_file parameter.
'''

import geneticAlgorithm as ga

popSize = 20
bitwiseMut = 0.5
nImp = 10
input_file = "FDPlayerDataProcessed/players_2023-09-29.csv"
# Optional
output_file = None
lineups, fitness, execTime = ga.run(popSize, bitwiseMut, nImp, input_file, output_file)
print(execTime, fitness)