'''
This file contains a mathematical optimization problem to maximize expected total
points in a FanDuel fantasy baseball lineup solved using the GLPK Mixed Integer
Linear Programming solver. It is intended to be used to compare results of the
genetic algorithm to the true optimal lineup for a given set of players.
'''

import pyomo.environ as pyo
import pandas as pd

#Read and process player data from scrapeFD.py
players = pd.read_csv('FDPlayerDataProcessed/players_2023-09-29.csv')
positions = players['position'].tolist()
positionList = ['P', 'C/1B', '2B', '3B', 'SS', 'OF', 'Util']
model = pyo.ConcreteModel()
model.players = range(len(players)) #set of players
model.B = 35000 #Budget
model.positions = range(len(positionList)) #set of positions
model.vij = {} #projected fantasy points of using player i in position j
model.cij = {} #cost of using player i in position j
model.aij = {} # aij = 1 if player i can perform position j; 0 otherwise
model.tj = [1,1,1,1,1,3,1] #Number of players needed for position j
#xij = 1 if player i is assigned to play position j; 0 otherwise
model.xij = pyo.Var(model.players,model.positions,within=pyo.Binary)
#Assign parameter values
for i in model.players:
    for j in model.positions:
        model.vij[(i,j)] = players.iloc[i,3]
        model.cij[(i,j)] = players.iloc[i,2]
        if j == 1:
            if ('C' in positions[i]) or ('1B' in positions[i]):
                model.aij[(i, j)] = 1
            else:
                model.aij[(i, j)] = 0
            continue
        if positionList[j] in positions[i]:
            model.aij[(i,j)] = 1
        else:
            model.aij[(i,j)] = 0
    if 'P' not in positions[i]:
         model.aij[(i,len(positionList)-1)] = 1
#Define objective function
model.obj = pyo.Objective(expr = sum(model.vij[(i,j)]*model.xij[(i,j)] for j in
                                 model.positions for i in model.players),sense=pyo.maximize)
#Solution contains required number of players at each position
model.requirements = pyo.ConstraintList()
for j in model.positions:
    model.requirements.add(sum(model.xij[(i,j)] for i in model.players)
                           == model.tj[j])
#Budget cannot be exceeded
model.budget = pyo.Constraint(
    expr = sum(model.xij[(i,j)]*model.cij[(i,j)] for j in model.positions
               for i in model.players) <= model.B
)
#Players are only assigned positions they can perform
model.withinAbility = pyo.ConstraintList()
for i in model.players:
    for j in model.positions:
        model.withinAbility.add(
            model.xij[(i,j)] <= model.aij[(i,j)]
        )
#Players can only be assigned to play 1 position
model.singleAssignment = pyo.ConstraintList()
for i in model.players:
    model.singleAssignment.add(
        sum(model.xij[(i,j)] for j in model.positions) <= 1
    )
#Solve model
solver = pyo.SolverFactory('glpk')
solver.solve(model)
print(round(pyo.value(model.obj), 2))
