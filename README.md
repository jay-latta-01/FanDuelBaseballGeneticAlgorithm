# FanDuel Baseball Genetic Algorithm

This repo centers around a genetic algorithm used to generate FanDuel fantasy baseball lineups optimized for expected player points. It contains the algorithm, a test file to run it with different parameters, a Pyomo mixed-integer programming model to find the optimal expected points for a given set of players, and processed player data files containing player names, positions, costs, and average fantasy points.

## Installation

Clone the repo into a Python virtual environment and install required packages using [pip](https://pip.pypa.io/en/stable/)
```bash
pip install requirements.txt
```
## Getting Started
Navigate to the test file (runGA.py) and change the following parameters as you see fit: 
```python
popSize = <YOUR POPULATION SIZE>
bitwiseMut = <YOUR MUTATION RATE>
nImp = <YOUR NUMBER OF ITERATIONS WITHOUT IMPROVEMENT>
input_file = <YOUR INPUT FILE>
output_file = <YOUR OUTPUT FILE (optional)>
```
Pre-processed data for testing can be found in the FDPlayerDataProcessed folder. If you are utilizing your own data files, ensure they are formatted and processed the same as those in FDPlayerDataProcessed. 

Then, add additional code to utilize or analyze your outputs if needed and run the file.

## Analyzing Results

The results of running the algorithm with the parameters you specify can be analyzed by examining the execution time (included in the function outputs) and comparing the "fitness" of the best found solution to the true optimal obtained by solving the mathematical model, which is included in mathModel.py. To execute this file and get the optimal total value for a given set of players, _you must install the glpk solver to your machine._ 

## License

This repository is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)