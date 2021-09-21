# Influence Maximization

## Guide: Prof. Shivaram Kalyanakrishnan, CS 747, IIT Bombay

This repository contains the following algorithms for Regret Minimization in the Multi-Armed Bandit setting:
 - Epsilon Greedy
 - UCB
 - KL-UCB
 - Thompson Sampling

Along with the above algorithms, it also contains the adaptation of the best performing algorithm - Thompson Sampling to the Finite Support variant and the Thresholded Reward variant of the multi-armed bandit.
For detailed information, kindly take a look at `report.pdf`.

### Requirements
 - Python 3
	 - Numpy
	 - Pandas (for output parsing, optional)
	 - Matplotlib (for plots, optional) 


### Usage
The main file for running the algorithms resides in `src/bandit.py` which only does a single run based on parameters.
```
usage: python3 bandit.py [-h] --instance in [--algorithm al] [--randomSeed rs] [--epsilon ep] [--scale c] [--threshold th]
                 [--horizon hz]

optional arguments:
  -h, --help       show this help message and exit
  --instance in    Path to the instance file (default: None)
  --algorithm al   Algorithm to use (default: epsilon-greedy-t1)
  --randomSeed rs  Number to set as seed for RNG (default: 42)
  --epsilon ep     Epsilon for epsilon greedy (default: 0.02)
  --scale c        Scale factor for exploration bonus of UCB (default: 2)
  --threshold th   Threshold for Task 4 (default: 0)
  --horizon hz     Total number of time steps (default: 10000)
```
To run/iterate on multiple parameters, `src/runner.py` can be used with appropriate modification.

### Output
The output of a single run is output as a comma delimited list of parameters and regret values. The format is as follows:
```
../instances/instances-task1/i-1.txt, epsilon-greedy-t1, 0, 0.02, 2, 0, 102400, 509, 0
```
Each field has the following column names,
```
instance, algorithm, randomSeed, epsilon, scale, threshold, horizon, regret, highs
```
To know more about the definition of the above columns, kindly take a look at `Assignment.pdf`. The above output format can be readily imported as a csv through any appropriate library.
A sample output over various parameters is present in `src/outputData.txt` file, which can be read as a CSV for further static analysis. To read a txt file as a Pandas DataFrame, `get_df` function in `src/plot.py` can be used. `plot.py` contains the code for plotting various graphs to analyse trends across parameters. 
