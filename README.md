# privacy-scheduling-tools: metrics optimization framework
Based on https://github.com/hu-dbis/privacy-scheduling-tools.git.

The objective of this repository is to optimize the epsilon metric for a fixed value of the delta metric and the other way around.

Based on this, the code is divided into two streams:

twct_optimization, delta_optimization.

Three algorithms were used for optimization: Genetic Algorithm, Simulated Annealing, and Beam Search.

All results, including the generation of three calendar scenarios and the optimization results from the three algorithms, are also provided in the results folder.

Python version.
A set of scripts for privacy attacks on schedules and utility-aware privacy protection.

# Prerequisites

Linux, Python in 3.9:
1. Create a virtual environment with `python3 -m venv env`
2. Activate the virtual environment with `source env/bin/activate`
3. Install needed packages with `python3 -m pip install -r requirements.txt`
4. Set pythonpath with `export PYTHONPATH=.`
