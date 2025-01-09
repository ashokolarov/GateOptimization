# Airport Gate Assignment Optimization

## Overview
This project focuses on optimizing airport gate assignments using a Mixed-Integer Linear Programming (MILP) approach. It balances two critical objectives:
1. Minimizing the total fuel consumption of aircraft during taxi operations.
2. Minimizing the total walking distance for passengers from gates to baggage carousels.

The model was tested with realistic flight schedules at the apron of Ankara Esenboga Airport and demonstrated robustness and effectiveness in achieving its goals.

---

## Features
- Joint optimization of multiple objectives for operational efficiency and passenger comfort.
- Mathematical model implemented using the Gurobi optimization package in Python.
- Includes sensitivity analysis for parameters such as buffer times, taxi speeds, and objective weighting.

---

## Usage
### Prerequisites
- Python 3.x
- Gurobi Optimization package
- Required Python packages listed in `requirements.txt`

### File Structure
- `data.xlsx`: Contains flight, gate, and baggage carousel data.
- `model.py`: Defines the MILP model and solves optimization problems.
- `data_processing.py`: Generates flight schedules and preprocesses input data.
- `utils.py`: Contains helper functions for visualization and data management.
- `config.yaml`: Configurable parameters for the optimization problems.

### Acknowledgments
- Developed as part of the AE4441-16 Operations Optimization course at TU Delft.
- Based on the MILP framework proposed by Cecen et al. (2021).
- Supervisors: Ir. P.C. Roling and A. Bombelli.



### References 
1. Cecen, R. K. Multi-objective optimization model for airport gate assignment problem. Aircraft Engineering and Aerospace Technology, 2021.
2. Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual, 2023.
