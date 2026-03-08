# CVRP Solver with Travel-Time Costs

This folder contains a Python implementation of a **Capacitated Vehicle Routing Problem (CVRP)** solver where **travel time** is used as the optimization cost instead of Euclidean distance.  
The code is part of a research activity carried out within a **Marie Skłodowska-Curie Postdoctoral Fellowship (MSCA)** under Horizon Europe.


## Overview

- Solves CVRP instances where the cost matrix encodes **travel times** between nodes.
- Supports **VRPLIB-style** input format with an explicit, symmetric cost matrix (`EDGE_WEIGHT_TYPE: EXPLICIT`, `EDGE_WEIGHT_FORMAT: FULL_MATRIX`).
- Designed for **algorithm selection and benchmarking**.
- Written in Python, intended for **reproducible research** and extension.


## Requirements

- Python ≥ 3.9
- Dependencies:
  - `ortools`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `vrplib`


## How to Use

**1. Launch the script:**
```bash
python "CVRP solver - travel times as cost - 14-11-2025 - cleaned.py"
```

**2. Provide the path to a CVRP instance** when prompted. A test instance is included in this folder.

**3. Set the number of vehicles** (or accept the minimum value suggested by the script).

**4. Select a construction heuristic:**

| Heuristic |
|---|
| `AUTOMATIC` |
| `PATH_CHEAPEST_ARC` |
| `PATH_MOST_CONSTRAINED_ARC` |
| `EVALUATOR_STRATEGY` |
| `SAVINGS` |
| `SWEEP` |
| `CHRISTOFIDES` |
| `ALL_UNPERFORMED` |
| `BEST_INSERTION` |
| `PARALLEL_CHEAPEST_INSERTION` |
| `LOCAL_CHEAPEST_INSERTION` |
| `GLOBAL_CHEAPEST_ARC` |
| `LOCAL_CHEAPEST_ARC` |
| `FIRST_UNBOUND_MIN_VALUE` |

**5. Optionally select a metaheuristic:**

| Metaheuristic |
|---|
| `AUTOMATIC` |
| `GREEDY_DESCENT` |
| `GUIDED_LOCAL_SEARCH` |
| `SIMULATED_ANNEALING` |
| `TABU_SEARCH` |
| `GENERIC_TABU_SEARCH` |

For more details on routing options, see the [OR-Tools documentation](https://developers.google.com/optimization/routing/routing_options).

**6. Set a timeout** to limit the solver's running time.


## Output

The solver plots the CVRP instance and the found solution, and saves the results to a `.json` file in the same directory.


## Input Format

Input instances must follow the VRPLIB format, with:
- `EDGE_WEIGHT_TYPE: EXPLICIT`
- `EDGE_WEIGHT_FORMAT: FULL_MATRIX`

For more details on the format, see the [VRPLIB documentation](https://github.com/PyVRP/VRPLIB?tab=readme-ov-file#vrplib-instance-format).





