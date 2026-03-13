# Sixth Sense Simulator

A prototype script for solving the **driver–route assignment problem** using [Google OR-Tools](https://developers.google.com/optimization). Given a pre-computed CVRP solution (a set of routes), the script assigns each route to a driver by maximizing the total *sixth sense* — a familiarity score that captures how well each driver knows the destinations along a route. The solver used is **CBC** (Coin-OR Branch and Cut) through the OR-Tools linear solver wrapper.

## Overview

The pipeline works in three stages:

1. **Load a CVRP solution** from a JSON file containing pre-computed routes.
2. **Build a familiarity matrix** φ\_i^d (driver × destination). In this prototype the matrix is generated via simulation; it can be replaced with real-world data (e.g., historical service times).
3. **Derive the route-level matrix** φ\_p^d (driver × route) by averaging destination-level scores, then **solve an Integer Programming model** that assigns exactly one driver to each route while maximizing total familiarity.

## Requirements

- Python 3.8+
- [ortools](https://pypi.org/project/ortools/)
- [numpy](https://pypi.org/project/numpy/)
- [matplotlib](https://pypi.org/project/matplotlib/)

Install all dependencies with:

```bash
pip install ortools numpy matplotlib
```

## Usage

```bash
python sixth_sense_simulator.py
```

The script will prompt for the path to a CVRP solution JSON file:

```
=== Assegnamento driver–percorso basato su sixth sense (prototipo) ===
Inserisci il percorso del file JSON di soluzione CVRP: solution_LDG30_1265_rain_30_0001.json
```

## Input Format

The script expects a JSON file structured as follows:

```json
{
    "instance_name": "LDG30_1265_rain_30_0001",
    "objective": 4580.47,
    "total_travel_time": 4580.47,
    "total_load": 1349,
    "routes": {
        "1": {
            "route": [0, 22, 3, 21, 24, 4, 13, 28, 9, 2, 5, 0],
            "travel_time": 2173.79,
            "load": 752
        },
        "2": {
            "route": [0, 19, 25, 29, 8, 15, 11, 26, 16, 10, 20, 23, 7, 27, 6, 18, 30, 14, 12, 1, 17, 0],
            "travel_time": 2406.68,
            "load": 597
        }
    }
}
```

Each route is a sequence of node indices where `0` represents the depot. The depot appears at the beginning and end of every route and is automatically excluded from the familiarity computation.

An example solution file (`solution_LDG30_1265_rain_30_0001.json`) is included in this repository.

## Output

The script produces:

- **Console output** — solver details, the φ\_i^d and φ\_p^d matrices, and the optimal assignment.
- **Heatmap** — a color-coded matrix of φ\_p^d values with a ★ marker on each assigned driver–route pair.

## Project Context

This script is part of the **SmartDelivery** research project, which investigates AI/ML-driven approaches to the Capacitated Vehicle Routing Problem (CVRP). The *sixth sense* concept models driver expertise as a quantifiable familiarity score that can be integrated into route-to-driver assignment decisions.

The script is based on the following paper:

> La Delfa, G. C., Prieto, J., Monteleone, S., & Rafique, H. (2026). SmartDelivery: IoT Platform for Last Mile Delivery Optimization with Real-Time Data and ML-Based Algorithm Selection. *Distributed Computing and Artificial Intelligence, 22nd International Conference (DCAI 2025)*, Lille, France. Zenodo. [https://doi.org/10.1007/978-3-032-04160-9_6](https://doi.org/10.1007/978-3-032-04160-9_6)


available at: [https://zenodo.org/records/18631923] (preprint) and at: [https://doi.org/10.1007/978-3-032-04160-9_6] 
