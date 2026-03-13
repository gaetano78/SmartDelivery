#!/usr/bin/env python3
"""
Example script for solving a driver–route assignment problem
using OR-Tools, based on:

1) A JSON file containing a CVRP solution (pre-computed routes).
2) A familiarity matrix φ_i^d (driver–destination) GENERATED via simulation.
3) The derivation of the matrix φ_p^d (driver–route) and the resolution of
   an Integer Programming problem to assign routes to drivers
   maximizing the total "sixth sense".

The script is intended as a prototype, easily adaptable in the future
when real service time / familiarity data become available.
"""

import json
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ortools.linear_solver import pywraplp


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS FOR LOADING THE JSON AND STRUCTURING THE DATA
# ---------------------------------------------------------------------------

def load_cvrp_solution(json_path: str) -> Dict:
    """
    Loads the JSON file containing the CVRP solution.

    The expected format is similar to:

    {
        "instance_name": "...",
        "objective": 4580.47,
        "routes": {
            "1": {
                "route": [0, 22, 3, 21, ..., 5, 0],
                "travel_time": ...,
                "load": ...
            },
            "2": {
                "route": [0, 19, 25, ..., 17, 0],
                "travel_time": ...,
                "load": ...
            }
        }
    }

    Returns the corresponding Python dictionary.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def extract_routes_and_customers(solution_data: Dict) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    Extracts route and customer information from the JSON data.

    - For each route p, we take the 'route' node list and remove the depot
      (assumed to be encoded as '0' at the beginning and end).
    - We thus obtain a mapping:
        routes[p] = [customer_list_without_0]
    - We also compute the total set of customers as the union of all lists.

    Returns:
    - routes: dict[int, list[int]]  -> e.g. {0: [22,3,21,...], 1: [19,25,...]}
    - customers: list[int]          -> sorted list of distinct customers
    """
    raw_routes = solution_data.get("routes", {})

    routes: Dict[int, List[int]] = {}
    customers_set = set()

    # The keys in the JSON are typically strings ("1", "2", ...).
    # We convert them to 0-based integers for convenience (0, 1, 2, ...).
    for idx, key in enumerate(sorted(raw_routes.keys(), key=lambda k: int(k))):
        route_info = raw_routes[key]
        full_route = route_info.get("route", [])

        # Remove the depot (0) at the beginning and end, if present
        # Safety checks to avoid errors if the format varies.
        customers_in_route = [
            node for node in full_route
            if node != 0
        ]

        routes[idx] = customers_in_route
        customers_set.update(customers_in_route)

    customers = sorted(customers_set)
    return routes, customers


# ---------------------------------------------------------------------------
# SIMULATED GENERATION OF THE MATRIX φ_i^d (driver–destination)
# ---------------------------------------------------------------------------

def generate_phi_id(
    drivers: List[str],
    customers: List[int],
    random_seed: int = 42
) -> Dict[str, Dict[int, float]]:
    """
    Generates a simulated familiarity matrix φ_i^d (driver–destination).

    Idea:
    - Each driver has a subset of customers for which they are particularly skilled.
    - For these "specialized" customers we assign high familiarity values (e.g. 0.8-1.0).
    - For the remaining customers we assign medium-low values (e.g. 0.2-0.7).

    This model is purely simulated, but it is consistent with the idea that
    each driver knows certain areas/routes better than others.

    Returns:
    - phi_id[driver][customer] = familiarity in [0,1].
    """
    random.seed(random_seed)

    phi_id: Dict[str, Dict[int, float]] = {}

    num_customers = len(customers)
    if num_customers == 0:
        # Rare case, but we handle it anyway: no customers
        for d in drivers:
            phi_id[d] = {}
        return phi_id

    for d in drivers:
        # Determine how many customers will be "preferred" by this driver.
        # We use ~40% of the customers (at least 1).
        num_special = max(1, int(0.4 * num_customers))

        # Randomly select the "specialized" customers for this driver.
        special_customers = set(random.sample(customers, num_special))

        phi_id[d] = {}
        for c in customers:
            if c in special_customers:
                # "Specialized" customer: high familiarity
                phi_id[d][c] = random.uniform(0.8, 1.0)
            else:
                # "Non-specialized" customer: medium-low familiarity
                phi_id[d][c] = random.uniform(0.2, 0.7)

    return phi_id


# ---------------------------------------------------------------------------
# COMPUTATION OF THE MATRIX φ_p^d (driver–route) FROM φ_i^d
# ---------------------------------------------------------------------------

def compute_phi_pd(
    phi_id: Dict[str, Dict[int, float]],
    routes: Dict[int, List[int]],
    drivers: List[str]
) -> Dict[str, Dict[int, float]]:
    """
    Computes the matrix φ_p^d (driver–route) as the average of the
    familiarity values φ_i^d over the destinations of route p.

    - routes[p] = [customer_list of route p]
    - phi_id[driver][customer] = driver–destination familiarity

    Definition:
        φ_p^d = (1 / |I_p|) * sum_{i in I_p} φ_i^d

    where I_p is the set of destinations in route p.

    Returns:
    - phi_pd[driver][route_index] = average driver familiarity on the route.
    """
    phi_pd: Dict[str, Dict[int, float]] = {}

    for d in drivers:
        phi_pd[d] = {}
        for p, customers_in_route in routes.items():
            if not customers_in_route:
                # Route with no customers (anomalous case) -> familiarity 0
                phi_pd[d][p] = 0.0
            else:
                values = [
                    phi_id[d].get(c, 0.0)
                    for c in customers_in_route
                ]
                phi_pd[d][p] = sum(values) / len(values)

    return phi_pd


# ---------------------------------------------------------------------------
# DRIVER–ROUTE ASSIGNMENT RESOLUTION WITH OR-TOOLS
# ---------------------------------------------------------------------------

def solve_assignment_with_ortools(
    phi_pd: Dict[str, Dict[int, float]],
    drivers: List[str],
    routes: Dict[int, List[int]]
):
    """
    Sets up and solves the driver–route assignment problem with OR-Tools.

    Model:

    - Binary variables:
        x[d, p] = 1 if driver d is assigned to route p, 0 otherwise.

    - Objective:
        max sum_{d} sum_{p} φ_p^d * x[d,p]

      i.e., maximize the sum of driver–route familiarity (sixth sense).

    - Constraints:
        1) Each driver can be assigned to at most one route:
            sum_{p} x[d,p] <= 1  for each driver d
        2) Each route must be assigned to exactly one driver:
            sum_{d} x[d,p] = 1   for each route p
        3) x[d,p] ∈ {0,1}
    """
    solver = pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        raise RuntimeError("Impossibile creare il solver CBC di OR-Tools.")
    
    #-------------------------------#
    #--- Timeout in milliseconds ---#
    #-------------------------------#
    solver.SetTimeLimit(10_000) # that is 10 seconds

    # --- Details on the OR-Tools solver used ---
    # In short: CBC is an open-source MIP solver based on branch-and-cut, which combines tree exploration,
    # valid cuts, and heuristics to quickly find good solutions and prove (or approach) optimality.

    print("\n=== OR-Tools solver details ===")
    print(f"Nome solver: CBC (Coin-OR Branch and Cut)")
    print("Descrizione: solver di programmazione lineare intera basato su CBC (branch-and-bound).")
    print("Parametri custom: nessuno (si usano le impostazioni di default).")

    # Create a mapping for integer indices (useful for loops)
    driver_indices = {d: idx for idx, d in enumerate(drivers)}
    route_indices = list(routes.keys())  # e.g. [0, 1, 2, ...]

    # Variables x[d_idx][p_idx] -> BoolVar
    x = {}
    for d in drivers:
        d_idx = driver_indices[d]
        x[d_idx] = {}
        for p in route_indices:
            # Variable name for debugging/readability
            var_name = f"x_d{d_idx}_p{p}"
            x[d_idx][p] = solver.BoolVar(var_name)

    # Objective function: maximize sum_{d,p} φ_p^d * x[d,p]
    objective = solver.Objective()
    for d in drivers:
        d_idx = driver_indices[d]
        for p in route_indices:
            coeff = phi_pd[d][p]
            objective.SetCoefficient(x[d_idx][p], coeff)
    objective.SetMaximization()

    # Constraint 1: each driver assigned to at most one route
    for d in drivers:
        d_idx = driver_indices[d]
        ct = solver.Constraint(0, 1)  # sum <= 1
        for p in route_indices:
            ct.SetCoefficient(x[d_idx][p], 1)

    # Constraint 2: each route assigned to exactly one driver
    for p in route_indices:
        ct = solver.Constraint(1, 1)  # sum == 1
        for d in drivers:
            d_idx = driver_indices[d]
            ct.SetCoefficient(x[d_idx][p], 1)

    # Call the solver
    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        print("Nessuna soluzione trovata (status solver:", status, ")")
        return

    # Build the assignment as a dictionary: assignment[driver] = route_index
    assignment = {}
    for d in drivers:
        d_idx = driver_indices[d]
        for p in route_indices:
            if x[d_idx][p].solution_value() > 0.5:
                assignment[d] = p

    # Basic printout (you can keep this as a "raw" log)
    print("\n=== DRIVER–ROUTE ASSIGNMENT RESULTS ===")
    print(f"Stato soluzione: {'Ottimale' if status == pywraplp.Solver.OPTIMAL else 'Fattibile'}")
    print(f"Valore della funzione obiettivo (somma φ_p^d): {solver.Objective().Value():.4f}\n")
    for d, p in assignment.items():
        print(f"Driver {d} assegnato al percorso {p}")
    print()

    # Return the data to the caller
    return assignment, solver.Objective().Value(), status

#---------------------------------------------#
#--- nice-looking visualization function ---#
#---------------------------------------------#
def visualize_results(phi_pd, drivers, routes, assignment, objective_value, status):
    """
    Displays the results in a clean format:
    - textual summary
    - heatmap of φ_p^d with the assigned driver–route pairs highlighted.
    """
    print("\n=== RESULTS SUMMARY (formatted) ===")
    print(f"Stato soluzione: {'Ottimale' if status == pywraplp.Solver.OPTIMAL else 'Fattibile'}")
    print(f"Valore obiettivo (somma φ_p^d): {objective_value:.4f}\n")

    print("Assegnamento driver–percorso:")
    for d in drivers:
        p = assignment.get(d, None)
        if p is not None:
            print(f"  - {d} -> percorso {p}")
        else:
            print(f"  - {d} non assegnato")
    print()

    # Build the matrix for the heatmap: rows = drivers, columns = routes
    route_keys_sorted = sorted(routes.keys())
    matrix = np.array([
        [phi_pd[d][p] for p in route_keys_sorted]
        for d in drivers
    ])

    driver_labels = drivers
    route_labels = [f"p{p}" for p in route_keys_sorted]

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(route_labels)))
    ax.set_xticklabels(route_labels)
    ax.set_yticks(range(len(driver_labels)))
    ax.set_yticklabels(driver_labels)
    plt.colorbar(im, ax=ax, label="φ_p^d")

    # Write the numerical values in the cells, with a star on the assigned pair
    for i, d in enumerate(drivers):
        for j, p in enumerate(route_keys_sorted):
            val = matrix[i, j]
            mark = "★" if assignment.get(d, None) == p else ""
            ax.text(j, i, f"{val:.2f}{mark}", ha="center", va="center")

    plt.title("Matrice φ_p^d e assegnamento ottimo (★)")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def main():
    """
    Main function:
    - Asks the user for the JSON file path.
    - Loads the CVRP solution.
    - Extracts routes and customers.
    - Defines the drivers (one per route).
    - Generates the simulated φ_i^d matrix.
    - Computes the φ_p^d matrix.
    - Solves the assignment problem with OR-Tools.
    - Prints some debug information.
    """
    print("=== Assegnamento driver–percorso basato su sixth sense (prototipo) ===")

    json_path = input("Inserisci il percorso del file JSON di soluzione CVRP: ").strip()

    # 1) Load data from the JSON
    solution_data = load_cvrp_solution(json_path)
    instance_name = solution_data.get("instance_name", "unknown_instance")
    print(f"\nIstanza: {instance_name}")

    # 2) Extract routes and customers
    routes, customers = extract_routes_and_customers(solution_data)
    num_routes = len(routes)
    num_customers = len(customers)

    print(f"Numero di percorsi (routes): {num_routes}")
    print(f"Numero di clienti distinti: {num_customers}")
    print(f"Clienti: {customers}")

    # 3) Define drivers
    #    For simplicity, we assume one driver per route:
    #    driver_1, driver_2, ..., driver_P
    drivers = [f"driver_{i+1}" for i in range(num_routes)]
    print(f"Driver definiti: {drivers}")

    # 4) Simulated generation of the φ_i^d matrix (driver–destination)
    phi_id = generate_phi_id(drivers, customers, random_seed=42)

     # --- Print φ_i^d matrix (driver–destination) ---
    print("\nMatrice φ_i^d (familiarità driver–destinazione):")
    for d in drivers:
        row_values = [f"{c}: {phi_id[d][c]:.3f}" for c in customers]
        print(f"{d}: " + ", ".join(row_values))

    # 5) Compute the φ_p^d matrix (driver–route)
    phi_pd = compute_phi_pd(phi_id, routes, drivers)

    # Debug printout: φ_p^d matrix
    # clearly there is an index for driver d and an index for route p
    # so with 2 drivers and 2 routes we get a 4-element matrix:
    #   DRIVER_1 => familiarity of driver_1 with route 0, familiarity of driver_1 with route 1
    #   DRIVER_2 => familiarity of driver_2 with route 0, familiarity of driver_2 with route 1
    print("\nMatrice φ_p^d (familiarità driver–percorso):")
    for d in drivers:
        row_values = [f"{phi_pd[d][p]:.3f}" for p in sorted(routes.keys())]
        print(f"{d}: {row_values}")

    # 6) Solve the assignment problem with OR-Tools
    assignment, objective_value, status = solve_assignment_with_ortools(phi_pd, drivers, routes)

    # 7) Clean and graphical visualization of the results
    visualize_results(phi_pd, drivers, routes, assignment, objective_value, status)

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

