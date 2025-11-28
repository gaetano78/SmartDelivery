# -*- coding: utf-8 -*-
"""
CVRP Solver - Cost = travel time between destination.
User can choose instance and parameters
"""

import vrplib
import sys

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import numpy as np
import time 
import matplotlib.pyplot as plt   
import matplotlib.colors as mcolors

import json   
import os   

#----------------------------------------------------------#
#--- HERE WE CREATE THE DATA MODEL FOR THE CVRP PROBLEM ---#
#----------------------------------------------------------#
#--- INPUT= the instance, a dictionary containing the keys in the CVRPLIB format:
#    dict_keys(['name', 'comment', 'type', 'dimension', 'edge_weight_type', 'capacity', 'node_coord', 'demand', 'depot'])
#    Also the algorithm need the minimum number of vehicles.
def create_data_model(instance, num_vehicles):
    """Stores the data for the problem based on the VRPLIB instance."""
    data = {}     

    data['scale_factor'] = 1000  # just a scale factor 

    #--- Coordinates of nodes
    data['locations'] = instance['node_coord']

    if instance['edge_weight_type'] == 'EXPLICIT':
        raw_matrix = instance['edge_weight']
        data['cost_matrix'] = [
            [int(round(entry * data['scale_factor'])) for entry in row]
            for row in raw_matrix
        ]
        # Here I print some values for debug purposes 
        print(">>> EXPLICIT matrix loaded and scaled:")
        print(data['cost_matrix'])
    else:
        print("Error: EDGE_WEIGHT_TYPE format incorrect. Expected 'EXPLICIT'.")
        sys.exit(1)
    
    #demands for each node (depot has demand equal to 0)
    data['demands'] = instance['demand']
    #capacity of vehicles (the same for all vehicles)
    data['vehicle_capacities'] = [instance['capacity']] * num_vehicles
    #--- min_numb_of_vehicles = (Total_demand)/(Vehicle_Capacity)
    data['num_vehicles'] = num_vehicles
    # Depot index (assuming it's zero-indexed)
    data['depot'] = int(instance['depot'][0])
    return data

def get_search_parameters(heuristic_name, use_metaheuristic, metaheuristic_name, time_limit):
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.log_search = True
    # print("Solver status: ", solver.status()) give us:
    #   0	ROUTING_NOT_SOLVED: Problem not solved yet.
    #   1	ROUTING_SUCCESS: Problem solved successfully.
    #   2	ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED: Problem solved successfully after calling RoutingModel.Solve(), except that 
    #       a local optimum has not been reached. Leaving more time would allow improving the solution.
    #   3	ROUTING_FAIL: No solution found to the problem.
    #   4	ROUTING_FAIL_TIMEOUT: Time limit reached before finding a solution.
    #   5	ROUTING_INVALID: Model, model parameters, or flags are not valid.
    #   6	ROUTING_INFEASIBLE: Problem proven to be infeasible.
    #list of heuristics in ORTools
    heuristics = {
        # Lets the solver detect which strategy to use according to the model being solved.
        'AUTOMATIC': routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
        # Starting from a route "start" node, connect it to the node which produces the cheapest route segment, then 
        # extend the route by iterating on the last node added to the route.
        'PATH_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        # Similar to PATH_CHEAPEST_ARC, but arcs are evaluated with a comparison-based selector which will favor the most 
        # constrained arc first. To assign a selector to the routing model, use the method ArcIsMoreConstrainedThanArc().
        'PATH_MOST_CONSTRAINED_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        # Similar to PATH_CHEAPEST_ARC, except that arc costs are evaluated using the function passed to SetFirstSolutionEvaluator().
        'EVALUATOR_STRATEGY': routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY,
        # Savings algorithm (Clarke & Wright). Reference Clarke, G. & Wright, J.W. "Scheduling of Vehicles from a Central Depot to a Number of Delivery Points"
        'SAVINGS': routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        # Sweep algorithm (Wren & Holliday). Reference Anthony Wren & Alan Holliday Computer Scheduling of Vehicles from One or More Depots to a Number 
        # of Delivery Points 
        'SWEEP': routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        #Christofides algorithm (actually a variant of the Christofides algorithm using a maximal matching instead of a maximum matching, 
        # which does not guarantee the 3/2 factor of the approximation on a metric travelling salesperson). Works on generic vehicle routing models 
        # by extending a route until no nodes can be inserted on it. 
        # Reference Nicos Christofides, Worst-case analysis of a new heuristic for the travelling salesman problem.
        'CHRISTOFIDES': routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        # Make all nodes inactive. Only finds a solution if nodes are optional (are element of a disjunction constraint with a finite penalty cost).
        'ALL_UNPERFORMED': routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        # Iteratively build a solution by inserting the cheapest node at its cheapest position; the cost of insertion is based on the global cost function of the routing model. As of 2/2012, only works on models with optional nodes (with finite penalty costs).
        'BEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
        # Iteratively build a solution by inserting the cheapest node at its cheapest position; the cost of insertion is based on the arc cost function. Is faster than BEST_INSERTION.
        'PARALLEL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        # Iteratively build a solution by inserting each node at its cheapest position; the cost of insertion is based on the arc cost function.
        # Differs from PARALLEL_CHEAPEST_INSERTION by the node selected for insertion; here nodes are considered in their order of creation. 
        # Is faster than PARALLEL_CHEAPEST_INSERTION.
        'LOCAL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        # Iteratively connect two nodes which produce the cheapest route segment.
        'GLOBAL_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
        # Select the first node with an unbound successor and connect it to the node which produces the cheapest route segment.
        'LOCAL_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
        # Select the first node with an unbound successor and connect it to the first available node. This is equivalent to the 
        # CHOOSE_FIRST_UNBOUND strategy combined with ASSIGN_MIN_VALUE (cf. constraint_solver.h).
        'FIRST_UNBOUND_MIN_VALUE': routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE
    }
    #List of metaheuristics
    metaheuristics = {
        # Lets the solver select the metaheuristic.
        'AUTOMATIC': routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
        # Accepts improving (cost-reducing) local search neighbors until a local minimum is reached.
        'GREEDY_DESCENT': routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
        # Uses guided local search to escape local minima This is generally the most efficient metaheuristic for vehicle routing.
        'GUIDED_LOCAL_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        # Uses simulated annealing to escape local minima.
        'SIMULATED_ANNEALING': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        # Uses tabu search to escape local minima.
        'TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        # Uses tabu search on the objective value of solution to escape local minima.
        'GENERIC_TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH
    }

    if heuristic_name in heuristics:
        # set chosen heuristic. 
        search_parameters.first_solution_strategy = heuristics[heuristic_name]
    else:
        print(f"Error: Heuristic '{heuristic_name}' is not valid. Use one of the following: {list(heuristics.keys())}")
        return None

    # Imposta la metaheuristic solo se richiesto
    if use_metaheuristic:
        # Imposta la metaheuristic scelta, se viene passato un valore non valido, mostra un errore
        if metaheuristic_name in metaheuristics:
            search_parameters.local_search_metaheuristic = metaheuristics[metaheuristic_name]
        else:
            print(f"Error: Metaheuristic '{metaheuristic_name}' is not valid. Use one of the following: {list(metaheuristics.keys())}")
            return None
    else:
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT

    search_parameters.time_limit.FromSeconds(time_limit)
    return search_parameters

#-----------------------------------#
#--- PRINT AND SAVE THE SOLUTION ---#
#-----------------------------------#

def print_and_save_solution(data, manager, routing, solution, scale_factor, instance_name, heuristic_name, metaheuristic_name, execution_time):
    """Prints the solution on the console and saves it to a file."""
    print("----------------------------")
    total_travel_time = 0   
    total_load = 0   
    routes = {}
    objective = solution.ObjectiveValue() / scale_factor

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        depot_node = manager.IndexToNode(index)
        route_travel_time = 0
        route_load = 0
        route_nodes = [depot_node]   
        visited_customer = False   
        last_index_before_end = None   

        while True:
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if routing.IsEnd(index):
                if visited_customer:
                    last_index_before_end = previous_index
                break

            node_index = manager.IndexToNode(index)
            route_load += int(data["demands"][node_index])  
            route_nodes.append(node_index)
            route_travel_time += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            visited_customer = True

        if last_index_before_end is not None:
            end_index = routing.End(vehicle_id)
            arc_cost = routing.GetArcCostForVehicle(last_index_before_end, end_index, vehicle_id)
            route_travel_time += arc_cost
            print(f"[DEBUG] Vehicle {vehicle_id + 1}: cost of the last arc (Customer -> depot) = {arc_cost / scale_factor} units")

        route_nodes.append(depot_node)

        total_travel_time += float(route_travel_time)
        total_load += int(route_load)

        print(f"Route for vehicle {vehicle_id + 1}: {route_nodes}")
        print(f"Travel time of the route: {route_travel_time / scale_factor} units")
        print(f"Load of the route: {route_load} units\n")

        # Salviamo il percorso del veicolo
        routes[vehicle_id + 1] = {
            'route': route_nodes,
            'travel_time': float(route_travel_time) / scale_factor,
            'load': int(route_load)
        }

    print(f"[DEBUG] Sum of travel times (scaled): {total_travel_time / scale_factor} vs objective {objective}")
    print(f"Total travel time of all routes: {total_travel_time / scale_factor} units")
    print(f"Total load of all routes: {total_load} units")
    print(f"Objective: {objective} units of travel time")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Heuristic used: {heuristic_name}")
    print(f"Metaheuristic used: {metaheuristic_name}")

    solution_data = {
        'instance_name': instance_name,
        'objective': float(objective),
        'total_travel_time': float(total_travel_time) / scale_factor,
        'total_load': int(total_load),
        'execution_time': float(execution_time),
        'heuristic': heuristic_name,
        'metaheuristic': metaheuristic_name,
        'routes': routes
    }

    solution_filename = f"solution_{instance_name}.json"

    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        else:
            return o.__str__()

    with open(solution_filename, 'w') as f:
        json.dump(solution_data, f, indent=4, default=default)

    print(f"Solution saved in the file {solution_filename}")

#-----------------------------------------------#
#--- FUNCTION TO PLOT ROUTES WITH MATPLOTLIB ---#
#-----------------------------------------------#
def plot_solution(data, manager, routing, solution):
    """Plots the solution routes on a graph."""
    print("Plotting the solution...")
    locations = data['locations']
    depot_index = data['depot']
    colors = list(mcolors.TABLEAU_COLORS.keys())
    plt.figure(figsize=(10, 6))
    ax = plt.gca()  

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = []
        route_node_indices = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(locations[node_index])
            route_node_indices.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
        route.append(locations[depot_index])
        route_node_indices.append(depot_index)
        route = np.array(route)
        color = colors[vehicle_id % len(colors)]

        plt.plot(route[:, 0], route[:, 1], marker='o', color=color, label=f'Vehicle {vehicle_id + 1}', linewidth=2, alpha=0.8)

        plt.plot(route[0, 0], route[0, 1], marker='D', markersize=10, color='black', markerfacecolor='yellow')
        for i in range(len(route)):
            x, y = route[i]
            node_index = route_node_indices[i]
            ax.annotate(str(node_index), (x, y), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title('Routes', fontsize=14)

    legend = ax.legend(loc='best', fontsize=10, framealpha=1, facecolor='white', edgecolor='black', shadow=True)

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()  
    plt.show()

#----------------------------------------------------#
#----- MINIMUM NUMBER OF VEHICLES AND SOME CHECKS ---#
#----------------------------------------------------#
def get_vehicle_number(instance):
    total_demand = sum(instance['demand'][1:])
    min_vehicles = int(np.ceil(total_demand / instance['capacity']))
    
    while True:
        num_vehicles_input = input(
            f"\nNumber of vehicles (min: {min_vehicles}, default: {min_vehicles}): "
        )
        if num_vehicles_input.strip() == '':
            return min_vehicles
        try:
            num_vehicles = int(num_vehicles_input)
            if num_vehicles < min_vehicles:
                print(f"❌ Error: At least {min_vehicles} vehicles are needed to satisfy the total demand of {total_demand} units")
                print(f"   (each vehicle has capacity {instance['capacity']})")
                continue
            return num_vehicles
        except ValueError:
            print("❌ Insert a valid integer number")
            continue


#--------------#
#----- MAIN ---#
#--------------#
def main(): 
    #insert the path of the CVRP instance file
    instance_path = input("Insert the path of CVRP instance (e.g.: 'instances/XML100_1111_01.vrp'): ")
    if not os.path.isfile(instance_path):
        print("The file does not exist.")
        return
    
    instance = vrplib.read_instance(instance_path, compute_edge_weights=False)

    instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    print(f"Instance name = {instance_name}")
    print("----------------------")

    #------------------------------------------#
    #--- PRINT SOME DATA FOR DEBUG PURPOSES ---#
    #------------------------------------------#
    # dict_keys(['name', 'comment', 'type', 'dimension', 'edge_weight_type', 'capacity', 'node_coord', 'demand', 'depot', 'edge_weight'])
    print("DEBUG — available keys in instance:", instance.keys())
    print("----------------------")
    
    num_vehicles = get_vehicle_number(instance)
    print(f"✅ Chosen number of vehicles: {num_vehicles}")
  
    data = create_data_model(instance, num_vehicles)
    
    manager = pywrapcp.RoutingIndexManager(len(data["cost_matrix"]), data["num_vehicles"], data['depot'])

    routing = pywrapcp.RoutingModel(manager)

    def myCost_callback(from_index, to_index):
        """Returns the cost between the two nodes."""
        #--- Convert from routing variable Index to cost matrix NodeIndex. 
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["cost_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(myCost_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return int(data["demands"][from_node])  

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  
        [int(capacity) for capacity in data["vehicle_capacities"]], 
        True,  
        "Capacity",
    )
    capacity_dimension = routing.GetDimensionOrDie("Capacity")

    heuristic = input("Insert the name of the heuristics you want to set (default 'PARALLEL_CHEAPEST_INSERTION'): ")
    if heuristic.strip() == '':
        heuristic = 'PARALLEL_CHEAPEST_INSERTION'

    use_metaheuristic_input = input("Do you want to choose a metaheuristic? (y/n, default 'y'): ")
    if use_metaheuristic_input.lower() == 'n':
        use_metaheuristic = False  
        metaheuristic = 'GREEDY_DESCENT'
        print("DEFAULT METAHEURISTIC USED: GREEDY_DESCENT")
    else:
        use_metaheuristic = True
        metaheuristic = input("Insert the name of the metaheuristic you want to set (default 'GUIDED_LOCAL_SEARCH'): ")
        if metaheuristic.strip() == '':
            metaheuristic = 'GUIDED_LOCAL_SEARCH'

    time_limit_input = input("Insert the time limit in seconds (default 300): ")
    if time_limit_input.strip() == '':
        time_limit = 300
    else:
        time_limit = int(time_limit_input)

    #--- 11) definisce i parametri di ricerca per il solutore del problema di routing. Questi parametri determinano come il solutore
    #    cerca la first solution e come esegue l'ottimizzazione per migliorare la soluzione
    #    search_parameters è l'oggetto che serve per configurare il comportamento del solver. Nello specifico, chiedo al solver 
    #    di risolvere il problema utilizzando i parametri di ricerca definiti. E' una specie di inizializzazione. 
    #    Poi i parametri li sistemo dopo

    # Crea i parametri di ricerca utilizzando una funzione, mi sarà comoda in seguito
    search_parameters = get_search_parameters(heuristic, use_metaheuristic, metaheuristic, time_limit)
    if search_parameters is None:
        return # Interrompe l'esecuzione se c'è stato un errore nei parametri

    # Risoluzione del problema
    print("Solving the problem...")
    start_time = time.time()
    # ecco dove mi è comoda!
    solution = routing.SolveWithParameters(search_parameters)
    execution_time = time.time() - start_time

    print("Solver status: ", routing.status())
    if solution:
        print_and_save_solution(
            data, manager, routing, solution, data['scale_factor'],
            instance_name, heuristic, metaheuristic, execution_time
        )
        # Plot della soluzione
        plot_solution(data, manager, routing, solution)
    else:
        print("No solution found!")
#--- Richiamo il main
if __name__ == "__main__":
    main()