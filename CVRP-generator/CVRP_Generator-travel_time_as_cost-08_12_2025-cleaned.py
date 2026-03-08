 # -*- coding: utf-8 -*-

"""
CVRP INSTANCE GENERATOR


1) Generates the customers (coordinates and demand) as in the original Uchoa generation script.

2)Computes the Euclidean distances dist(i,j) and then calculates the travel times between nodes.

3) For each pair (i,j), samples alpha_ij from a uniform distribution and defines the base travel time as
t_base(i,j) = alpha_ij * dist(i,j).

4) Applies road congestion using a shifted and truncated exponential distribution (upper bound 2.5), obtaining
t_congest(i,j) = t_base(i,j) * c_ij.

5) Applies weather congestion via factor w, obtaining t_meteo(i,j).

6) Adds the additive term beta * Gamma_ij, where Gamma_ij is sampled from
U(0,10), U(10,40), or U(40,90) depending on whether dist(i,j) < T1 = 300,
T1 ≤ dist(i,j) < T2 = 700, or dist(i,j) ≥ T2, with beta = 1.
The final value is t_final(i,j) = t_meteo(i,j) + Gamma_ij.

7) Writes a .vrp file in VRPLIB format with
EDGE_WEIGHT_TYPE : EXPLICIT and EDGE_WEIGHT_FORMAT : FULL_MATRIX,
where the EDGE_WEIGHT_SECTION contains the full N x N matrix t_final.

8) Saves a CSV file with various summary statistics.

"""


#--- EXAMPLE TO LAUNCH THE SCRIPT: 
# python Name_of_Script.py 30 1 2 6 5 rain 30_0001

import sys
import random
import math
import csv
import os
import numpy as np
import pandas as pd
import hashlib
import numpy as np
from scipy.sparse.csgraph import shortest_path

#--- Ensure output files are saved in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import argparse

from pathlib import Path   

import argparse

#---------------------------------------------#
#--- CLI parser for command-line arguments ---#
#---------------------------------------------#
def parse_cli():
    """
    Parser of Command line arguments

   Positional parameters:
      1. n              → number of customers
      2. rootPos        → depot position (1-Random, 2-Centro, 3-Angolo)
      3. custPos        → layout of customers (1-Random, 2-Cluster, 3-Mix random+cluster)
      4. demandType     → type of demand(1..7 come nel paper di Uchoa)
      5. avgRouteSize   → Average route length class (1..6)
      6. weather_condition → 'none', 'rain', 'snow', 'fog'
      7. instanceID     → instance ID(es. 30_0001)

    Optional flags
      --debug_plots     → if present, it saves histograms(PNG) of factors (c, w, γ, t/dist).
    """
    parser = argparse.ArgumentParser(
        description="Generatore di istanze CVRP con travel times"
    )

    parser.add_argument("n", type=int, help="Numero di clienti")

    parser.add_argument(
        "rootPos",
        type=int,
        choices=[1, 2, 3],
        help="Posizione del deposito (1-Random, 2-Centro, 3-Angolo)",
    )

    parser.add_argument(
        "custPos",
        type=int,
        choices=[1, 2, 3],
        help="Layout clienti (1-Random, 2-Cluster, 3-Mix random+cluster)",
    )

    parser.add_argument(
        "demandType",
        type=int,
        choices=list(range(1, 8)),
        help="Tipo di domanda (1..7, come nel paper di Uchoa)",
    )

    parser.add_argument(
        "avgRouteSize",
        type=int,
        choices=list(range(1, 7)),
        help="Classe di lunghezza media delle rotte (1..6)",
    )

    parser.add_argument(
        "weather_condition",
        choices=["none", "rain", "snow", "fog"],
        help="Condizioni meteo ('none', 'rain', 'snow', 'fog')",
    )

    parser.add_argument(
        "--debug_plots",
        action="store_true",
        help="Se presente, salva istogrammi (PNG) dei fattori c, w, γ e del rapporto t_ij/dist_ij",
    )

    parser.add_argument(
        "instanceID",
        type=str,
        help="ID istanza (es. 30_0001)",
    )

    return parser.parse_args()


#------------------------------------------------------------------------------#
#----------- Initial parameters (for now from CLI) ----------------------------#
#------------------------------------------------------------------------------#

lambda_cluster = 1
lambda_random  = 4
lambda_random_cluster = 2.5

T1 = 300
T2 = 700
beta = 1

maxCoord = 1000  # grid dimension
decay = 40       # for cluster generation (exponential decay)

#---------------------------------------------------------------------------------#
#----------- Original Uchoa functions + helper for congestion, meteo, etc --------#
#---------------------------------------------------------------------------------#

def distance(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

#--------------------------------------------------------------------------------------#
#--- Random generation of depot (root positioning) based on rootPos values (3 cases)---#
#--------------------------------------------------------------------------------------#
def generate_depot(rootPos, maxCoord):
    #--- 1: Random position in a casual point in the grid.
    if rootPos == 1:  # Random
        x_ = random.randint(0, maxCoord)
        y_ = random.randint(0, maxCoord)
    #--- 2: position at the center of the grid
    elif rootPos == 2:  # Center
        x_ = y_ = int(maxCoord/2)
    #--- 3: position at corner (0,0), upper left => coordinates x_, y_ equal to 0
    elif rootPos == 3:  # Corner (0,0)
        x_ = y_ = 0
    else:
        raise ValueError("Depot Positioning out of range!")
    return (x_, y_)

#----------------------------------------------------------------------------------#
#--- Customers generation - based on Uchoa Algorithms - Accept-Reject mechanism ---#
#----------------------------------------------------------------------------------#
def generate_customer_positions(n, custPos, depot, nSeeds, maxCoord, decay):
    S = set() # Set of coordinates for the customers
    seeds = [] # seeds for clusters

    if custPos == 3:
         nRandCust = int(n/2)
    elif custPos == 2:
         nRandCust = 0
    elif custPos == 1:
        nRandCust = n
        nSeeds = 0   
    else:
        raise ValueError("Customer Positioning out of range!")

    nClustCust = n - nRandCust

    #----------------------------#
    #--- CUSTOMERS GENERATION ---#
    #----------------------------#

    for _ in range(nRandCust):
        x_ = random.randint(0, maxCoord)
        y_ = random.randint(0, maxCoord)
        
        while (x_, y_) in S or (x_, y_) == depot:
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)
        S.add((x_, y_))
    nS = nRandCust

    if nClustCust > 0:
        if nClustCust < nSeeds:
            raise ValueError("Too many seeds for clusterization!")
        
        for _ in range(nSeeds):
            
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)

            while (x_, y_) in S or (x_, y_) == depot:
                x_ = random.randint(0, maxCoord)
                y_ = random.randint(0, maxCoord)
            
            S.add((x_, y_))
            seeds.append((x_, y_))
        nS += nSeeds

        maxWeight = 0.0
        
        for (i, j) in seeds:
            w_ij = 0.0

            for (ip, jp) in seeds:
                w_ij += 2**(-distance((i, j), (ip, jp))/decay)
            if w_ij > maxWeight:
                maxWeight = w_ij
                maxWeight = w_ij
        
        norm_factor = 1.0 / maxWeight

        while nS < n:
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)
            while (x_, y_) in S or (x_, y_) == depot:
                x_ = random.randint(0, maxCoord)
                y_ = random.randint(0, maxCoord)
            # initialize weight to zero
            weight = 0.0
            # Loop over all pairs of seed coordinates, and for each pair of coordinates
            # sum the influences of all seeds on (x_, y_). The exponential function
            # assigns a higher weight to points close to the seeds.
            #  - If (x_, y_) is very close to a seed, its weight will be high.
            #  - If (x_, y_) is far from all seeds, its weight will be low.
            for (ip, jp) in seeds:
                weight += 2**(-distance((x_, y_), (ip, jp))/decay)
            # Normalize the weight into a range between 0 and 1 using the previously calculated norm_factor.
            # Therefore, Weight will be at most 1.
            weight *= norm_factor
            #------------------------------------------#
            
            if random.uniform(0,1) <= weight:
                S.add((x_, y_))
                nS += 1
    
    customers_list = sorted(list(S)) 
    random.shuffle(customers_list)
    
    V = [depot] + customers_list
    return V, seeds

#-----------------------------------------------------#
#--- Demand generation - based on Uchoa Algorithms ---#
#-----------------------------------------------------#
def generate_demands(V, demandType, r, n, maxCoord=1000):
    demandMinValues = [1, 1, 5, 1, 50, 1, 51, 50, 1]
    demandMaxValues = [1, 10, 10, 100, 100, 50, 100, 100, 10]
    demandMin = demandMinValues[demandType - 1]
    demandMax = demandMaxValues[demandType - 1]
    demandMinEvenQuadrant = 51
    demandMaxEvenQuadrant = 100
    demandMinLarge = 50
    demandMaxLarge = 100
    largePerRoute = 1.5
    demandMinSmall = 1
    demandMaxSmall = 10

    D = []
    sumDemands = 0
    maxDemand = 0
    
    for i in range(2, n+2):
        j = int((demandMax - demandMin + 1)*random.uniform(0,1) + demandMin)

        if demandType == 6:  # Q
            x_, y_ = V[i-1]
            if (x_ < maxCoord/2 and y_ < maxCoord/2) or (x_>=maxCoord/2 and y_>=maxCoord/2):
                j = int((demandMaxEvenQuadrant - demandMinEvenQuadrant + 1)*random.uniform(0,1) + demandMinEvenQuadrant)

        if demandType == 7:  # Msfl (70-95% small, rest large) – here we use the logic "for i < (n/r)*largePerRoute"
            if i < (n/r)*largePerRoute:
                j = int((demandMaxLarge - demandMinLarge +1)*random.uniform(0,1) + demandMinLarge)
            else:
                j = int((demandMaxSmall - demandMinSmall +1)*random.uniform(0,1) + demandMinSmall)

        D.append(j)
        maxDemand = max(maxDemand, j)
        sumDemands += j

    return D, sumDemands, maxDemand

#-------------------------------------------------------#
#--- Capacity generation - based on Uchoa Algorithms ---#
#-------------------------------------------------------#
def compute_capacity(sumDemands, maxDemand, r, n):
    if sumDemands == n:
        capacity = math.floor(r)
    else:
        capacity = max(maxDemand, math.ceil(r * sumDemands / n))
    return capacity

#----------------------------------------------------------------------------#
#--------- New Functions added by me to calculate the travel times matrix ---#
#----------------------------------------------------------------------------#

def truncated_exponential_translated(lambda_, cutoff=1.5):
    while True:
        f_x_lambda_sample = random.expovariate(lambda_)
        if f_x_lambda_sample <= cutoff:  # cutoff (1.5)
            return 1.0 + f_x_lambda_sample
        # altrimenti rigenero

#--- Function to correct triangle inequality using Floyd-Warshall algorithm
def ensure_triangle_inequality(matrix):
    
    mat_array = np.array(matrix)
    corrected_matrix = shortest_path(mat_array, method='FW', directed=False, unweighted=False)
    
    return corrected_matrix.tolist()

def compute_travel_time_matrix(V, layout, weather_condition, lambda_cluster, lambda_random,
                               T1, T2, beta):
    N = len(V)
    distMat  = [[0.0] * N for _ in range(N)]
    alphaMat = [[0.0] * N for _ in range(N)]
    cMat     = [[0.0] * N for _ in range(N)]
    wMat     = [[0.0] * N for _ in range(N)]
    gammaMat = [[0.0] * N for _ in range(N)]
    tFinal   = [[0.0] * N for _ in range(N)]

    if layout == 'cluster':
        lambda_ = lambda_cluster
    elif layout == 'random':
        lambda_ = lambda_random
    else:  # 'random-cluster'
        lambda_ = lambda_random_cluster

    for i in range(N):
        for j in range(i, N):
            if i == j:
                distMat[i][j] = 0.0
            else:
                d = distance(V[i], V[j])
                distMat[i][j] = distMat[j][i] = d

    if weather_condition == 'none':
        w_global = 1.0
    else:
        if weather_condition == 'rain':
            base_min, base_max = 1.1, 1.2
        elif weather_condition == 'snow':
            base_min, base_max = 1.3, 1.7
        elif weather_condition == 'fog':
            base_min, base_max = 1.2, 1.4
        else:
            base_min, base_max = 1.0, 1.0

        if layout == 'cluster':
            layout_factor = 1.0
        elif layout == 'random-cluster':
            layout_factor = 0.7
        else:  # 'random'
            layout_factor = 0.4

        delta_min = base_min - 1.0
        delta_max = base_max - 1.0
        w_min = 1.0 + layout_factor * delta_min
        w_max = 1.0 + layout_factor * delta_max

        w_global = random.uniform(w_min, w_max)

    for i in range(N):
        for j in range(i, N):
            if i == j:
                alphaMat[i][j] = 1.0
                cMat[i][j]     = 1.0
                wMat[i][j]     = 1.0
                gammaMat[i][j] = 0.0
                tFinal[i][j]   = 0.0
                continue

            d = distMat[i][j]

            alpha_ij = random.uniform(0.8, 1.2)
            alphaMat[i][j] = alphaMat[j][i] = alpha_ij
            tij_base = alpha_ij * d

            c_ij = truncated_exponential_translated(lambda_)
            cMat[i][j] = cMat[j][i] = c_ij
            tij_cong = tij_base * c_ij

            wMat[i][j] = wMat[j][i] = w_global
            tij_meteo = tij_cong * w_global

            if d <= T1:
                gamma_ij = random.uniform(0, 10)
            elif d <= T2:
                gamma_ij = random.uniform(5, 30)
            else:
                gamma_ij = random.uniform(15, 60)
            gammaMat[i][j] = gammaMat[j][i] = gamma_ij

            tij_tot = tij_meteo + beta * gamma_ij

           
            tij_rounded = round(tij_tot, 2)
            tFinal[i][j] = tFinal[j][i] = tij_rounded
 
     
    tFinal = ensure_triangle_inequality(tFinal)

    return tFinal, distMat, alphaMat, cMat, wMat, gammaMat


#----------------------------#
#--- WRITE.VRP FINAL FILE ---#
#----------------------------#
#--- format: CVRPLIB
def write_full_matrix_vrp_file(filename, instanceName, commentLine, N, capacity, V, D, tFinal):
    
    with open(filename, 'w') as f:
        f.write("NAME: " + instanceName + "\n")
        f.write("COMMENT: " + commentLine + "\n")
        f.write("TYPE: CVRP\n")
        f.write("DIMENSION: " + str(N) + "\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("CAPACITY: " + str(int(capacity)) + "\n")

        f.write("NODE_COORD_SECTION\n")
        for i, (xx,yy) in enumerate(V):
            f.write(f"{i+1} {xx} {yy}\n")

        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(N):
            row_str = []
            for j in range(N):
                cost_ij = tFinal[i][j]
                row_str.append(f"{cost_ij:.2f}")
            f.write(" ".join(row_str) + "\n")

        # Demand
        f.write("DEMAND_SECTION\n")
        D_ = [0] + D  # 0 per il depot
        for i in range(N):
            f.write(f"{i+1} {D_[i]}\n")

        # Depot
        f.write("DEPOT_SECTION\n1\n-1\n")
        f.write("EOF\n")

#--------------------------------#
#--- SAVE CSV WITH STATISTICS ---#
#--------------------------------#
def write_descriptive_statistics(csvfilename, V, D, capacity,
                                 distMat, tFinal, alphaMat, cMat, wMat, gammaMat,
                                 layout, weather_condition,
                                 lambda_cluster, lambda_random, lambda_random_cluster,
                                 T1, T2, beta, r, instanceName,
                                 summary_filename=None, debug_plots=False):
    
    N = len(V)
    n = N - 1   

    all_dist = []  
    all_tt = []  
    all_alpha = []  
    all_c = []  
    all_w = []  
    all_gamma = []  
    all_ratio = []   

     
    for i in range(N):
        for j in range(i + 1, N):
            d = distMat[i][j]
            tij = tFinal[i][j]
            if d <= 0:
                continue
            all_dist.append(d)
            all_tt.append(tij)
            all_alpha.append(alphaMat[i][j])
            all_c.append(cMat[i][j])
            all_w.append(wMat[i][j])
            all_gamma.append(gammaMat[i][j])
            all_ratio.append(tij / d)

    def _basic_stats(values):
        if not values:
            return dict(min=0.0, max=0.0, mean=0.0, std=0.0)
        arr = np.array(values, dtype=float)
        return dict(
            min=float(arr.min()),
            max=float(arr.max()),
            mean=float(arr.mean()),
            std=float(arr.std()),
        )

    dist_stats  = _basic_stats(all_dist)
    tt_stats    = _basic_stats(all_tt)
    alpha_stats = _basic_stats(all_alpha)
    c_stats     = _basic_stats(all_c)
    w_stats     = _basic_stats(all_w)
    gamma_stats = _basic_stats(all_gamma)
    ratio_stats = _basic_stats(all_ratio)

    total_demand = float(sum(D))
    demand_mean  = float(np.mean(D)) if len(D) > 0 else 0.0
    demand_std   = float(np.std(D))  if len(D) > 0 else 0.0
    # Estimated Routes (Lower Bound)
    estimated_routes = total_demand / capacity if capacity > 0 else 0.0

    
    c_over_13 = 100.0 * sum(1 for c in all_c if c > 1.3) / len(all_c) if all_c else 0.0
    c_over_16 = 100.0 * sum(1 for c in all_c if c > 1.6) / len(all_c) if all_c else 0.0

   
    rows = []
    rows.append(["SECTION", "METRIC", "VALUE"])

    # Generale
    rows.append(["GENERAL", "instance_name", instanceName])
    rows.append(["GENERAL", "layout", layout])
    rows.append(["GENERAL", "weather_condition", weather_condition])
    rows.append(["GENERAL", "n_customers", n])
    rows.append(["GENERAL", "capacity", capacity])
    rows.append(["GENERAL", "lambda_cluster", lambda_cluster])
    rows.append(["GENERAL", "lambda_random", lambda_random])
    rows.append(["GENERAL", "lambda_random_cluster", lambda_random_cluster])
    rows.append(["GENERAL", "T1", T1])
    rows.append(["GENERAL", "T2", T2])
    rows.append(["GENERAL", "beta", beta])
    rows.append(["GENERAL", "seed_r", r])

    # Domanda
    rows.append(["DEMAND", "total_demand", total_demand])
    rows.append(["DEMAND", "demand_mean", demand_mean])
    rows.append(["DEMAND", "demand_std", demand_std])
    rows.append(["DEMAND", "estimated_routes", estimated_routes])

    # Distanze
    for k, v in dist_stats.items():
        rows.append(["DIST", f"dist_{k}", v])

    # Tempi di viaggio
    for k, v in tt_stats.items():
        rows.append(["TRAVEL_TIME", f"tt_{k}", v])

    # Fattori
    for prefix, stats in [
        ("alpha", alpha_stats),
        ("c", c_stats),
        ("w", w_stats),
        ("gamma", gamma_stats),
    ]:
        for k, v in stats.items():
            rows.append(["FACTOR", f"{prefix}_{k}", v])

    for k, v in ratio_stats.items():
        rows.append(["RATIO_TT_DIST", f"ratio_{k}", v])
     
    rows.append(["CONGESTION", "c>1.3_pct", c_over_13])
    rows.append(["CONGESTION", "c>1.6_pct", c_over_16])

    with open(csvfilename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    if summary_filename is not None:
        summary = {
            "instance": instanceName,
            "n_customers": n,
            "layout": layout,
            "weather": weather_condition,
            "capacity": capacity,
            "total_demand": total_demand,
            "estimated_routes": estimated_routes,
            "lambda_cluster": lambda_cluster,
            "lambda_random": lambda_random,
            "lambda_random_cluster": lambda_random_cluster,
            "T1": T1,
            "T2": T2,
            "beta": beta,
            "dist_mean": dist_stats["mean"],
            "dist_std": dist_stats["std"],
            "tt_mean": tt_stats["mean"],
            "tt_std": tt_stats["std"],
            "c_mean": c_stats["mean"],
            "c_std": c_stats["std"],
            "w_mean": w_stats["mean"],
            "w_std": w_stats["std"],
            "gamma_mean": gamma_stats["mean"],
            "gamma_std": gamma_stats["std"],
            "ratio_mean": ratio_stats["mean"],
            "ratio_std": ratio_stats["std"],
            "c_over_13_pct": c_over_13,
            "c_over_16_pct": c_over_16,
        }
        file_exists = os.path.exists(summary_filename)
        with open(summary_filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)

    if debug_plots:
        base_no_ext, _ = os.path.splitext(csvfilename)

        def _save_hist(values, title, filename, xlabel):
            if not values:
                return
            plt.figure()
            plt.hist(values, bins=30)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("Frequenza")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        _save_hist(
            all_c,
            "Distribuzione fattore di congestione c_ij",
            base_no_ext + "_hist_c.png",
            "c_ij",
        )
        _save_hist(
            all_w,
            "Distribuzione fattore meteo w",
            base_no_ext + "_hist_w.png",
            "w",
        )
        _save_hist(
            all_gamma,
            "Distribuzione componente additiva γ_ij",
            base_no_ext + "_hist_gamma.png",
            "γ_ij",
        )
        _save_hist(
            all_ratio,
            "Rapporto tempi/distanze t_ij / dist_ij",
            base_no_ext + "_hist_ratio_tt_dist.png",
            "t_ij / dist_ij",
        )

    return rows

#--------------------------#
#--- VISUALIZE INSTANCE ---#
#--------------------------#
def plot_instance(V, seeds, instanceName):
    
    depot_x, depot_y = V[0]
    cust_coords = V[1:]
    cust_x = [c[0] for c in cust_coords]
    cust_y = [c[1] for c in cust_coords]
    seed_x = [s[0] for s in seeds] if seeds else []
    seed_y = [s[1] for s in seeds] if seeds else []

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150, layout="constrained")

    ax.scatter(
        cust_x,
        cust_y,
        label=f"Customers ({len(cust_coords)})",
        marker="o",
        facecolor="#1f77b4",
        edgecolor="black",
        s=30,
        linewidth=0.3,
    )

    if seeds:
        ax.scatter(
            seed_x,
            seed_y,
            label=f"Seed nodes ({len(seeds)})",
            marker="o",
            facecolor="#d62728",
            edgecolor="black",
            s=70,
            linewidth=0.4,
        )

    ax.scatter(
        [depot_x],
        [depot_y],
        label="Depot",
        marker="s",
        facecolor="#ffdd00",
        edgecolor="black",
        s=120,
        linewidth=0.6,
    )

    ax.set_aspect("equal", adjustable="box")
    padding = maxCoord * 0.05  # 5 % di padding
    ax.set_xlim(-padding, maxCoord + padding)
    ax.set_ylim(-padding, maxCoord + padding)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.7)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Instance: {os.path.basename(instanceName)}", fontweight="bold")
    ax.legend(frameon=True, loc="upper right")

    fig.savefig(instanceName, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return fig

#---------------------------------------------------------------------------#
#--------------------- MAIN SCRIPT -----------------------------------------#
#---------------------------------------------------------------------------#

def main():

    seed = int(hashlib.md5(instanceID.encode()).hexdigest(), 16) % 2**32
    random.seed(seed)  # <--- Uso la variabile 'seed', non 'instanceID'
    print(f"Seed utilizzato: {seed}") # Utile per debug

    nSeeds = random.randint(2,6)

   
    In = {
        1: (3,5),
        2: (5,8),
        3: (8,12),
        4: (12,16),
        5: (16,25),
        6: (25,50)
    }

    if avgRouteSize < 1 or avgRouteSize > 6:
      print("Average route size out of range! Must be in [1..6]")
      exit(0)

    r_low, r_high = In[avgRouteSize]
    r = random.uniform(r_low, r_high)

    instanceName = (
        f"LDG{n}_{rootPos}{custPos}{demandType}{avgRouteSize}_"
        f"{weather_condition}_{instanceID}"
    )

    depot = generate_depot(rootPos, maxCoord)

    V, seeds = generate_customer_positions(n, custPos, depot, nSeeds, maxCoord, decay)

    D, sumDemands, maxDemand = generate_demands(V, demandType, r, n, maxCoord)
    capacity = compute_capacity(sumDemands, maxDemand, r, n)

    commentLine = (
      f"CVRP with travel times. "
      f" rootPos={rootPos},custPos={custPos},demandType={demandType},avgRouteSizeIndex={avgRouteSize},weather={weather_condition},lambda_cluster={lambda_cluster},lambda_random={lambda_random}, T1={T1},T2={T2},beta={beta}, alpha(i,j) ~ U(0.8,1.2)"
    )

    # Layout = lo deduciamo da custPos, che è un parametro di ingresso
    if custPos==2:
        layout_str = 'cluster'
    elif custPos==1:
        layout_str = 'random'
    else:
        layout_str = 'random-cluster'

    tFinal, distMat, alphaMat, cMat, wMat, gammaMat = compute_travel_time_matrix(
        V, layout_str, weather_condition, lambda_cluster, lambda_random,
        T1, T2, beta
    )

    # ------------------------------------------------------------------
    # VALIDATE INSTANCE(fail-fast)
    # ------------------------------------------------------------------
    assert len(V) == 1 + n, "Numero di nodi incoerente"

    assert len({tuple(v) for v in V}) == len(V), "Coordinate duplicate rilevate"

    assert all(d > 0 for d in D), "Domande nulle o negative"
    assert max(D) <= capacity, "Domanda > capacità"

    for i in range(len(V)):
        for j in range(len(V)):
            assert tFinal[i][j] >= 0, "Tempo di viaggio negativo"
            assert abs(tFinal[i][j] - tFinal[j][i]) < 1e-6, "Matrice non simmetrica"

    # triangle inequality (spot-check)
    for i, j, k in [(0, len(V)//2, -1), (1, 2, 3)]:
        assert tFinal[i][j] <= tFinal[i][k] + tFinal[k][j] + 1e-6, "Violazione triangolo"

    
    base_dir  = Path(script_dir) / "dataset" / f"{n:03d}" / weather_condition
    inst_dir  = base_dir / instanceName
    inst_dir.mkdir(parents=True, exist_ok=True)

    filenameVRP  = inst_dir / f"{instanceName}.vrp"
    csvfilename  = inst_dir / f"{instanceName}_stats.csv"
    pngfilename  = inst_dir / f"{instanceName}.png"

    summary_filename = base_dir / "istanze_summary.csv"
    
    N = len(V)
    write_full_matrix_vrp_file(
        str(filenameVRP),
        instanceName,
        commentLine,
        N,
        capacity,
        V,
        D,
        tFinal
    )
    print(f"VRP file {filenameVRP} generato con FULL_MATRIX dei travel times.")

    write_descriptive_statistics(
        str(csvfilename),
        V,
        D,
        capacity,
        distMat,
        tFinal,
        alphaMat,
        cMat,
        wMat,
        gammaMat,
        layout_str, 
        weather_condition,
        lambda_cluster,
        lambda_random,
        lambda_random_cluster,
        T1,
        T2,
        beta,
        r,
        instanceName,
        summary_filename=str(summary_filename),
        debug_plots=debug_plots
    )

     
    fig = plot_instance(V, seeds, str(pngfilename))  
    plt.close(fig)

if __name__ == "__main__":
     
    args = parse_cli()
    n = args.n
    rootPos = args.rootPos
    custPos = args.custPos
    demandType = args.demandType
    avgRouteSize = args.avgRouteSize
    weather_condition = args.weather_condition
    instanceID = args.instanceID
    debug_plots = args.debug_plots    

    main()
 
