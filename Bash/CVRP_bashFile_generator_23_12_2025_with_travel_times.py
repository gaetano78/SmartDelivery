 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVRP Dataset Generator - BASH SCRIPT CREATOR
"""

import argparse
import os
import math
import random
from itertools import product
from typing import List, Sequence, Dict, Optional
from collections import Counter
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
 
#-----------------------#
#--- PARSE ARGUMENTS ---#
#-----------------------#


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a .sh file for creating travel-time-balanced CVRP instances')

    parser.add_argument('--total_instances', type=int, default=10000,
                        help='Total number of customers to generate (default: 10000)')
    
    parser.add_argument('--min_customers', type=int, default=30,
                        help='Minimum number of customers (default: 30, min: 30)')
    
    parser.add_argument('--max_customers', type=int, default=200,
                        help='Maximum number of customers (default: 200, max:300)') 
    
    parser.add_argument('--step_size', type=int, default=5,
                        help='Step size for increasing the number of customers (default: 5)')
    
    parser.add_argument('--output_file', type=str, default='genInstances.sh',
                        help='Name of the .sh output file (default: genInstances.sh)')
    
    parser.add_argument('--generator_path', type=str, default='LDG_Cleaned_Generatore_istanze_CON_TRAVEL_TIMES.py',
                        help='path to the instance generator script (default: LDG_Cleaned_Generatore_istanze_CON_TRAVEL_TIMES.py)')
    
    parser.add_argument('--python_cmd', type=str, default='python3',
                        help='Python interpreter to be used in bash (default: python3)')

    parser.add_argument('--small_weight', type=float, default=0.3,
                        help='Weight for small instances [30-80 customers] (default: 0.3)')
    parser.add_argument('--medium_weight', type=float, default=0.4,
                        help='Weight for average instances [85-140 customers] (default: 0.4)')
    parser.add_argument('--large_weight', type=float, default=0.3,
                        help='Weight for big instances [145-200 customers] (default: 0.3)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Seed for random generation (default: 42)')
        
    args = parser.parse_args()

    #-------------------#
    #--- BASIC CHECKS---#
    #-------------------#
    
    if abs(args.small_weight + args.medium_weight + args.large_weight - 1.0) > 0.001:
        parser.error("La sum of weights (small_weight + medium_weight + large_weight) must be 1.0")
    
    if args.min_customers >= args.max_customers:
        parser.error("min_customers must be < max_customers")

    if args.min_customers<30 or args.max_customers>300 :
        parser.error("min_customers must be > 30, max_customers must be < 300")

    if args.step_size <= 0:
        parser.error("step_size must be > 0")
    
    if args.total_instances <= 0:
        parser.error("total_instances must be > 0")
    
    return args

#----------------------------------------------#
#--- FUNCTION TO CREATE GROUPS OF CUSTOMERS ---#
#----------------------------------------------#

#--- INPUT = min and max number of customers (30 e 200) and step_size, default equal to 5 
def create_customer_size_groups(min_customers, max_customers, step_size):
    
    #--- 1) Generate all possible values of n (number of customers), from min_customers to max_customers, at step of step_size. customer_sizes will be:
    #       [30, 35, 40, 45, 50, 55, 60, ... 175, 180, 185, 190, 195, 200]; 
    customer_sizes = list(range(min_customers, max_customers + 1, step_size))

    # split into 3 groups
    #---    [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80];
    small_sizes = [n for n in customer_sizes if n <= 80]
    
    #--- [85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140];
    medium_sizes = [n for n in customer_sizes if 85 <= n <= 140]
    
    #--- [145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200].
    large_sizes = [n for n in customer_sizes if n >= 145]
    
    print("\n==========  SMALL, MEDIUM E LARGE SIZE  ==========")
    print("\ncustomer_sizes:", customer_sizes)
    print("\nsmall_sizes", small_sizes)
    print("\nmedium_sizes", medium_sizes)
    print("\nlarge_sizes", large_sizes)
    print("\n==================================================")

    return customer_sizes, small_sizes, medium_sizes, large_sizes

#---------------------#
#--- MAIN FUNCTION ---#
#---------------------#

#-------------------------------------------------------------------------------------------------------------#
#--- Algorithm: Hamilton – Hare–Niemeyer “largest-remainder” apportionment + balanced randomized rounding. ---#
#-------------------------------------------------------------------------------------------------------------#

def allocate_instances_to_sizes(
    total_instances: int,
    small_weight: float,
    medium_weight: float,
    large_weight: float,
  
    small_sizes: Sequence[int],
    medium_sizes: Sequence[int],
    large_sizes: Sequence[int],
    *,
    random_seed: Optional[int] = None
) -> Dict[int, int]:
    
    #----------------------#
    #--- Simple checks ---#
    #----------------------#
    if total_instances <= 0:
        raise ValueError("total_instances must be > 0.")
    weights = [small_weight, medium_weight, large_weight]
    
    if any(w < 0 for w in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError("Weights must be non-negative and the sum must be 1.0.")
    all_sizes = [set(small_sizes), set(medium_sizes), set(large_sizes)]
    
    if any(all_sizes[i] & all_sizes[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("small_sizes, medium_sizes e large_sizes must not be overlapped")
    
    rng = random.Random(random_seed)

    target_counts = [round(total_instances * w) for w in weights]

    diff = total_instances - sum(target_counts)
    if diff: 
        
        fracs = [total_instances * w - tc for w, tc in zip(weights, target_counts)]
        
        order = sorted(range(3), key=lambda i: fracs[i], reverse=(diff > 0))
        idx = 0
        while diff: # sarebbe diff is not zero
            target_counts[order[idx % 3]] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            idx += 1

    size_to_instances: Dict[int, int] = {}
    for sizes, group_total in zip(
        (small_sizes, medium_sizes, large_sizes), target_counts
    ):
        q, r = divmod(group_total, len(sizes))   # q istanze di base, r resti
        for s in sizes:
            size_to_instances[s] = q
        for s in rng.sample(list(sizes), r):
            size_to_instances[s] += 1
    return size_to_instances


#-----------------------------------------------------------#
#--- IS INSTANCE PHYSICALLY VALID? - FILTER ----------------#
#-----------------------------------------------------------#
def is_valid_combination(size_n: int, avg_route_idx: int) -> bool:
    
    if avg_route_idx == 6: # Ultra Long
        if size_n < 60:
            return False
    elif avg_route_idx == 5: # Very Long
        if size_n < 40:
            return False
    return True

#-------------------------------#
#--- GENERATE BASH FILE ROWS ---#
#-------------------------------#

def generate_commands(
    size_to_instances: Dict[int, int],
    generator_path: str,
    *,
    random_seed: Optional[int] = None,
    python_cmd: str = "python3"
) -> List[str]:
    
    # -------------------- PARAMETER SPACE ---------------------------
    depot_positions       = [1, 2, 3]                     # R, C, E  - Random, Centered, Eccentric
    customer_distributions= [1, 2, 3]                     # R, C, RC - Random, Clustered, Random-clustered
    demand_types          = [1, 2, 3, 4, 5, 6, 7]         # 7 different type of demand
    avg_route_sizes       = [1, 2, 3, 4, 5, 6]            # 6 different average route sizes
    weather_conditions    = ["none", "rain", "snow", "fog"] # 4 different weather conditions

    partial_params = list(product(
        depot_positions,
        customer_distributions,
        demand_types,
        weather_conditions
    ))

    rng = random.Random(random_seed)
    commands: List[str] = []

    # -------------------- cycle for each size (30, 35, etc.) ---------------------------
    for size in sorted(size_to_instances):        
        n_needed = size_to_instances[size] 
        
        if n_needed == 0: continue

        valid_combinations = []
        for r_size in avg_route_sizes:
            if not is_valid_combination(size, r_size):
                continue
            
            for p in partial_params:

                full_tuple = (p[0], p[1], p[2], r_size, p[3])
                valid_combinations.append(full_tuple)

        # Sampling
        if n_needed <= len(valid_combinations):
            sample = rng.sample(valid_combinations, n_needed)
        else:
            sample = list(valid_combinations)
            remainder = n_needed - len(valid_combinations)
            sample.extend(rng.choices(valid_combinations, k=remainder))

        # String commands generation
        for idx, (depot_pos, cust_pos, demand_type, avg_route_size, weather) in enumerate(sample, start=1):
            # id stabile:  e.g. 30_0001, 30_0002 …
            inst_id = f"{size}_{idx:04d}"
            
            cmd = (
                f"{python_cmd} {generator_path} "
                f"{size} {depot_pos} {cust_pos} {demand_type} "
                f"{avg_route_size} {weather} {inst_id}"
            )
            commands.append(cmd)

    return commands

#--------------------------#
#--- BASH FILE CREATION ---#
#--------------------------#

def create_bash_script(commands, output_file):
    
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        # FIX: Header robusto
        f.write("set -euo pipefail\n\n")
        f.write("# Automatically generated script for the generation of balanced CVRP instances with travel times\n")
        f.write("# NOTE: It includes filter for physical coherence (N<60 no UltraLong, N<40 no VeryLong)\n")
        f.write(f"# Total istances: {len(commands)}\n\n")
        for command in commands:
            f.write(f"{command}\n")
    
    os.chmod(output_file, 0o755)

    print(f"\nBash script successfully generated generated: {output_file}")
    print(f"\nTotal instances to generate: {len(commands)}")

#-----------------------#
#--- GENERATE REPORT ---#
#-----------------------#
def generate_report(args,
                    size_to_instances: Dict[int, int],
                    commands: List[str],
                    output_file_base: str,
                    small_sizes, medium_sizes, large_sizes) -> None:
   
    def pct(x: int, total: int) -> str:
        return f"{100 * x / total:5.1f}%" if total > 0 else "0.0%"

    report_file = f"{output_file_base.split('.')[0]}_report.txt"
    total_cmds = len(commands)
    # Counters
    route_c = Counter()  
    depot_c = Counter()  
    weather_c = Counter()  
    wrong_filter_count = 0 # Instances that violate the filter to avoid bad instances
    group_counts = {"Small (30-80)": 0, "Medium (85-140)": 0, "Large (145-200)": 0}

    # Parse commands and count
    for line in commands:
        parts = line.strip().split()
        if len(parts) < 8: continue
        
        size    = int(parts[2])
        depot   = int(parts[3])
        route   = int(parts[6])
        weather = parts[7]

        route_c[route] += 1
        depot_c[depot] += 1
        weather_c[weather] += 1
        
        if size in small_sizes: group_counts["Small (30-80)"] += 1
        elif size in medium_sizes: group_counts["Medium (85-140)"] += 1
        else: group_counts["Large (145-200)"] += 1

        if not is_valid_combination(size, route):
            wrong_filter_count += 1

    # BUILDING THE REPORT - UNIQUE LINE
    lines = []
    lines.append("=" * 60)
    lines.append(f"{'DATASET STATISTICS T-CVRP 1.0':^60}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("► CONFIGURATION")
    lines.append(f"  Total instances requested : {args.total_instances}")
    lines.append(f"  Total instances generated : {total_cmds}")
    lines.append(f"  Customers Range           : {args.min_customers} – {args.max_customers}")
    lines.append(f"  Random seed               : {args.random_seed}")
    lines.append(f"  Python interpreter        : {args.python_cmd}")
    lines.append(f"  Anti-Degeneration filter : Active (N<60 no UltraLong)")
    lines.append("")
    
    lines.append("► FILTER CHECK")
    if wrong_filter_count == 0:
        lines.append("  [OK] No degenerated instance (TSP-like) detected.")
    else:
        lines.append(f"  [ERROR] Found {wrong_filter_count} instances which violate the filter condition!")
    lines.append("")

    lines.append("► DISTRIBUTIONS (Target: 30% - 40% - 30%)")
    keys_order = ["Small (30-80)", "Medium (85-140)", "Large (145-200)"]
    for k in keys_order:
        v = group_counts[k]
        lines.append(f"  {k:<20}: {v:5} ({pct(v, total_cmds)})")
    lines.append("")

    lines.append("► PARAMETER DISTRIBUTION")
    
    lines.append("  Average Route Size:")
    mapping_route = {1: "Very short", 2: "Short", 3: "Medium", 4: "Long",
                     5: "Very long", 6: "Ultra long"}
    for k in sorted(mapping_route):
        val = route_c[k]
        lines.append(f"    {mapping_route[k]:<15}: {val:5} ({pct(val, total_cmds)})")
    
    lines.append("\n Depot position:")
    mapping_depot = {1: "Random", 2: "Central", 3: "Eccentric"}
    for k in sorted(mapping_depot):
        val = depot_c[k]
        lines.append(f"    {mapping_depot[k]:<15}: {val:5} ({pct(val, total_cmds)})")
        
    lines.append("\n  Weather:")
    for k in sorted(weather_c):
        val = weather_c[k]
        lines.append(f"    {k:<15}: {val:5} ({pct(val, total_cmds)})")
    
    lines.append("")
    lines.append("=" * 60)

    # UNIFY ROWS
    report_content = "\n".join(lines)

    # 1. PRINT ON CONSOLE
    print(report_content)

    # 2. WRITE ON FILE
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n✔ Generated Bash Script: {args.output_file}")
    print(f"✔ Report saved in : {report_file}")

def main():
    args = parse_arguments()
    random.seed(args.random_seed)
    
    # Create groups: customer_sizes, small_sizes (s_s), medium_sizes (s_m), large_sizes (s_l).
    customer_sizes, s_s, s_m, s_l = create_customer_size_groups(
        args.min_customers, args.max_customers, args.step_size
    )
    
    size_to_instances = allocate_instances_to_sizes(
        args.total_instances, args.small_weight, args.medium_weight, args.large_weight,
        s_s, s_m, s_l, random_seed=args.random_seed
    )
    
    # Generate commmands
    commands = generate_commands(
        size_to_instances, 
        args.generator_path, 
        random_seed=args.random_seed,
        python_cmd=args.python_cmd
    )
    
    # Create bash
    create_bash_script(commands, args.output_file)
    
    # Generate report 
    generate_report(args, size_to_instances, commands, args.output_file, s_s, s_m, s_l)

if __name__ == "__main__":
    main()
