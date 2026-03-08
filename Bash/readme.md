# Bash File Generator for T-CVRP Dataset Creation

This folder contains a Python script that generates a Bash file to automate the creation of a balanced set of 10,000 CVRP instances with travel time-based cost matrices.

## Usage

- The script assumes the CVRP instance generator is named `LDG_Cleaned_Generatore_istanze_CON_TRAVEL_TIMES.py`. Update this name in the script if your file is named differently.
- An example of a generated `.sh` file is included in this folder.
- Running the generated Bash file will invoke the generator script 10,000 times, producing 10,000 instances.
