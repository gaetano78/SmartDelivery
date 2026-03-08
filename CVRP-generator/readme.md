# CVRP Instance Generator with Realistic Travel Times 🚚

This repository contains a Python-based generator for **Capacitated Vehicle Routing Problem (CVRP)** instances.
Unlike standard benchmarks that rely solely on Euclidean distances, this tool focuses on generating **realistic travel times** by introducing stochastic factors such as traffic congestion, weather conditions, and additive delays due to, e.g., traffic lights and parking maneuvers.

The customer distribution and demand generation logic are based on the work of **Uchoa et al. (2017)**, ensuring the instances remain structurally comparable to state-of-the-art benchmarks while adding a layer of temporal complexity.

## 🌟 Key Features

- **Customer Topology:** Generates customers using Random, Clustered, or Mixed layouts with configurable seeds.
- **Travel Time Matrices:** Calculates travel time matrices instead of simple distances.
- **Stochastic Factors:**
  - **Congestion:** Modeled via truncated shifted exponential distributions to simulate traffic.
  - **Weather:** Global factors for Rain, Snow, or Fog that scale travel times.
  - **Additive Delays:** Fixed time penalties representing intersections or parking maneuvers, proportional to arc length.
- **Data Integrity:** Automatically applies the **Floyd-Warshall algorithm** to ensure the triangular inequality holds for the generated travel time matrix.
- **VRPLIB Compatible:** Exports instances in the standard `.vrp` format (`EDGE_WEIGHT_FORMAT: FULL_MATRIX`).
- **Interactive Dashboard:** Available at https://gaetanoldg-cvrp-generator.hf.space/
- **Documentation:** A more detailed documentation is available in the [`Instance Generator script - Documentation.pdf`](./Instance%20Generator%20script%20-%20Documentation.pdf) file.
