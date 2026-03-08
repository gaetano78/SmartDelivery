# **SmartDelivery**

Repository for the MSCA project SmartDelivery.

In the CVRP-solver folder you can find a Python script to solve a generic, travel time-based CVRP instance, along with a test instance to see how the script works.
A simple Streamlit-based dashboard is also available on Hugging Face Space (note: computational power is limited):

https://gaetanoldg-smartdelivery-cvrpsolver.hf.space/

In the CVRP-generator folder you can find a Python script to generate generic, travel time-based CVRP instances. Several parameters can be adjusted to produce different configurations. Generated instances can be solved using the script in the CVRP-solver folder.
A simple Streamlit-based dashboard for the generator is also available on Hugging Face Space (note: computational power is limited):

https://gaetanoldg-cvrp-generator.hf.space/

The repository also includes a dataset of 10,000 CVRP instances generated with the scripts above. The full dataset is available in the releases section, or can be downloaded directly here:
https://github.com/gaetano78/SmartDelivery/releases/download/T-CVRP_v1.0_dataset_directories/T-CVRP.v1.0_dataset_directories_structure.zip

The archive contains the dataset organized in a clean directory structure, with a statistics file accompanying each instance.
A flat version of the dataset (all .vrp files, no directory structure) is also available in the releases, or can be downloaded here:

https://github.com/gaetano78/SmartDelivery/releases/download/T-CVRP_V1.0_dataset_no_directories_no_statistics/dataset_no_directories_no_statistics.zip

The dataset is also available on Zenodo: https://zenodo.org/records/18415031

-------------------
Author: Gaetano Carmelo La Delfa

Role: Marie Skłodowska-Curie Postdoctoral Fellow (MSCA), Computer Engineering / Operations Research

Contact: [gaetano.ladelfa@usal.es]

Issues, questions, or suggestions are welcome.



