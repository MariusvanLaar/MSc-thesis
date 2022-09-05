# MSc-thesis
Repo for my MSc Thesis on the learning performance and computational hardness of Variational Quantum Circuits.

The main_v*.py scripts contain the overall ML pipeline used to conduct the experiments. main.py can be used for local testing, the later versions are for batch experiments only. 

The datasets folder contains all data related items including the raw data files, preprocessing scripts, post-processed data files (in datsets/data_files, not visible). datafactory.py is the core dataloader class used for data loading in PyTorch, and dataclasses.py contain the dataset specific classes.

The models folder contains the code for the simulation of the quantum circuits. layers.py contains the classes for the various quantum gates used. base_model.py contains the core class representing the skeleton of a quantum circuit, whilst the models.py script contains all the classes that define explicitly the circuit architectures used in the experiments. 

old_mains contains older main scripts where each new version represents the addition of some major feature.  old_exps tracks batch and analysis scripts for preliminary experiments.
