# CosmoGNN
Estimation of cosmological parameter $\Omega_m$ from [Quijote simulations](https://quijote-simulations.readthedocs.io/en/latest/) using Graph Neural Networks.

<p align="center">
<img src="https://github.com/LauraRavagnani/CosmoGNN/assets/137277088/09bf91fb-1a7e-465a-a17b-23ba5f2db712" width=600>
</p>


## Data
The data used in this work can be retrieved from [globus](https://app.globus.org/file-manager?origin_id=e0eae0aa-5bca-11ea-9683-0e56c063f437&origin_path=%2F) following the path:
```console
Halos/FoF/latin_hypercube/
```
selecting for each simulation number, the folder with redshift $z = 1$:
```console
groups_002
```

## Requisites
The code runs on GPU. The one used for this work is a _Tesla T4_.

Libraries:
* ```numpy```
* ```pytorch```
* ```pytorch-geometric```
* ```matplotlib```
* ```scipy```
* ```sklearn```
* ```optuna``` 

## Scripts
* ```clean_main.py```: main driver to train and test the network

* ```clean_hyperparameters.py```: definition of the hyperparameters employed by the networks

* ```clean_gridsearch.py```: optimize the hyperparameters using optuna

* ```visualize_graphs.py```: display graphs of DM halos from the simulations

The folder Source contains:

* ```constants.py```: basic constants and initialization

* ```load_data.py```: routines to load data from simulation files

* ```plotting.py```: functions for displaying the results from the training and test

* ```metalayer.py```: definition of the Graph Neural Network architecture

* ```training.py```: routines for training and testing the network

## Authors and Acknowledgments
### Authors
* **Lorenzo Cavezza** - [Lorycav](https://github.com/Lorycav)
* **Giulia Doda** - [giuliadoda](https://github.com/giuliadoda)
* **Giacomo Longaroni** - [GiacomoLongaroni](https://github.com/GiacomoLongaroni)
* **Laura Ravagnani** - [LauraRavagnani](https://github.com/LauraRavagnani)

### Acknowledgments
This work is based on:

#### Reference Papers
<a id="1">[1]</a> 
Villanueva-Domingo, Pablo, and Francisco Villaescusa-Navarro. "Learning cosmology and clustering with cosmic graphs." The Astrophysical Journal 937.2 (2022): 115.

<a id="2">[2]</a>
Makinen, T. Lucas, et al. "The cosmic graph: Optimal information extraction from large-scale structure using catalogues." arXiv preprint arXiv:2207.05202 (2022).

#### Original Code
PabloVD, (2023). CosmoGraphNet: "Graph Neural Networks to predict the cosmological parameters or the galaxy power spectrum from galaxy catalogs". [GitHub](https://github.com/PabloVD/CosmoGraphNet.git)


