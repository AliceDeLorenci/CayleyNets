# CayleyNets

Project developed for the Geometrical Data Analysis course of the MVA Master (ENS Paris-Saclay) by:
- Alice Valença De Lorenci
- Julián Avalarez de Giorgi

The goal of the project was to study the following paper: *RonLevie,FedericoMonti,XavierBresson,andMichaelM.Bronstein. 2017. CayleyNets: Graph Convolutional Neural Networks with Complex Rational Spectral Filters. CoRR abs/1705.07664 (2017). arXiv:1705.07664 http://arxiv.org/abs/1705.07664*

The repository contains the source code associated with the project, notably, we propose a Pytorch Geometric implementation of Cayley convolutional layers, the building block of CayleyNets. 

The following package versions were used:
```
numpy 1.26.2
torch 2.0.1+cu117
torch_geometric 2.4.0
matplotlib 3.5.1
scipy 1.8.0
sknetwork 0.31.0
pickle 4.0
```

This repository is organized as follows:
- *src/CayleyNet.py*: implementation of Cayley convolutional layers and CayleyNets
- *src/CayleyTransform.py*: implementation of the Cayley transform
- *src/ChebNet.py*: implementation of ChebNets
- *src/CommunitiesGraph.py*: communities graph handler
- *src/CORA.py*: CORA dataset handler
- *src/Dataset.py*: generic dataset handler
- *src/utils.py*: miscellaneous helper methods used to train and evaluate the models
- *CayleyTransform.ipynb*: experiments around the Cayley transform
- *Experiments.ipynb*: experiments with Spectral GNNs
