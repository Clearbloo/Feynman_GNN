# Feynman_GNN
---
This is the code for https://arxiv.org/abs/2211.15348. For a complete list of references, see the paper. However, most of the code was inspired from:

* The tutorials from pytorch geometric
https://github.com/pyg-team/pytorch_geometric

* And deepfindr's gnn-project
https://github.com/deepfindr/gnn-project


# Problems
---
- Currently doing a big refactor that broke master. Will need to revert to 8c576d0ad6b097465caefe669af028f33a87fe00 for the last working copy
- requirments.txt isn't working for some torch wheels. If `pip install -r "requirements.txt" fails`, you may need to manually install the failing requirements

# Details
---
This project contains 2 Jupyter notebooks:

* Dataset builder
* Feynman GNN

These are the original code.

Run dataset builder to construct the datasets. Then run Feynman GNN to train the network and make predictions. 

The results of running the dataset builder are stored in "./data/raw" as csv files

The ongoing refactor will have a new README in the future

# Dataset builder
---
Builds csv files containing the data stored as graphs. Currently builds 6 datsets. 3 QED datasets, 1 combined QED dataset, 1 QCD dataset and 1 combined dataset.


# Feynman GNN
Graph neural network architecture is listed in the paper. Trains on a GAT network. Need to choose the arguments for train_feyn(). Some initial values have been included for a working example. Also the hyperparameter example is working for the QED dataset. 


# Branch notes
This branch contains the building out of more diagrams