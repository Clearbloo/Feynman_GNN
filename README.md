# Feynman_GNN
This is the code for https://arxiv.org/abs/2211.15348. For a complete list of references, see the paper. However, most of the code was inspired from:

* The tutorials from pytorch geometric
https://github.com/pyg-team/pytorch_geometric

* And deepfindr's gnn-project
https://github.com/deepfindr/gnn-project

This project contains 2 Jupyter notebooks:

* Dataset builder
* Feynman GNN

Run dataset builder to construct the datasets. Then run Feynman GNN to train the network and make predictions. 

The results of running the dataset builder are stored in "./data/raw" as csv files

---

# Dataset builder
Builds csv files containing the data stored as graphs. Currently builds 6 datsets. 3 QED datasets, 1 combined QED dataset, 1 QCD dataset and 1 combined dataset.

---

# Feynman GNN
Graph neural network architecture is listed in the paper. Trains on a GAT network. Need to choose the arguments for train_feyn(). Some initial values have been included for a working example. Also the hyperparameter example is working for the QED dataset. 
