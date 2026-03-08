import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch import nn
from torch.nn import Linear, BatchNorm1d, functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.loader import DataLoader

from feynman_gnn.dataset_builder import QED_dataset_builder

from pytorch_geometric import nn as geom_nn
from pytorch_geometric import TopKPooling
from pathlib import Path

import math

CHECKPOINT_PATH = Path("/content/gdrive/MyDrive/Part_III_Project/saved_models")
# layer name dictionary,
gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
    "NNConv": geom_nn.NNConv,
    "RGCN": geom_nn.RGCNConv,
    "Trans": geom_nn.TransformerConv,
}
# Hyperparameters to use if not tuning",
HYPERPARAMETERS = {
    "model_batch_size": [80],
    "model_weight_decay": [0.000001],
    "model_learning_rate": [8.128305161640993e-10],
    "model_embedding_size": [2],
    "model_attention_heads": [2],
    "model_layers": [2],
    "model_dropout_rate": [0.7],
    "model_top_k_ratio": [0.6],
    "model_top_k_every_n": [1],
    "model_dense_neurons": [6],
    "model_edge_dim": [11],
    "model_lin_dropout_prob": [0.8],
}


class MyLoss(nn.Loss):
    pass


class FeynmanModel(LightningModule):
    def __init__(self, c_in, c_out, layer_name, model_params, filename="QED_data.csv"):
        """
        c_in = channels in (feature dimensions, e.g. RGB is 3)
        c_out = channels out (target dimension, e.g. classification is 1)
        """
        super().__init__()
        self.filename = filename
        self.batch_size = model_params["model_batch_size"]
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"] - 3  # remove momenta from edge_attr
        edge_num = 5  # need to update this

        gnn_layer = gnn_layer_by_name[layer_name]
        self.lr = model_params["model_learning_rate"]
        self.weight_decay = model_params["model_weight_decay"]
        self.lin_dropout_prob = model_params["model_lin_dropout_prob"]
        self.save_hyperparameters()
        self.loss_fn = MyLoss()

        self.conv_layers = nn.ModuleList([])
        self.transf_layers = nn.ModuleList([])
        self.pooling_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])

        # Transformation layer
        self.conv1 = gnn_layer(
            in_channels=c_in,
            out_channels=embedding_size,
            heads=n_heads,
            dropout=dropout_rate,
            edge_dim=edge_dim,
        )

        self.transf1 = nn.Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(
                gnn_layer(
                    embedding_size,
                    embedding_size,
                    heads=n_heads,
                    dropout=dropout_rate,
                    edge_dim=edge_dim,
                )
            )

            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(
                    TopKPooling(embedding_size, ratio=top_k_ratio)
                )

        # Final layer
        self.conv_fin = gnn_layer(
            in_channels=embedding_size,
            out_channels=1,
            heads=n_heads,
            dropout=dropout_rate,
            edge_dim=edge_dim,
        )

        # Linear layers
        self.linear0 = Linear(embedding_size * 2 + 3 * 2 * edge_num, embedding_size * 2)
        self.linear1 = Linear((embedding_size) * 2, dense_neurons)
        self.linear2 = Linear(dense_neurons, dense_neurons)
        self.linear3 = Linear(dense_neurons, c_out)
        self.linear4 = Linear(embedding_size * 2 + 3 * 2 * edge_num, 1)

        # could use super node instead of topKPooling and linear layers
        # or more topK pooling rather than linear layers

    def forward(self, x, edge_index, edge_attr, batch_index):
        # Remove momenta from edge features
        p = edge_attr[:, 8:11]
        # select just initial and final momenta
        print(p.size())
        p = p.reshape(max(batch_index) + 1, -1)
        edge_attr = edge_attr[:, 0:8]

        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = F.leaky_relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[
                    int(i / self.top_k_every_n)
                ](x, edge_index, edge_attr, batch_index)
                # Add current representation
                global_representation.append(
                    torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
                )

        x = sum(global_representation)

        # add momenta on
        x = torch.cat((x, p), 1)

        # Output block

        x = F.relu(self.linear0(x))
        x = F.dropout(x, p=self.lin_dropout_prob, training=self.training)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.lin_dropout_prob, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.lin_dropout_prob, training=self.training)
        x = torch.sigmoid(self.linear3(x))

        return x

    def training_step(self, batch, batch_idx):
        x, edge_index, edge_attr, y = (
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["y_norm"],
        )
        batch_idx = batch["batch"]
        y_hat = self(x, edge_index, edge_attr, batch_idx)
        loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=max(batch_idx) + 1,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr, y = (
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["y_norm"],
        )
        batch_idx = batch["batch"]
        y_hat = self(x, edge_index, edge_attr, batch_idx)
        loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=max(batch_idx) + 1,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attr, y = (
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["y_norm"],
        )
        batch_idx = batch["batch"]
        y_hat = self(x, edge_index, edge_attr, batch_idx)
        loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=max(batch_idx) + 1,
        )
        return loss

    def predict_step(self, batch, batch_idx):
        x, edge_index, edge_attr, y = (
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["y_norm"],
        )
        batch_idx = batch["batch"]
        y_hat = self(x, edge_index, edge_attr, batch_idx)
        return y_hat.item(), y.item()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


def train_feyn_no_tune(params, num_gpus, num_epochs=10, lr_tune=False):
    """
    Function to train the Feynman GNN without a hyperparameter search.
    params = The hyperparameters to use, stored as a dictionary with the notation "model_..."
    """
    filename = "QED_data.csv"
    print("Loading datasets...")
    train_dataset = FeynmanDataset(
        1000000, reprocess=False, filename=filename, train=True
    )
    test_dataset = FeynmanDataset(100, reprocess=False, filename=filename, test=True)
    val_dataset = FeynmanDataset(50, reprocess=False, filename=filename, val=True)
    pred_dataset = FeynmanDataset(10, reprocess=False, filename=filename, pred=True)
    print("Finished all!")
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, num_workers=0)
    pred_loader = DataLoader(
        dataset=pred_dataset, batch_size=1
    )  # keep this batch_size as one to get predictions to work
    # need to make layer type a hyperparameter
    model_params = {k: v[0] for k, v in params.items() if k.startswith("model_")}
    model = FeynmanModel(
        c_in=-1,  # train_dataset.num_node_features
        c_out=1,  # train_dataset.num_classes
        layer_name="GAT",
        model_params=model_params,
    )
    trainer = pl.Trainer(
        logger=TensorBoardLogger(CHECKPOINT_PATH, name="tb_logs"),
        max_epochs=num_epochs,
        gpus=math.ceil(num_gpus),
        log_every_n_steps=10,
        auto_lr_find=True,
        # progress_bar_refresh_rate=0,
        # callbacks=[EarlyStopping('val_loss')],
    )
    if lr_tune is True:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(
            model, train_loader, val_loader, min_lr=1e-10, max_lr=1e-4
        )

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print(new_lr)

        model_params["model_learning_rate"] = new_lr

    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)
    trainer.test(model, test_loader)

    return model, trainer


def main():
    train_feyn_no_tune(HYPERPARAMETERS, 0)


if __name__ == "__main__":
    main()
