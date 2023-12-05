import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import ChebConv


class ChebNet(nn.Module):
    """
    ChebNet implementation reproduced from: https://github.com/dmlc/dgl/tree/master/examples/pytorch/model_zoo/citation_network
    Check ChebConv implementation at: https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/chebconv.py
    """
    def __init__(self, g, in_feats, n_classes, n_hidden, n_layers, k, bias):
        super(ChebNet, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(in_feats, n_hidden, k, bias=bias))
        for _ in range(n_layers - 1):
            self.layers.append(ChebConv(n_hidden, n_hidden, k, bias=bias))

        self.layers.append(ChebConv(n_hidden, n_classes, k, bias=bias))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h, [2])
        return h