import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import ChebConv


class ChebNet(nn.Module):
    """
    ChebNet implementation reproduced from https://github.com/dmlc/dgl/tree/master/examples/pytorch/model_zoo/citation_network
    with minor modifications. 
    Check out ChebConv implementation at: https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/chebconv.py
    """
    def __init__(self, in_feats, n_classes, n_hidden, n_layers, k, p_dropout, bias=False):
        super(ChebNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(in_feats, n_hidden, k, bias=bias))
        for _ in range(n_layers - 1):
            self.layers.append(ChebConv(n_hidden, n_hidden, k, bias=bias))

        self.layers.append(ChebConv(n_hidden, n_classes, k, bias=bias))
        self.p = p_dropout # dropout probability

    def forward(self, g, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(g, h, [2])
            h = F.dropout(h, p=self.p, training=self.training)
        h = self.layers[-1](g, h, [2])
        return h