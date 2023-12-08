import scipy.sparse 

from sknetwork.data import load_netset
from sknetwork.utils import directed2undirected

class CORA():
    """
    CORA dataset.
    For more information on the Cora dataset check the NetSet page: https://netset.telecom-paris.fr/pages/cora.html
    """
    def __init__(self):
        self.dataset = load_netset('cora')
        self.adjacency = directed2undirected(self.dataset.adjacency) # undirected graph
        self.features = self.dataset.biadjacency.toarray() # features are given by the biadjacency matrix (between articles and words)
        self.labels = self.dataset.labels # labels are article categories

        self.edge_index = None

        self.n = self.adjacency.shape[0]
        self.n_edges = self.adjacency.nnz
        self.n_features = self.features.shape[1]
        self.n_classes = self.labels.max() + 1

        self.name = 'CORA'
        self.description = 'CORA dataset'

    def get_edge_index(self):
        """
        Returns edge indices according to pytorch-geometric convention
        """
        if self.edge_index is None:
            import torch
            self.adjacency_coo = scipy.sparse.coo_matrix( self.adjacency )
            self.edge_index = torch.stack([torch.tensor(self.adjacency_coo.row, dtype=torch.int64), torch.tensor(self.adjacency_coo.col, dtype=torch.int64)], dim=0)
        return self.edge_index