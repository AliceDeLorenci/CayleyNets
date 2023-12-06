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

        self.n = self.adjacency.shape[0]
        self.n_edges = self.adjacency.nnz/2
        self.n_features = self.features.shape[1]
        self.n_classes = self.labels.max() + 1

        self.name = 'CORA'
        self.description = 'CORA dataset'
