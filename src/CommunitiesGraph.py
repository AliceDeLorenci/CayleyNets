import numpy as np

from scipy import sparse

from IPython.display import display, SVG
from sknetwork.visualization import svg_graph

import src.Dataset as Dataset
import importlib
importlib.reload(Dataset)

class CommunitiesGraph(Dataset.Dataset):
    """
    Generate a synthetic n-communities graph. Uses scikit-network to display graph.
    """

    def __init__(self, n_community, p, q, n, seed=0):
        """
        Generate a random graph with n_community communities, p intra-community probability, q inter-community probability and n nodes.
        
        Parameters
        ----------
        n_community : int
            Number of communities   
        p : float   
            Probability of an edge within a community
        q : float
            Probability of an edge between communities
        n : int
            Number of nodes
        """
        super(CommunitiesGraph, self).__init__()

        np.random.seed(seed)
        
        label = np.zeros(n, dtype=int)
        position = np.zeros( (n,2) )
        adjacency = np.zeros( (n,n) )

        for i in range(n):
            label[i] = np.random.randint(n_community)
            c = label[i]
            position[i] = np.random.multivariate_normal( [np.round(1+c/n_community)*np.cos(2*np.pi*2*c/n_community),np.round(1+c/n_community)*np.sin(2*np.pi*2*c/n_community)],
                                                        0.05*np.eye(2) )

        for i in range(n):
            for j in range(i+1,n):
                if label[i] == label[j]:
                    adjacency[i,j] = int(np.random.binomial(1, p))
                else:
                    adjacency[i,j] = int(np.random.binomial(1, q))
                adjacency[j,i] = adjacency[i,j]
        
        self.adjacency = adjacency
        self.features = np.ones( (self.adjacency.shape[0],1) )
        self.labels = label
        self.positions = position

        self.n_community = n_community
        self.p = p
        self.q = q

        self.n = n
        self.n_edges = np.count_nonzero(self.adjacency)
        self.n_features = self.features.shape[1]
        self.n_classes = self.n_community

        self.name = 'Communities'
        self.description = 'Communities dataset'

    def display(self, labels=None):
        """
        Display the graph.
        """
        if labels is None:
            labels=self.labels
        image = svg_graph(sparse.csr_matrix(self.adjacency), self.positions, labels=labels, node_size=2, edge_width=0.05)
        display( SVG(image) )
