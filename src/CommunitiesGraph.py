import numpy as np

from scipy import sparse

from IPython.display import display, SVG
from sknetwork.visualization import svg_graph

class CommunitiesGraph:
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
        self.label = label
        self.position = position
        self.n_community = n_community
        self.p = p
        self.q = q
        self.n = n

    def display(self):
        """
        Display the graph.
        """
        image = svg_graph(sparse.csr_matrix(self.adjacency), self.position, labels=self.label, node_size=2, edge_width=0.05)
        display( SVG(image) )