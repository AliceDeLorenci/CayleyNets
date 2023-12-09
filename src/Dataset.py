import scipy.sparse 

class Dataset():
    """
    Class meant to standardize dataset treatment.
    """
    def __init__(self):
        self.edge_index = None
        self.adjacency = None
        self.description = None
        self.n = None
        self.n_edges = None
        self.n_feature = None

    def get_edge_index(self):
        """
        Returns edge indices according to pytorch-geometric convention
        """
        if self.edge_index is None:
            import torch
            self.adjacency_coo = scipy.sparse.coo_matrix( self.adjacency )
            self.edge_index = torch.stack([torch.tensor(self.adjacency_coo.row, dtype=torch.int64), torch.tensor(self.adjacency_coo.col, dtype=torch.int64)], dim=0)
        return self.edge_index
    
    def print_properties(self):
        print(self.description)
        print(f'Number of nodes: {self.n}')
        print(f'Number of edges: {self.n_edges}')
        print(f'Number of features per node: {self.n_features}')