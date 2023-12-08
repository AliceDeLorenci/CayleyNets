import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import get_laplacian, add_self_loops, to_dense_adj

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def jacobi_method(A, b, K):
    '''
    Computes approximate solution to
    Ax = b, 
    where A is a symmetric matrix using Jacobi's Method with K iterations
    
    Parameters
    ----------
    A : torch.Tensor 
        Symmetric matrix of shape (n, n)
    b : torch.Tensor
        Vector of shape (n, 1)
    K : int
        Number of iterations
    
    Returns
    -------
    x : torch.Tensor
        Solution of the system of shape (n, 1)
    '''

    # Iterative method notation: u(k+1) = d + J@u(k)

    sparse = A.layout == torch.sparse_coo

    if sparse:  # sparse matrix representation
        # Obtain vector of 1/degrees
        diag_nz_mask = A.indices()[0] == A._indices()[1]
        diag_nz_idx = A.indices()[0][diag_nz_mask]
        diag_inv = torch.zeros(A.size()[0], device = device, dtype=torch.complex64)
        diag_inv[diag_nz_idx] = 1/A.values()[diag_nz_mask]
        # Off diagonal matrix
        off_diag = A.clone() 
        off_diag.values()[diag_nz_mask] = 0.0

    else: # dense matrix representation (A.layout == torch.strided)
        diag_inv = torch.diag(A)
        diag_inv = torch.where(diag_inv==0, diag_inv, diag_inv**-1)

        off_diag = A - torch.diag( torch.diag(A) )
        # CSR format affords efficient matrix-vector multiplication
        # too bad that '_to_sparse_csr does not support automatic differentiation for outputs with complex dtype'
        # off_diag = off_diag.to_sparse_csr() 

    diag_inv = torch.reshape(diag_inv, (-1,1))
    d = diag_inv.mul(b) # elementwise multiplication of diag_inv by each column of b

    # initialize x 
    x = b.clone()
    # Jacobi iteration
    for k in range(K):
        x = d - diag_inv.mul( off_diag.matmul(x) )
    return x


class CayleyConv(nn.Module):

    def __init__(self, in_channels, out_channels, r, normalization='sym', jacobi_iterations=10, sparse=False):
        """
        Implementation of a graph convolutional layer based on Cayley filters.
        Adapted from https://github.com/anon767/CayleyNet
        """
        super(CayleyConv, self).__init__()

        assert r > 0, 'Invalid polynomial degree'
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.normalization = normalization
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.jacobi_iterations = jacobi_iterations

        self.sparse = sparse
        
        self.r = r 
        self.c0 = nn.Linear(in_channels, out_channels, device = device, bias = False)
        self.c = [torch.nn.Linear(in_channels, out_channels, device = device, bias = False).to(torch.cfloat) for _ in range(r)]  # c parameter
        self.h = Parameter(torch.ones(1, device = device))  # zoom parameter
            
        self.reset_parameters()


    def reset_parameters(self):
        self.h = Parameter(torch.ones(1, device = device))
        self.c0.reset_parameters()
        for c in self.c:
            c.reset_parameters()
            

    def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor = None):
        """
        Source: https://github.com/WhiteNoyse/SiGCN
        """
        L = get_laplacian(edge_index, edge_weight, self.normalization, dtype=torch.complex64)
        #L = torch.sparse_coo_tensor(L[0], L[1]).coalesce()
        #L = self.h * L
        #edge_index = edge_index.to(device)

        #Jacobi method
        # L to dense matrix
        if self.sparse:
            A = add_self_loops(L[0], self.h * L[1], fill_value = torch.tensor(1j))
            A = torch.sparse_coo_tensor(A[0], A[1]).coalesce()
            B = add_self_loops(L[0], self.h * L[1], fill_value = torch.tensor(-1j))
            B = torch.sparse_coo_tensor(B[0], B[1]).coalesce()
        else:
            L = to_dense_adj(edge_index = L[0], edge_attr = L[1])
            # L is of size (1, N, N)
            L = self.h * L
            L = L.squeeze() # L is of size (N, N)
            A = L + 1j*torch.eye(L.size()[0], device = device)
            B = L - 1j*torch.eye(L.size()[0], device = device)
           

        
        # A = (hL + iI),  b = (hL - iI)x
        y_i = torch.complex(x, torch.zeros(x.size(), device = device))
        
        # B = hL - iI
        cumsum = 0 + 0j
        for i in range(1, self.r):
            # Jacobi method
            b = B @ y_i
            y_i = jacobi_method(A, b, self.jacobi_iterations)
            cumsum = cumsum + self.c[i](y_i)
        #print('cumsum', cumsum)

        return self.c0(x) + 2*torch.real(cumsum)

    def __repr__(self):
        return '{}({}, {}, r={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.r, self.normalization)


class CayleyNet(torch.nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, n_layers, r=5, p_dropout=0.5, normalization = 'sym', sparse=False, seed=0):

        super(CayleyNet, self).__init__()
        torch.manual_seed(seed)
    
        self.layers = nn.ModuleList()
        self.layers.append(CayleyConv(in_feats, n_hidden, r, normalization=normalization, sparse=sparse))
        for _ in range(n_layers - 1):
            self.layers.append(CayleyConv(n_hidden, n_hidden, r, normalization=normalization, sparse=sparse))

        self.layers.append(CayleyConv(n_hidden, n_classes, r, normalization=normalization, sparse=sparse))
        self.p = p_dropout # dropout probability


        # self.conv1 = CayleyConv(num_node_features, hidden_channels, r, normalization = normalizartion, bias = bias, sparse=sparse)
        #self.pool = TopKPooling(hidden_channels, ratio=0.9)
        # self.conv2 = CayleyConv(hidden_channels, hidden_channels, r, normalization = normalizartion, bias = bias, sparse=sparse)
        # self.lin = Linear(hidden_channels, num_classes)
        self.sparse = sparse
        self.p_dropout = p_dropout
        self.normalizartion = normalization


    def forward(self, x, edge_index):

        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

        x = self.layers[-1](x, edge_index)

        return F.log_softmax(x.float(), dim=-1)