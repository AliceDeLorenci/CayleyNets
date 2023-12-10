import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import get_laplacian, add_self_loops, to_dense_adj

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ComplexLinear(torch.nn.Module):

    ''' 
    Linear layer for complex valued weights and inputs

    For a complex valued input $z = a + ib $ and a complex valued weight $M=M_R+iM_b, the output is
    $Mz = M_R a - M_I b + i ( M_I a + M_R b)$

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool
        If True, adds a learnable complex bias to the output
    '''

    def __init__(self, in_channels, out_channels, bias=False, weight_initializer='glorot'):
        super(ComplexLinear, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        self.linear_re = Linear(in_channels, out_channels, bias=bias, weight_initializer=weight_initializer)
        self.linear_im = Linear(in_channels, out_channels, bias=bias, weight_initializer=weight_initializer)

    def reset_parameters(self):
        self.linear_re.reset_parameters()
        self.linear_im.reset_parameters()

    def forward(self,x):
        # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        x_re = self.linear_re(x.real) - self.linear_im(x.imag)
        x_im = self.linear_re(x.real) + self.linear_im(x.imag)
        
        return torch.complex(x_re, x_im)   
         
    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.bias})')
    

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
        '''
        # Obtain vector of 1/degrees
        diag_nz_mask = A.indices()[0] == A._indices()[1]
        diag_nz_idx = A.indices()[0][diag_nz_mask]
        diag_inv = torch.zeros(A.size()[0], device = device, dtype=torch.complex64)
        diag_inv[diag_nz_idx] = 1/A.values()[diag_nz_mask]

        diag_mat = torch.sparse_coo_tensor(diag_nz_idx.repeat(2,1), diag_inv, A.size(),  dtype=torch.complex64).coalesce()

        # Off diagonal matrix
        off_diag = A.clone() 
        off_diag.values()[diag_nz_mask] = 0.0

        J = (-1.0) * diag_mat.matmul(off_diag)
        d = diag_mat.matmul(b)
        '''
        # Obtain vector of 1/degrees
        diag_nz_mask = A.indices()[0] == A._indices()[1]
        diag_nz_idx = A.indices()[0][diag_nz_mask]
        #diag_inv = torch.zeros(A.size()[0], device = device, dtype=torch.complex64)
        #diag_inv[diag_nz_idx] = 1/A.values()[diag_nz_mask]
        diag_inv = torch.sparse_coo_tensor(torch.stack([diag_nz_idx, diag_nz_idx]), 1/A.values()[diag_nz_mask],A.size()).coalesce()
        # Off diagonal matrix
        off_diag = A.clone() 
        off_diag.values()[diag_nz_mask] = 0.0

        d = diag_inv @ b 
        # check if d is sparse or dense
        # initialize x 
        x = b.clone()
        # Jacobi iteration
        for k in range(K):
            x = d - diag_inv @ off_diag.matmul(x.clone())
        return x

    else: # dense matrix representation (A.layout == torch.strided)
        diag_inv = torch.diag(A)
        diag_inv = torch.where(diag_inv==0, diag_inv, diag_inv**-1)

        off_diag = A - torch.diag( torch.diag(A) )
        # CSR format affords efficient matrix-vector multiplication
        # too bad that '_to_sparse_csr does not support automatic differentiation for outputs with complex dtype'
        # off_diag = off_diag.to_sparse_csr() 

        diag_inv = torch.reshape(diag_inv, (-1,1))
        d = diag_inv.mul(b) # elementwise multiplication of diag_inv by each column of b

        J = (-1.0)*diag_inv.mul(off_diag)

        # initialize x 
        x = b.clone()
        # Jacobi iteration
        for k in range(K):
            x = d + J.matmul(x)
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
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        
        self.jacobi_iterations = jacobi_iterations

        self.sparse = sparse
        
        self.r = r 

        self.c = torch.nn.ModuleList(
            [Linear(in_channels, out_channels, bias = False, weight_initializer='glorot')]+
            [ComplexLinear(in_channels, out_channels, bias=False, weight_initializer='glorot') for _ in range(r)]
                   )

        # self.c0 = nn.Linear(in_channels, out_channels, device = device, bias = False)
        # self.c = torch.nn.ModuleList( [torch.nn.Linear(in_channels, out_channels, device = device, bias = False).to(torch.cfloat) for _ in range(r)] )  # c parameter
        self.h = Parameter(torch.ones(1, device = device))  # zoom parameter
            
        self.reset_parameters()


    def reset_parameters(self):
        self.h = Parameter(torch.ones(1, device = device))
        # self.c0.reset_parameters()
        for c in self.c:
            c.reset_parameters()
            

    def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor = None):
        """
        Source: https://github.com/WhiteNoyse/SiGCN
        """
        L = get_laplacian(edge_index, edge_weight, normalization=self.normalization, dtype=torch.complex64)
        #L = torch.sparse_coo_tensor(L[0], L[1]).coalesce()
        #L = self.h * L
        #edge_index = edge_index.to(device)

        #Jacobi method
        # L to dense matrix
        if self.sparse:
            A = add_self_loops(L[0], self.h * L[1], fill_value = torch.tensor(1j))
            # A = torch.sparse_coo_tensor(A[0], A[1]).coalesce()
            A = torch.sparse_coo_tensor(A[0], A[1], torch.Size([x.size()[0], x.size()[0]])).coalesce()
            B = add_self_loops(L[0], self.h * L[1], fill_value = torch.tensor(-1j))
            # B = torch.sparse_coo_tensor(B[0], B[1]).coalesce()
            B = torch.sparse_coo_tensor(B[0], B[1], torch.Size([x.size()[0], x.size()[0]])).coalesce()
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
        for i in range(1, self.r+1):
        # for i in range(0, self.r):
            # Jacobi method
            b = B@y_i
            y_i = jacobi_method(A, b, self.jacobi_iterations)
            cumsum = cumsum + self.c[i](y_i)
        #print('cumsum', cumsum)

        # return self.c0(x) + 2*torch.real(cumsum)
        return self.c[0](x) + 2*torch.real(cumsum)

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

        self.sparse = sparse
        self.p_dropout = p_dropout
        self.normalization = normalization


    def forward(self, x, edge_index):

        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

        x = self.layers[-1](x, edge_index)

        return F.log_softmax(x.float(), dim=-1)