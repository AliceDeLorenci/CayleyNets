import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, add_self_loops, to_dense_adj
from scipy.linalg import eigh
from numpy import array, identity, diagonal, double
import math

# Ceck if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')   
print('Using device:', device)


class CLinear(nn.Module):
    '''
    Complex Linear Layer
    
    This layer was taken from https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks from the author Mehdi Hosseini Moghadam
    The input format was modified from tensor of shape (N, d, 2) to a complex tensor of shape (N, d) 
    '''
    def __init__(self, in_channels, out_channels, device = device, bias = True):
        super(CLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.bias = bias

        self.re_linear = nn.Linear(self.in_channels, self.out_channels, device = self.device, bias = self.bias)
        self.im_linear = nn.Linear(self.in_channels, self.out_channels, device = self.device, bias = self.bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.re_linear.weight)
        nn.init.xavier_uniform_(self.im_linear.weight)


    def forward(self, x):  
        x_re = x.real
        x_im = x.imag

        # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        out_re = self.re_linear(x_re) - self.im_linear(x_im) 
        out_im = self.re_linear(x_im) + self.im_linear(x_re)

        out = torch.complex(out_re, out_im)

        return out


def jacobi_torch(A, b, K, sparse = False):
    ''' Ax = b, A is a symmetric matrix, Jacobi method with K iterations
    
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

    if sparse:
        indx_nonzero = A._indices()[0] == A._indices()[1]
        diag_indx_nonzero = A._indices()[0][indx_nonzero]
        diag_inv = torch.zeros(A.size()[0], device = device, dtype=torch.complex64)
        diag_inv[diag_indx_nonzero] = 1/A._values()[indx_nonzero]
        diag_mat = torch.sparse_coo_tensor(diag_indx_nonzero.repeat(2,1), diag_inv, A.size(),  dtype=torch.complex64).coalesce()
        #diag_mat = torch.sparse.FloatTensor(diag_inv.repeat(2,1), diag_inv, A.size())
        off_diag = A.clone() 
        off_diag._values()[diag_indx_nonzero] = 0.0
        J = (-1.0) * diag_mat.mm(off_diag)
        b = diag_mat.mm(b)
    else:
        # get diagonal of A sparse matrix
        diag_inv = torch.diag(A)**-1
        # delete inf values
        diag_inv[diag_inv == float('inf')] = 0
        # create off-diagonal matrix of A 
        off_diag = A - torch.diag(torch.diag(A))
        # Jacobbi matrix
        J = -torch.diag(diag_inv) @ off_diag
        b = torch.diag(diag_inv) @ b
        
    # initialize x 
    x = b.clone()
    # Jacobi iteration
    for k in range(K):
        x = J @ x + b 
    return x







class CayleyConv(nn.Module):

    def __init__(self, in_channels, out_channels, r, normalization='sym',
                 bias=False, jacobi_iterations=4, sparse = False):
        """
        Cayley Filter Convolutional Layer
        Jacobi method for approximate eigen decomposition (default to 10 iterations)
        Reference Implementation https://github.com/anon767/CayleyNet
        """
        super(CayleyConv, self).__init__()

        assert r > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.normalization = normalization
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparse = sparse
        
        self.jacobi_iterations = jacobi_iterations
        
        self.r = r 
        self.c0 = torch.nn.Linear(in_channels, out_channels, device = device, bias = bias)
        self.c = nn.ModuleList([CLinear(in_channels, out_channels, device = device, bias = bias) for _ in range(r)])
        #self.c = [torch.nn.Linear(in_channels, out_channels, dtype = torch.complex64, device = device, bias = bias) for _ in range(r)]  # c parameter
        self.h = Parameter(torch.ones(1, device = device))  # zoom parameter

      
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()


    def reset_parameters(self):
        self.h = Parameter(torch.ones(1, device = device))
        self.c0.reset_parameters()
        for c in self.c:
            c.reset_parameters()
        if self.bias is not None:
            self.bias.data.fill_(0)
            

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
            y_i = jacobi_torch(A, b, self.jacobi_iterations, sparse = self.sparse)
            cumsum = cumsum + self.c[i](y_i)
        #print('cumsum', cumsum)
        return self.c0(x) + 2*torch.real(cumsum)

    def __repr__(self):
        return '{}({}, {}, r={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.r, self.normalization)
