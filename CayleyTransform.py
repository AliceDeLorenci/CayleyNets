import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from scipy import linalg
from scipy import sparse

class CayleyTransform:
    """
    Cayley transform of a graph Laplacian. Provides static methods for computing the Laplacian, the spectrum and the Cayley transform.
    """

    def __init__(self, adjacency):
        assert( np.allclose(adjacency, adjacency.T) ) # the graph should be undirected

        self.A = adjacency

        self.L = CayleyTransform.laplacian(self.A)
        self.Ln = CayleyTransform.normalized_laplacian(self.A)

        self.spectrum = np.sort( CayleyTransform.compute_spectrum(self.L).real )

        self.CL = None
        self.C_spectrum = None
        self.h = None

    @staticmethod
    def laplacian(A):
        """Unnormalized Laplacian of an adjacency matrix"""
        return np.diag(np.sum(A, axis=1)) - A

    @staticmethod
    def normalized_laplacian(A):
        """Normalized Laplacian of an adjacency matrix"""
        d_sqr = np.sqrt( np.sum(A, axis=1) )
        d_inv = [1/x if x != 0 else 0 for x in d_sqr]
        return np.eye(A.shape[0]) - np.diag(d_inv) @ A

    @staticmethod
    def compute_spectrum(L):
        """Compute the eigenvalues and eigenvectors of a matrix"""
        return linalg.eigvals(L)

    @staticmethod
    def cayley_transform(L, h):
        """Cayley transform of a matrix L with zoom parameter h"""
        return (h*L - 1j*np.eye(L.shape[0])) @ np.linalg.inv(h*L + 1j*np.eye(L.shape[0]))
    
    def transform(self, h):
        """Cayley transform of the unnormalized Laplacian with zoom parameter h"""
        self.CL = CayleyTransform.cayley_transform(self.L, h)
        self.C_spectrum = CayleyTransform.compute_spectrum(self.CL)
        self.C_spectrum = np.array( sorted( self.C_spectrum, key=lambda x: x.real ) )
        self.h = h

    def plot_spectrum(self, k = 15):
        """
        Plot the eigenvalues of the unnormalized Laplacian
        
        Parameters
        ----------
        k : int
            Number of eigenvalues to highlight
        """
        eigvals = self.spectrum
        plt.figure(figsize=(4,4))
        plt.plot(eigvals[:k], 'ro', linestyle='None', markersize=1)
        plt.plot(eigvals[k:], 'ko', linestyle='None', markersize=1)
        plt.ylabel(r'$\lambda$', fontsize=12)
        plt.show()

    def plot_transformed_spectrum(self, k = 15):
        """
        Plot the spectrum of the Cayley transform of the laplacian

        Parameters
        ----------
        k : int
            Number of eigenvalues to highlight
        """

        assert( self.C_spectrum is not None )

        

        def generate_semicircle(center_x=0, center_y=0, radius=1, stepsize=0.01):
            """
            Generates coordinates for a semicircle, centered at center_x, center_y
            """        
            angle = np.linspace(0, -np.pi, int(np.pi/stepsize))
            x = np.cos(angle)*radius
            y = np.sin(angle)*radius

            return x, y

        plt.figure(figsize=(4,3))
        plt.scatter(self.C_spectrum.real[:k], self.C_spectrum.imag[:k], c='r', marker='+', linewidth=0.5, s=20)
        plt.scatter(self.C_spectrum.real[k:], self.C_spectrum.imag[k:], c='k', marker='+', linewidth=0.5, s=20)
        plt.xlabel('Re', fontsize=12)
        plt.ylabel('Im', fontsize=12)

        x, y = generate_semicircle()
        plt.plot(x, y, 'k', linewidth=0.5)

        plt.show()