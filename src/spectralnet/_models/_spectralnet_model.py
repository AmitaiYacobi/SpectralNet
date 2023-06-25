import torch
import numpy as np
import torch.nn as nn


class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        
        current_dim = self.input_dim
        for layer, dim in self.architecture.items():
            next_dim = dim
            if layer == "output_dim":
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                self.layers.append(layer)
            else:
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
                self.layers.append(layer)
                current_dim = next_dim
  
    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        This function orthonormalizes the output of the network 
        using the Cholesky decomposition.

        Args:
            Y (torch.Tensor): The output of the network

        Returns:
            torch.Tensor: The orthonormalized output
        """
        m = Y.shape[0]
        to_factorize = torch.mm(Y.t(), Y)
        
        try:
            L = torch.linalg.cholesky(to_factorize, upper=False)
        except torch._C._LinAlgError:
            to_factorize += 0.1 * torch.eye(to_factorize.shape[0])
            L = torch.linalg.cholesky(to_factorize, upper=False)

        L_inverse = torch.inverse(L)
        orthonorm_weights = np.sqrt(m) * L_inverse.t()
        return orthonorm_weights
    
    
    def forward(self, x: torch.Tensor, should_update_orth_weights: bool = True) -> torch.Tensor:
        """
        This function performs the forward pass of the model.
        If should_update_orth_weights is True, the orthonormalization weights are updated 
        using the Cholesky decomposition.

        Args:
            x (torch.Tensor):                             The input tensor
            should_update_orth_weights (bool, optional):  Whether to update the orthonormalization 
                                                          weights or not

        Returns:
            torch.Tensor: The output tensor
        """

        for layer in self.layers:
            x = layer(x)

        Y_tilde = x
        if should_update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)
        
        Y = torch.mm(Y_tilde, self.orthonorm_weights)
        return Y