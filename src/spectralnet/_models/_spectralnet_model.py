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
        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                )
            else:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
                )
                current_dim = next_dim

    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the output of the network using the Cholesky decomposition.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the network.

        Returns
        -------
        torch.Tensor
            The orthonormalized output.

        Notes
        -----
        This function applies the Cholesky decomposition to orthonormalize the output (`Y`) of the network.
        The orthonormalized output is returned as a tensor.
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

    def forward(
        self, x: torch.Tensor, should_update_orth_weights: bool = True
    ) -> torch.Tensor:
        """
        Perform the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        should_update_orth_weights : bool, optional
            Whether to update the orthonormalization weights using the Cholesky decomposition or not.

        Returns
        -------
        torch.Tensor
            The output tensor.

        Notes
        -----
        This function takes an input tensor `x` and computes the forward pass of the model.
        If `should_update_orth_weights` is set to True, the orthonormalization weights are updated
        using the Cholesky decomposition. The output tensor is returned.
        """

        for layer in self.layers:
            x = layer(x)

        Y_tilde = x
        if should_update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)

        Y = torch.mm(Y_tilde, self.orthonorm_weights)
        return Y
