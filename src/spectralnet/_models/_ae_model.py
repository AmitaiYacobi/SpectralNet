import torch
import torch.nn as nn


class AEModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(AEModel, self).__init__()
        self.architecture = architecture
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        current_dim = input_dim
        for layer in self.architecture:
            next_dim = layer
            self.encoder.append(
                nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
            )
            self.decoder.append(
                nn.Sequential(nn.Linear(next_dim, current_dim), nn.ReLU())
            )
            current_dim = next_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
