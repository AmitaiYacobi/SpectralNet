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
            current_dim = next_dim

        self.architecture = [input_dim] + self.architecture
        for layer in reversed(self.architecture):
            next_dim = layer
            self.decoder.append(
                nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
            )
            current_dim = next_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x
