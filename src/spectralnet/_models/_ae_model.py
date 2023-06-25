import torch
import torch.nn as nn

class AEModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(AEModel, self).__init__()
        self.architecture = architecture

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.architecture["hidden_dim1"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim1"], self.architecture["hidden_dim2"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim2"], self.architecture["hidden_dim3"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim3"], self.architecture["output_dim"]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.architecture["output_dim"], self.architecture["hidden_dim3"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim3"], self.architecture["hidden_dim2"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim2"], self.architecture["hidden_dim1"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim1"], input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x