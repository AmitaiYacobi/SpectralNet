import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """

        Args:
            output1 (torch.Tensor):     First output of the siamese network
            output2 (torch.Tensor):     Second output of the siamese network
            label (torch.Tensor):       Should be 1 if the two outputs are similar 
                                        and 0 if they are not

        Returns:
            torch.Tensor: loss value
        """
        
        euclidean = nn.functional.pairwise_distance(output1, output2)
        positive_distance = torch.pow(euclidean, 2)
        negative_distance = torch.pow(torch.clamp(self.margin - euclidean, min=0.0), 2)
        loss = torch.mean((label * positive_distance) + ((1 - label) * negative_distance))
        return loss