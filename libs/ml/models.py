
from torch.nn import Module
import torch.nn as nn


class MLP_L2(Module):
    # examples
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super(MLP_L2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = MLP_L2(2, 4, 1)
    print(model)
    for name, param in model.named_parameters():
        print(f"Parameter {name}, shape {param.shape}")
