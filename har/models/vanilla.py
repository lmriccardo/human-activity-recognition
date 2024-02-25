import torch
import torch.nn as nn
import torch.nn.functional as F

from har.utils.constants import DEVICE
from typing import Tuple


class VanillaRNNModel(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_dim: int, n_layers: int,
        bidirectional: bool=False, bias: bool=True, batch_first: bool=True
    ) -> None:
        super(VanillaRNNModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.bias = bias
        self.batch_first = batch_first
        self.d = 1 if not self.bidirectional else 2

        # Define all the RNN network
        self.rnn = nn.RNN(
            self.input_size, self.hidden_dim, self.n_layers,
            bidirectional=self.bidirectional, batch_first=self.batch_first
        )

        # Defined the final Fully Connected Layer
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Apply the RNN and the Fully Connected layer at the end """
        batch_size = x.shape[0]
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)

        # In case the input is batched also the output will be batched
        # For this reason let's reshape it to be given to the FC Network
        out = out[:, -1, :]
        out = self.fc(out)

        return F.softmax(out, dim=1), hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """ Initialize hidden state with zeros """
        hidden = torch.zeros(self.d * self.n_layers, batch_size, self.hidden_dim)
        hidden = hidden.to(DEVICE)
        return hidden