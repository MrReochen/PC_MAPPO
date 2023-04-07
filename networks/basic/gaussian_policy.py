import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.basic.util import check

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class Gaussian_Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=16, device=torch.device("cpu")):

       	super(Gaussian_Policy, self).__init__()

        self.linear = nn.Linear(input_dim, hidden_size)
        self.mean = nn.Linear(hidden_size, output_dim)
        self.log_std = nn.Linear(hidden_size, output_dim)
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)

    def forward(self, inputs):

        # forward pass of NN
        inputs = check(inputs).to(**self.tpdv)
        x = inputs
        x = F.relu(self.linear(x))

        mean = self.mean(x)
        log_std = self.log_std(x) # if more than one action this will give you the diagonal elements of a diagonal covariance matrix
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # We limit the variance by forcing within a range of -2,20
        std = log_std.exp()

        return mean, std