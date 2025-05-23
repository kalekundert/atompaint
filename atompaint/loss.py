import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    # [Kendall2018] DOI:10.48550/arXiv.1705.07115

    def __init__(self, n=2):
        super().__init__()
        log_σ2 = torch.ones(n, requires_grad=True)
        self.log_σ2 = nn.Parameter(log_σ2)

    def __repr__(self):
        return f'AutomaticWeightedLoss({self.log_σ2.shape[0]})'

    def forward(self, *losses):
        loss_sum = 0

        for i, loss in enumerate(losses):
            weighted_loss = loss / (2 * torch.exp(self.log_σ2[i]))
            regularization = self.log_σ2[i] / 2
            loss_sum += weighted_loss + regularization

        return loss_sum

