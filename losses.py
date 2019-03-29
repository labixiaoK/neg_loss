
import torch
import torch.nn as nn


class NEGLoss(nn.Module):
    def __init__(self, dist, n_classes, n_samples, device):
        super(NEGLoss, self).__init__()

        self.n_samples = n_samples
        self.device = device
        self.n_classes = n_classes

        if dist is not None:
            self.dist = dist
        else:
            self.dist = torch.Tensor([1/self.n_classes]).repeat(self.n_classes)

    def get_samples(self, targets):
        dist = self.dist.expand(targets.shape[0], self.dist.shape[0])
        dist[range(targets.shape[0]), targets] = .0
        return torch.multinomial(dist, self.n_samples, replacement=True)

    def forward(self, probs, targets):
        batch_size = probs.shape[0]
        negs = self.get_samples(targets).to(self.device)

        rng = range(batch_size)
        # Log targets
        log_targets = probs[rng, targets].sigmoid().log().sum()

        # Log samples
        idx = torch.arange(batch_size).expand(self.n_samples, batch_size).transpose(0, 1)
        log_samples = probs[idx, negs].neg().sigmoid().log().sum()

        return - (log_targets + log_samples) / batch_size

