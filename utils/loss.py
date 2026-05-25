import torch


class TestLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(TestLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)
        if self.reduction:
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        epsilon = 1e-8
        x_flat = x.reshape(num_examples, -1)
        y_flat = y.reshape(num_examples, -1)

        diff = x_flat - y_flat
        diff_sum_sq = torch.sum(diff ** 2, dim=1)
        stable_diff_norms = torch.sqrt(diff_sum_sq + epsilon)

        y_sum_sq = torch.sum(y_flat ** 2, dim=1)
        stable_y_norms = torch.sqrt(y_sum_sq.clamp_min(1e-12))

        relative_error = stable_diff_norms / stable_y_norms
        if self.reduction:
            return torch.mean(relative_error) if self.size_average else torch.sum(relative_error)
        return relative_error

    def __call__(self, x, y, loss_type='relative'):
        if loss_type == 'relative':
            return self.rel(x, y)
        elif loss_type == 'absolute':
            return self.abs(x, y)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
