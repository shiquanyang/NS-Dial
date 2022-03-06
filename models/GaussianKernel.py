import torch
import torch.nn as nn


class GaussianKernel(nn.Module):
    def __init__(self, emb_size, slope=1.0, boundaries=None):
        super(GaussianKernel, self).__init__()
        self.name = "GaussianKernel"

        self.emb_size = emb_size
        self.slope = slope

        if self.slope is None:
            self.slope = nn.Parameter(torch.ones(1))

        self.boundaries = boundaries

    def forward(self, query, facts):  # query: 3*emb_dim, facts: nb_facts*3*emb_dim
        emb_dim = facts.shape[2]

        query = torch.reshape(query, [-1, 3*emb_dim])
        facts = torch.reshape(facts, [-1, 3*emb_dim])

        dim_x, emb_size_x = query.shape[:-1], query.shape[-1]
        dim_y, emb_size_y = facts.shape[:-1], facts.shape[-1]

        c = - 2 * query @ facts.t()
        na = torch.sum(query ** 2, dim=1, keepdim=True)
        nb = torch.sum(facts ** 2, dim=1, keepdim=True)

        l2 = (c + nb.t()) + na
        l2 = torch.clamp(l2, 1e-6, 1000)
        l2 = torch.sqrt(l2)

        sim = torch.exp(- l2 * self.slope)
        res = torch.reshape(sim, dim_x + dim_y)

        if self.boundaries is not None:
            vmin, vmax = self.boundaries
            scaling_factor = vmax - vmin
            res = (res * scaling_factor) + vmin

        return res

    def pairwise(self, query, facts):
        dim_x, emb_size_x = query.shape[:-1], query.shape[-1]
        dim_y, emb_size_y = facts.shape[:-1], facts.shape[-1]

        a = torch.reshape(query, [-1, emb_size_x])
        b = torch.reshape(facts, [-1, emb_size_y])

        c = - 2 * a @ b.t()
        na = torch.sum(a ** 2, dim=1, keepdim=True)
        nb = torch.sum(b ** 2, dim=1, keepdim=True)

        l2 = (c + nb.t()) + na
        l2 = torch.clamp(l2, 1e-6, 1000)
        l2 = torch.sqrt(l2)

        sim = torch.exp(- l2 * self.slope)
        res = torch.reshape(sim, dim_x + dim_y)

        if self.boundaries is not None:
            vmin, vmax = self.boundaries
            scaling_factor = vmax - vmin
            res = (res * scaling_factor) + vmin

        return res