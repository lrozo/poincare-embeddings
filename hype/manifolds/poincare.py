#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch.autograd import Function
from .euclidean import EuclideanManifold
import numpy as np


class PoincareManifold(EuclideanManifold):
    def __init__(self, eps=1e-5, K=None, **kwargs):
        self.eps = eps
        super(PoincareManifold, self).__init__(max_norm=1 - eps)
        self.K = K
        if K is not None:
            self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * self.K))

    def distance(self, u, v):
        return Distance.apply(u, v, self.eps)

    def half_aperture(self, u):
        eps = self.eps
        sqnu = u.pow(2).sum(dim=-1)
        sqnu.clamp_(min=0, max=1 - eps)
        return th.asin((self.inner_radius * (1 - sqnu) / th.sqrt(sqnu))
            .clamp(min=-1 + eps, max=1 - eps))

    def angle_at_u(self, u, v):
        norm_u = u.norm(2, dim=-1)
        norm_v = v.norm(2, dim=-1)
        dot_prod = (u * v).sum(dim=-1)
        edist = (u - v).norm(2, dim=-1)  # euclidean distance
        num = (dot_prod * (1 + norm_v ** 2) - norm_v ** 2 * (1 + norm_u ** 2))
        denom = (norm_v * edist * (1 + norm_v**2 * norm_u**2 - 2 * dot_prod).sqrt())
        return (num / denom).clamp_(min=-1 + self.eps, max=1 - self.eps).acos()

    def rgrad(self, p, d_p):
        if d_p.is_sparse:
            p_sqnorm = th.sum(
                p[d_p._indices()[0].squeeze()] ** 2, dim=1,
                keepdim=True
            ).expand_as(d_p._values())
            n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
            n_vals.renorm_(2, 0, 5)
            d_p = th.sparse.DoubleTensor(d_p._indices(), n_vals, d_p.size())
        else:
            p_sqnorm = th.sum(p ** 2, dim=-1, keepdim=True)
            d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
        return d_p

    # Special non-associative and non-commutative operation
    # which is closed in the Poincare ball.
    # Performed column-wise here.
    def mobius_addition(self, X, Y):
        scalar_product = np.sum(X * Y, axis=0)
        norm2X = np.sum(X * X, axis=0)
        norm2Y = np.sum(Y * Y, axis=0)

        return (X * (1 + 2 * scalar_product + norm2Y) + Y * (1 - norm2X)) / (
                1 + 2 * scalar_product + norm2X * norm2Y
        )

    def exp(self, X, U):
        norm_U = np.linalg.norm(U, axis=0)
        # Handle the case where U is null.
        W = U * np.divide(
            np.tanh(norm_U / (1 - np.sum(X * X, axis=0))),
            norm_U,
            out=np.zeros_like(U),
            where=norm_U != 0,
        )
        return self.mobius_addition(X, W)

    def log(self, X, Y):
        W = self.mobius_addition(-X, Y)
        norm_W = np.linalg.norm(W, axis=0)
        return (1 - np.sum(X * X, axis=0)) * np.arctanh(norm_W) * W / norm_W

    # def init_weights(self, w):
    #     grid_limits = [-0.5, 0.5, -0.5, 0.5]
    #     x_coord = np.linspace(grid_limits[0], grid_limits[1], 5)
    #     y_coord = np.linspace(grid_limits[3], grid_limits[2], 5)
    #     grid_embeddings = np.zeros((5 * 5, 2))
    #     node = 0
    #     for node_y in range(5):
    #         for node_x in range(5):
    #             grid_embeddings[node, :] = np.array([x_coord[node_x], y_coord[node_y]])
    #             node += 1
    #     w.weight.data = th.from_numpy(grid_embeddings)

class Distance(Function):
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha, 2))\
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = th.sqrt(th.pow(z, 2) - 1)
        z = th.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v, eps):
        squnorm = th.clamp(th.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = th.sqrt(th.pow(x, 2) - 1)
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = Distance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = Distance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None

