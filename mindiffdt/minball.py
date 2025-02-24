'''
Min ball of a (d-1) simplex
'''

import torch as th
from typing import NamedTuple

class Ball(NamedTuple):
    center: th.Tensor
    radius: th.Tensor

class MB2_V0:
    '''
    Min ball computation in 2D, using Pytorch autograd
    '''
    @staticmethod
    def forward(p1: th.Tensor, p2: th.Tensor):
        '''
        @ p1: [N, 2]
        @ p2: [N, 2]
        '''

        center = (p1 + p2) * 0.5
        d1 = th.norm(p1 - center, dim=-1, p=2)
        d2 = th.norm(p2 - center, dim=-1, p=2)
        radius = (d1 + d2) * 0.5

        return Ball(center, radius)

class MB3_V0:
    '''
    Min ball computation in 3D, using Pytorch autograd
    '''
    @staticmethod
    def forward(p1: th.Tensor, p2: th.Tensor, p3: th.Tensor):
        '''
        @ p1: [N, 3]
        @ p2: [N, 3]
        @ p3: [N, 3]

        https://gamedev.stackexchange.com/questions/60630/how-do-i-find-the-circumcenter-of-a-triangle-in-3d
        '''
        min_len = 1e-5
        min_ang = 1e-4

        b_a = p2 - p1
        c_a = p3 - p1
        c_b = p3 - p2

        '''
        Stability check:
        1) (b_a) and (c_a) should not be parallel
        2) (b_a) and (c_a) should not be very short
        '''
        with th.no_grad():
            b_a_len = th.norm(b_a, dim=-1, p=2)
            c_a_len = th.norm(c_a, dim=-1, p=2)
            stable_mask_0 = th.logical_and(b_a_len > min_len, c_a_len > min_len)

            b_a_unit = b_a / b_a_len.unsqueeze(-1)
            c_a_unit = c_a / c_a_len.unsqueeze(-1)
            cos_angle = th.sum(b_a_unit * c_a_unit, dim=-1)
            stable_mask_1 = th.abs(cos_angle) < 1 - min_ang
            stable_mask = th.logical_and(stable_mask_0, stable_mask_1)

            smaller_len = th.stack([b_a_len, c_a_len], dim=-1).min(dim=-1).values
            multiplier = 1.0 / smaller_len

        center = th.zeros_like(p1)
        radius = th.zeros_like(p1[:, 0])

        b_a = (b_a * multiplier.unsqueeze(-1))[stable_mask]
        c_a = (c_a * multiplier.unsqueeze(-1))[stable_mask]
        c_b = (c_b * multiplier.unsqueeze(-1))[stable_mask]

        m1 = th.sum(b_a * b_a, dim=-1, keepdim=True)
        m2 = th.sum(c_a * c_a, dim=-1, keepdim=True)
        m3 = th.sum(c_a * c_b, dim=-1, keepdim=True)
        m4 = th.sum(b_a * c_b, dim=-1, keepdim=True)

        denom = (c_a * m1 * m3) - (b_a * m2 * m4)

        m5 = th.sum(b_a * c_a, dim=-1, keepdim=True)

        nom = 2 * ((m1 * m2) - (m5 * m5))

        add = (denom / nom) / multiplier[stable_mask].unsqueeze(-1)
        center[stable_mask] = p1[stable_mask] + add
        
        d1 = th.norm(p1 - center, dim=-1, p=2)[stable_mask]
        d2 = th.norm(p2 - center, dim=-1, p=2)[stable_mask]
        d3 = th.norm(p3 - center, dim=-1, p=2)[stable_mask]
        ds = th.stack([d1, d2, d3], dim=-1)

        radius[stable_mask] = th.mean(ds, dim=-1)
        radius_var = th.var(ds, dim=-1)

        # assert radius_var.max() < 1e-5, "Unstable radius"
        assert center.isnan().sum() == 0, "Center is NaN"
        assert radius.isnan().sum() == 0, "Radius is NaN"

        return Ball(center, radius), stable_mask