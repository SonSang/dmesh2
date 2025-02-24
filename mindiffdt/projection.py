import torch as th
from pytorch3d.ops.knn import knn_points

'''
Final projection algorithm, which is differentiable
'''
def projection(
        simplexes: th.Tensor,
        simp_ball_center: th.Tensor,
        simp_ball_radius: th.Tensor,
        point_positions: th.Tensor,
        nearest: th.Tensor):
    '''
    For each simplex, compute signed distance between its
    (minimum) bounding ball and the nearest points.

    The positive value means the simplex exists.
    The negative value means the simplex does not exist.

    @ simplexes: [# simplex, # k + 1]
    @ simp_ball_center: [# simplex, # dim]
    @ simp_ball_radius: [# simplex,]
    @ point_positions: [# point, # dim]
    @ nearest: [# simplex,]
    '''

    nearest_point_positions = point_positions[nearest]                          # [# simplex, # dim]

    d1 = th.norm(nearest_point_positions - simp_ball_center, dim=-1, p=2)       # [# simplex,]
    sdist = d1 - simp_ball_radius                                               # [# simplex,]

    return sdist

def projection_multi(
        simplexes: th.Tensor,
        simp_ball_center: th.Tensor,
        simp_ball_radius: th.Tensor,
        point_positions: th.Tensor,
        nearest: th.Tensor,
        nearest_is_valid: th.Tensor):
    '''
    For each simplex, compute signed distance between its
    (minimum) bounding ball and the nearest points.

    The positive value means the simplex exists.
    The negative value means the simplex does not exist.

    @ simplexes: [# simplex, # k + 1]
    @ simp_ball_center: [# simplex, # dim]
    @ simp_ball_radius: [# simplex,]
    @ point_positions: [# point, # dim]
    @ nearest: [# simplex, #topk]
    @ nearest_is_valid: [# simplex, #topk]
    '''
    assert nearest.ndim == 2, "nearest should be 2D tensor"
    assert nearest_is_valid.ndim == 2, "nearest_is_valid should be 2D tensor"

    nearest_point_positions = point_positions[nearest]                                          # [# simplex, # topk, # dim]

    d1 = th.norm(nearest_point_positions - simp_ball_center.unsqueeze(1), dim=-1, p=2)          # [# simplex, # topk]
    d1_max = d1.max()
    with th.no_grad():
        d1.data[~nearest_is_valid] = d1_max + 1                                                 # [# simplex, # topk]

    sdist = d1.min(dim=-1).values - simp_ball_radius                                            # [# simplex,]

    return sdist

'''
KNN-based algorithm
'''
@th.no_grad()
def knn_search(
    simplexes: th.Tensor,
    simp_ball_center: th.Tensor,
    simp_ball_radius: th.Tensor,
    point_positions: th.Tensor,):
    '''
    For each (k-)simplex, find the [topk] nearest 
    points to its bounding ball center.
    
    This function conducts knn-based search.
    
    @ simplexes: [# simplex, # k + 1]
    @ simp_ball_center: [# simplex, # dim]
    @ simp_ball_radius: [# simplex,]
    @ point_positions: [# point, # dim]
    
    @ return: [# simplex,], index of the nearest points
    to the bounding ball center of given simplex.
    '''
    k = simplexes.shape[1] - 1

    topk = k + 2
    knn_result = knn_points(
        simp_ball_center.unsqueeze(0),
        point_positions.unsqueeze(0),
        K=topk,
        return_nn=True,
    )
    knn_idx = knn_result.idx[0, :, 0]                                       # [# simplex,]
    knn_idx_fixed = th.zeros_like(knn_idx, dtype=th.bool)
    
    # choose the point that does not belong to the simplex
    for i in range(k + 2):
        curr_knn_idx = knn_result.idx[0, :, i]                              # [# simplex,]
        curr_knn_idx_is_simp = (curr_knn_idx.unsqueeze(-1) == simplexes).any(dim=-1)
        curr_knn_idx_change = ~curr_knn_idx_is_simp & ~knn_idx_fixed
        knn_idx[curr_knn_idx_change] = curr_knn_idx[curr_knn_idx_change]

        knn_idx_fixed[curr_knn_idx_change] = True

    info = {}

    return knn_idx, info

@th.no_grad()
def knn_search_multi(
    simplexes: th.Tensor,
    simp_ball_center: th.Tensor,
    simp_ball_radius: th.Tensor,
    point_positions: th.Tensor,
    topk: int):
    '''
    For each (k-)simplex, find the [topk] nearest 
    points to its bounding ball center.
    
    This function conducts knn-based search.
    
    @ simplexes: [# simplex, # k + 1]
    @ simp_ball_center: [# simplex, # dim]
    @ simp_ball_radius: [# simplex,]
    @ point_positions: [# point, # dim]
    
    @ return: [# simplex,], index of the nearest points
    to the bounding ball center of given simplex.
    '''
    assert topk > 0, "topk should be larger than 0"
    k = simplexes.shape[1] - 1

    topk = topk + k + 1
    knn_result = knn_points(
        simp_ball_center.unsqueeze(0),
        point_positions.unsqueeze(0),
        K=topk,
        return_nn=True,
    )

    knn_idx = knn_result.idx[0]                                       # [# simplex, topk]
    knn_idx_is_valid = (knn_idx.unsqueeze(-1) != simplexes.unsqueeze(1)).all(dim=-1)
    
    return knn_idx, knn_idx_is_valid