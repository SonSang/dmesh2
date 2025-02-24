import torch as th
from pytorch3d.ops import knn_points
from mindiffdt.cgaldt import CGALDTStruct

'''
Collecting query faces
'''

def qface_knn_spatial(points: th.Tensor, n: int, k: int):
    '''
    Find query faces based on k-nearest neighbors in the space

    @ n = number of neighbors
    @ k = dimension of query face (k-simplex)
    '''
    device = points.device
    num_points = points.shape[0]
    if n + 1 > num_points:
        n = num_points - 1

    # find k-nearest neighbors
    knn_result = knn_points(points.unsqueeze(0), points.unsqueeze(0), K=n + 1)
    nn_idx = knn_result.idx.squeeze(0)
    nn_idx = nn_idx[:, 1:]      # exclude the point itself, [# points, n]

    # find query faces by combining k-nearest neighbors
    combs = th.combinations(th.arange(n, device=device), k)                # [nCk, k]
    n_combs = combs.shape[0]                                
    combs = combs.unsqueeze(0).expand(num_points, -1, -1)   # [num_points, nCk, k]

    nn_idx = nn_idx.unsqueeze(1).expand(-1, n_combs, -1)    # [num_points, nCk, n]

    faces = th.gather(nn_idx, 2, combs)                     # [num_points, nCk, k]
    point0_id = th.arange(num_points, device=device).unsqueeze(1).expand(-1, n_combs)      # [num_points, nCk]
    faces = th.cat([point0_id.unsqueeze(-1), faces], dim=-1)                # [num_points, nCk, k+1]
    faces = faces.view(-1, k + 1)                                           # [num_points * nCk, k+1]
    faces = th.sort(faces, dim=-1)[0]                                       # [num_points * nCk, k+1]
    faces = faces.unique(dim=0)                                             # [num_faces, k+1]

    return faces

def qface_dt(points: th.Tensor, is_real_point: th.Tensor):
    '''
    Find query faces using Delaunay triangulation
    '''
    cgal_dt = CGALDTStruct()
    dt_result = cgal_dt.forward(points, False)
    tets = dt_result.dsimp_point_id.to(dtype=th.long)
    faces = tets[:, [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]].view(-1, 3)
    faces = th.sort(faces, dim=-1)[0]
    faces = faces.unique(dim=0)
    faces_is_real_point = is_real_point[faces]
    real_faces = faces[faces_is_real_point.all(dim=-1)]
    return real_faces

def qedge_dt(points: th.Tensor, is_real_point: th.Tensor):
    '''
    Find query edges using Delaunay triangulation
    '''
    cgal_dt = CGALDTStruct()
    dt_result = cgal_dt.forward(points, False)
    tris = dt_result.dsimp_point_id.to(dtype=th.long)
    edges = tris[:, [0, 1, 1, 2, 0, 2]].view(-1, 2)
    edges = th.sort(edges, dim=-1)[0]
    edges = edges.unique(dim=0)
    edges_is_real_point = is_real_point[edges]
    real_edges = edges[edges_is_real_point.all(dim=-1)]
    return real_edges