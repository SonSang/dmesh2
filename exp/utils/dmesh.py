import torch as th
from mindiffdt.tgrid import TriGrid, TetGrid

'''
2D
'''
def extract_mesh_from_tgrid_2d(tgrid: TriGrid, f_preal: th.Tensor, consider_apex: bool = True):
    '''
    Extract mesh from TriGrid using point reals.
    '''
    f_edge_reals = f_preal[tgrid.f_edge_idx]
    f_apex_reals = f_preal[tgrid.f_apex_idx]
    f_apex_reals[tgrid.f_apex_idx < 0] = 0

    if consider_apex:
        existing_edges = (f_edge_reals.min(dim=-1)[0] > 0.5) & (f_apex_reals.max(dim=-1)[0] < 0.5)
    else:
        existing_edges = (f_edge_reals.min(dim=-1)[0] > 0.5)
    existing_f_edge_idx = tgrid.f_edge_idx[existing_edges]
    
    return existing_f_edge_idx

def extract_mesh_from_tgrid_logudf_2d(tgrid: TriGrid, f_plogudf: th.Tensor, logudf_thresh: float):
    '''
    Extract mesh from TriGrid using point logudfs.
    '''
    f_edge_logudfs = f_plogudf[tgrid.f_edge_idx]
    f_apex_logudfs = f_plogudf[tgrid.f_apex_idx]
    f_apex_logudfs[tgrid.f_apex_idx < 0] = 10 + logudf_thresh

    max_f_edge_logudfs = f_edge_logudfs.max(dim=-1)[0]
    min_f_apex_logudfs = f_apex_logudfs.min(dim=-1)[0]

    existing_edges = (max_f_edge_logudfs < logudf_thresh) & (min_f_apex_logudfs > logudf_thresh)
    existing_f_edge_idx = tgrid.f_edge_idx[existing_edges]
    
    return existing_f_edge_idx

def extract_mesh_from_tgrid_2d_strict(ppos: th.Tensor, pr: th.Tensor, tgrid: th.Tensor):
    '''
    Strict version: an edge exists only if its minimum bounding circle does not include
    any other points. Therefore, even if an edge is in [tgrid], it could be ignored.
    '''
    # extract edges
    edges = []
    for comb in [[0, 1], [1, 2], [2, 0]]:
        edges.append(tgrid[:, comb])
    edges = th.cat(edges, dim=0)
    edges = th.sort(edges, dim=1)[0]
    edges = th.unique(edges, dim=0)

    # exclude edges that are connected to imaginary points
    edges_pr = pr[edges]
    is_real_edge = edges_pr.min(dim=-1)[0] > 0.5
    edges = edges[is_real_edge]

    # exclude edges that are not strict edges
    edge_cen = (ppos[edges[:, 0]] + ppos[edges[:, 1]]) * 0.5                # [num_edges, 2]
    edge_len = (ppos[edges[:, 0]] - ppos[edges[:, 1]]).norm(dim=-1)
    edge_rad = edge_len * 0.5                                               # [num_edges]

    num_edges = edges.size(0)
    e_ppos = ppos.unsqueeze(0).expand(num_edges, -1, -1)                    # [num_edges, num_points, 2]
    e_edge_cen = edge_cen.unsqueeze(1).expand(-1, e_ppos.size(1), -1)       # [num_edges, num_points, 2]
    e_edge_cen_ppos_dist = (e_ppos - e_edge_cen).norm(dim=-1)               # [num_edges, num_points]
    e_edge_ppos_dist = e_edge_cen_ppos_dist - edge_rad.unsqueeze(1)         # [num_edges, num_points]
    e_edge_ppos_dist.scatter_(1, edges, float('inf'))

    is_strict_edge = e_edge_ppos_dist.min(dim=-1)[0] > 0

    edges = edges[is_strict_edge]

    return edges

'''
3D
'''
def extract_mesh_from_tgrid_3d(tgrid: TetGrid, preal: th.Tensor, threshold: float=0.5):
    '''
    Extract mesh from TetGrid using point reals.
    '''
    face_reals = preal[tgrid.tri_idx]
    existing_faces = face_reals.min(dim=-1)[0] > threshold
    existing_face_idx = tgrid.tri_idx[existing_faces]
    
    return existing_face_idx