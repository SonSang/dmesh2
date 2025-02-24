import torch as th
import numpy as np
from torch_scatter import scatter_min

C0 = np.sqrt(3) * 0.5

class TriGrid:
    '''
    Triangle grid structure, where every simplex in the grid satisfies
    the strong delaunay property trivially. That is, the minimum circumscribing
    ball of each simplex does not contain any other vertex in the grid.
    '''

    def __init__(self, device):

        self.verts: th.Tensor = None                    # [# row, # col, 2]
        self.f_verts: th.Tensor = None                  # [# row * # col, 2]

        self.tri_idx: th.Tensor = None                  # [# tri, 3, 2]
        self.edge_idx: th.Tensor = None                 # [# edge, 2, 2]
        self.apex_idx: th.Tensor = None                 # [# edge, 2, 2]
        self.bdry_idx: th.Tensor = None                 # [# bdry, 2]

        self.f_tri_idx: th.Tensor = None                # [# tri, 3]
        self.f_edge_idx: th.Tensor = None               # [# edge, 2]
        self.f_apex_idx: th.Tensor = None               # [# edge, 2]
        self.f_bdry_idx: th.Tensor = None               # [# bdry, 2]

        # some edges have only one apex
        self.apex_idx_valid: th.Tensor = None           # [# edge, 2]

        # data structure for [safe_prange]
        self.f_apex_idx_nonneg: th.Tensor = None        # [# edge, 2]

        self.device = device

        self.grid_tri_edge_length = None
        self.grid_tri_height_length = None
    
    def init(self, domain_min: tuple, domain_max: tuple, max_grid_size: float):
        '''
        Initialize [self.verts] so that this grid covers the given domain.

        @ max_grid_size: maximum grid size, the grid size can be smaller than this value.
        '''
        device= self.device

        '''
        1. Set verts
        '''

        x_min, y_min = domain_min
        x_max, y_max = domain_max
        x_length = x_max - x_min
        y_length = y_max - y_min

        # compute grid size
        num_grid_x = int(x_length / max_grid_size) + 1
        grid_tri_edge_length = x_length / num_grid_x
        num_grid_x = num_grid_x + 1         # append one more triangle for 2d matrix

        grid_tri_height_length = C0 * grid_tri_edge_length
        num_grid_y = int(y_length / grid_tri_height_length) + 1

        # length
        self.grid_tri_edge_length = grid_tri_edge_length
        self.grid_tri_height_length = grid_tri_height_length

        # count number of rows and cols
        num_even_rows = (num_grid_y // 2) + 1
        num_odd_rows = num_even_rows

        num_rows = num_even_rows + num_odd_rows
        num_cols = num_grid_x + 1

        odd_row_idx = th.arange(1, num_rows, 2, device=device, dtype=th.long)

        # compute grid vertices
        grid_verts = th.zeros((num_rows, num_cols, 2), dtype=th.float32, device=device)

        row_verts_x = th.arange(0, num_cols, device=device, dtype=th.float32) * grid_tri_edge_length
        row_verts_y = th.arange(0, num_rows, device=device, dtype=th.float32) * grid_tri_height_length
        row_verts_x = row_verts_x + x_min
        row_verts_y = row_verts_y + y_min

        grid_verts[:, :, 0] = row_verts_x.unsqueeze(0)
        grid_verts[:, :, 1] = row_verts_y.unsqueeze(1)

        # odd rows
        grid_verts[odd_row_idx, :, 0] = grid_verts[odd_row_idx, :, 0] - (grid_tri_edge_length * 0.5)

        # save
        self.verts = grid_verts
        self.f_verts = self.flattened_verts()

        '''
        2. Set tri_idx
        '''
        self.tri_idx = self.tri_vertex_idx()
        self.f_tri_idx = self.vertex_flattened_id(self.tri_idx)

        '''
        3. Set edge_idx and apex_idx
        '''
        self.edge_idx, self.apex_idx = self.edge_vertex_idx()
        self.f_edge_idx = self.vertex_flattened_id(self.edge_idx)
        self.f_apex_idx = self.vertex_flattened_id(self.apex_idx)
        self.f_apex_idx_nonneg = self.f_apex_idx.clone()

        # set valid apex indices
        apex_idx_valid_x = (self.apex_idx[..., 0] >= 0) & (self.apex_idx[..., 0] < num_rows)
        apex_idx_valid_y = (self.apex_idx[..., 1] >= 0) & (self.apex_idx[..., 1] < num_cols)
        self.apex_idx_valid = apex_idx_valid_x & apex_idx_valid_y
        self.f_apex_idx_nonneg[~self.apex_idx_valid] = 0

        '''
        4. Set bdry_idx
        '''
        self.bdry_idx = self.bdry_vertex_idx()
        self.f_bdry_idx = self.vertex_flattened_id(self.bdry_idx)

    def update_flattened_verts(self, f_verts: th.Tensor):
        self.f_verts = f_verts
        self.verts = f_verts.view(self.verts.shape)

    def flattened_verts(self):
        '''
        Flatten [self.verts] to a 2d tensor.
        '''
        return self.verts.view(-1, 2)

    def vertex_flattened_id(self, idx):
        '''
        Change 2d grid indices to flattened indices.

        @ idx: 2d grid indices, (..., 2)
        '''
        row = idx[..., 0]
        col = idx[..., 1]
        row_valid = (row >= 0) & (row < self.verts.shape[0])
        col_valid = (col >= 0) & (col < self.verts.shape[1])
        valid = row_valid & col_valid

        num_cols = self.verts.shape[1]
        result = row * num_cols + col
        result[~valid] = -1
        
        return result

    def tri_vertex_idx(self):
        '''
        Get vertex indices that compose each triangle.
        Each triangle is defined in a CCW order.
        '''
        device = self.device
        num_rows, num_cols, _ = self.verts.shape
        tri_idx = th.zeros((num_rows - 1, num_cols - 1, 2, 3, 2), dtype=th.long, device=self.device)

        # fill centers
        center_indices = th.arange(0, (num_rows - 1) * (num_cols - 1), device=device, dtype=th.long)
        center_indices_x = center_indices // (num_cols - 1)
        center_indices_y = center_indices % (num_cols - 1)
        
        center_indices = th.stack([center_indices_x, center_indices_y], dim=-1)
        center_indices = center_indices.view(num_rows - 1, num_cols - 1, 2)
        tri_idx[:, :, :, 0, :] = center_indices.unsqueeze(2)

        # fill corners
        even_rows = th.arange(0, num_rows - 1, 2, device=self.device, dtype=th.long)
        odd_rows = th.arange(1, num_rows - 1, 2, device=self.device, dtype=th.long)
        
        # even rows
        tri_idx[even_rows, :, 0, 1, 0] = tri_idx[even_rows, :, 0, 0, 0] + 1
        tri_idx[even_rows, :, 0, 1, 1] = tri_idx[even_rows, :, 0, 0, 1] + 1
        tri_idx[even_rows, :, 0, 2, 0] = tri_idx[even_rows, :, 0, 0, 0] + 1
        tri_idx[even_rows, :, 0, 2, 1] = tri_idx[even_rows, :, 0, 0, 1]

        tri_idx[even_rows, :, 1, 1, 0] = tri_idx[even_rows, :, 0, 0, 0]
        tri_idx[even_rows, :, 1, 1, 1] = tri_idx[even_rows, :, 0, 0, 1] + 1
        tri_idx[even_rows, :, 1, 2, 0] = tri_idx[even_rows, :, 0, 0, 0] + 1
        tri_idx[even_rows, :, 1, 2, 1] = tri_idx[even_rows, :, 0, 0, 1] + 1

        # odd rows
        # shift centers
        tri_idx[odd_rows, :, :, 0, 1] = tri_idx[odd_rows, :, :, 0, 1] + 1

        tri_idx[odd_rows, :, 0, 1, 0] = tri_idx[odd_rows, :, 0, 0, 0] + 1
        tri_idx[odd_rows, :, 0, 1, 1] = tri_idx[odd_rows, :, 0, 0, 1] - 1
        tri_idx[odd_rows, :, 0, 2, 0] = tri_idx[odd_rows, :, 0, 0, 0]
        tri_idx[odd_rows, :, 0, 2, 1] = tri_idx[odd_rows, :, 0, 0, 1] - 1

        tri_idx[odd_rows, :, 1, 1, 0] = tri_idx[odd_rows, :, 0, 0, 0] + 1
        tri_idx[odd_rows, :, 1, 1, 1] = tri_idx[odd_rows, :, 0, 0, 1]
        tri_idx[odd_rows, :, 1, 2, 0] = tri_idx[odd_rows, :, 0, 0, 0] + 1
        tri_idx[odd_rows, :, 1, 2, 1] = tri_idx[odd_rows, :, 0, 0, 1] - 1

        tri_idx = tri_idx.view(-1, 3, 2)

        return tri_idx

    def edge_vertex_idx(self):
        '''
        Get vertex indices that compose each edge.
        Also, find apex vertex indices for each edge.
        '''
        device = self.device
        num_rows, num_cols, _ = self.verts.shape
        edge_idx = th.zeros((num_rows - 1, num_cols - 1, 3, 2, 2), dtype=th.long, device=self.device)
        apex_idx = th.zeros((num_rows - 1, num_cols - 1, 3, 2, 2), dtype=th.long, device=self.device)

        # fill centers
        center_indices = th.arange(0, (num_rows - 1) * (num_cols - 1), device=device, dtype=th.long)
        center_indices_x = center_indices // (num_cols - 1)
        center_indices_y = center_indices % (num_cols - 1)
        
        center_indices = th.stack([center_indices_x, center_indices_y], dim=-1)
        center_indices = center_indices.view(num_rows - 1, num_cols - 1, 2)
        edge_idx[:, :, :, 0, :] = center_indices.unsqueeze(2)

        # fill corners
        even_rows = th.arange(0, num_rows - 1, 2, device=self.device, dtype=th.long)
        odd_rows = th.arange(1, num_rows - 1, 2, device=self.device, dtype=th.long)
        
        # even rows
        # edge 0
        edge_idx[even_rows, :, 0, 1, 0] = edge_idx[even_rows, :, 0, 0, 0]
        edge_idx[even_rows, :, 0, 1, 1] = edge_idx[even_rows, :, 0, 0, 1] + 1

        apex_idx[even_rows, :, 0, 0, 0] = edge_idx[even_rows, :, 0, 0, 0] - 1
        apex_idx[even_rows, :, 0, 0, 1] = edge_idx[even_rows, :, 0, 0, 1] + 1
        apex_idx[even_rows, :, 0, 1, 0] = edge_idx[even_rows, :, 0, 0, 0] + 1
        apex_idx[even_rows, :, 0, 1, 1] = edge_idx[even_rows, :, 0, 0, 1] + 1

        # edge 1
        edge_idx[even_rows, :, 1, 1, 0] = edge_idx[even_rows, :, 1, 0, 0] + 1
        edge_idx[even_rows, :, 1, 1, 1] = edge_idx[even_rows, :, 1, 0, 1] + 1

        apex_idx[even_rows, :, 1, 0, 0] = edge_idx[even_rows, :, 1, 0, 0]
        apex_idx[even_rows, :, 1, 0, 1] = edge_idx[even_rows, :, 1, 0, 1] + 1
        apex_idx[even_rows, :, 1, 1, 0] = edge_idx[even_rows, :, 1, 0, 0] + 1
        apex_idx[even_rows, :, 1, 1, 1] = edge_idx[even_rows, :, 1, 0, 1]

        # edge 2
        edge_idx[even_rows, :, 2, 1, 0] = edge_idx[even_rows, :, 2, 0, 0] + 1
        edge_idx[even_rows, :, 2, 1, 1] = edge_idx[even_rows, :, 2, 0, 1]

        apex_idx[even_rows, :, 2, 0, 0] = edge_idx[even_rows, :, 2, 0, 0] + 1
        apex_idx[even_rows, :, 2, 0, 1] = edge_idx[even_rows, :, 2, 0, 1] + 1
        apex_idx[even_rows, :, 2, 1, 0] = edge_idx[even_rows, :, 2, 0, 0]
        apex_idx[even_rows, :, 2, 1, 1] = edge_idx[even_rows, :, 2, 0, 1] - 1

        # odd rows
        # shift centers
        edge_idx[odd_rows, :, :, 0, 1] = edge_idx[odd_rows, :, :, 0, 1] + 1

        # edge 0
        edge_idx[odd_rows, :, 0, 1, 0] = edge_idx[odd_rows, :, 0, 0, 0] + 1
        edge_idx[odd_rows, :, 0, 1, 1] = edge_idx[odd_rows, :, 0, 0, 1]

        apex_idx[odd_rows, :, 0, 0, 0] = edge_idx[odd_rows, :, 0, 0, 0]
        apex_idx[odd_rows, :, 0, 0, 1] = edge_idx[odd_rows, :, 0, 0, 1] + 1
        apex_idx[odd_rows, :, 0, 1, 0] = edge_idx[odd_rows, :, 0, 0, 0] + 1
        apex_idx[odd_rows, :, 0, 1, 1] = edge_idx[odd_rows, :, 0, 0, 1] - 1

        # edge 1
        edge_idx[odd_rows, :, 1, 1, 0] = edge_idx[odd_rows, :, 1, 0, 0] + 1
        edge_idx[odd_rows, :, 1, 1, 1] = edge_idx[odd_rows, :, 1, 0, 1] - 1

        apex_idx[odd_rows, :, 1, 0, 0] = edge_idx[odd_rows, :, 1, 0, 0] + 1
        apex_idx[odd_rows, :, 1, 0, 1] = edge_idx[odd_rows, :, 1, 0, 1]
        apex_idx[odd_rows, :, 1, 1, 0] = edge_idx[odd_rows, :, 1, 0, 0]
        apex_idx[odd_rows, :, 1, 1, 1] = edge_idx[odd_rows, :, 1, 0, 1] - 1

        # edge 2
        edge_idx[odd_rows, :, 2, 1, 0] = edge_idx[odd_rows, :, 2, 0, 0]
        edge_idx[odd_rows, :, 2, 1, 1] = edge_idx[odd_rows, :, 2, 0, 1] - 1

        apex_idx[odd_rows, :, 2, 0, 0] = edge_idx[odd_rows, :, 2, 0, 0] + 1
        apex_idx[odd_rows, :, 2, 0, 1] = edge_idx[odd_rows, :, 2, 0, 1] - 1
        apex_idx[odd_rows, :, 2, 1, 0] = edge_idx[odd_rows, :, 2, 0, 0] - 1
        apex_idx[odd_rows, :, 2, 1, 1] = edge_idx[odd_rows, :, 2, 0, 1] - 1

        # flatten
        edge_idx = edge_idx.view(-1, 2, 2)
        apex_idx = apex_idx.view(-1, 2, 2)

        return edge_idx, apex_idx
    
    def bdry_vertex_idx(self):
        '''
        Get boundary vertex indices.
        '''
        device = self.device
        num_rows, num_cols, _ = self.verts.shape

        bdry_verts = th.zeros((num_rows, num_cols), dtype=th.bool, device=device)
        bdry_verts[0, :] = True
        bdry_verts[-1, :] = True
        bdry_verts[:, 0] = True
        bdry_verts[:, -1] = True
        bdry_idx = th.nonzero(bdry_verts, as_tuple=False)     # (num_boundary_verts, 2)

        return bdry_idx

class TetGrid:
    '''
    Tetrahedra grid structure, where every simplex in the grid satisfies
    the strong delaunay property trivially. That is, the minimum circumscribing
    ball of each simplex does not contain any other vertex in the grid.
    '''

    def __init__(self, device):

        self.grid_size = None
        self.domain_min = None
        self.domain_max = None

        self.verts: th.Tensor = None                    # [# verts, 2]
        
        self.tet_idx: th.Tensor = None                  # [# tet, 4]
        self.tri_idx: th.Tensor = None                  # [# tri, 3]
        self.apex_idx: th.Tensor = None                 # [# tri, 2]

        # some tris have only one apex
        self.apex_idx_valid: th.Tensor = None           # [# tri, 2]

        # dist between apex and circumball
        self.apex_circumball_dist = None

        self.num_grid_x = None
        self.num_grid_y = None
        self.num_grid_z = None

        self.device = device

    def _x_edge_tets(self, num_grid_x, num_grid_y, num_grid_z, device, len_cube_lattice_verts):
        edge_x_start_idx = th.stack(
            th.meshgrid(
                th.arange(0, num_grid_x, device=device, dtype=th.long),
                th.arange(0, num_grid_y + 1, device=device, dtype=th.long),
                th.arange(0, num_grid_z + 1, device=device, dtype=th.long),
                indexing='ij'
            ),
            dim=-1
        )
        edge_x_end_idx = edge_x_start_idx + th.tensor([1, 0, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)

        edge_x_adj_cube_idx_0 = edge_x_start_idx - th.tensor([0, 1, 1], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        edge_x_adj_cube_idx_1 = edge_x_start_idx - th.tensor([0, 1, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        edge_x_adj_cube_idx_2 = edge_x_start_idx - th.tensor([0, 0, 1], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        edge_x_adj_cube_idx_3 = edge_x_start_idx - th.tensor([0, 0, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)

        edge_x_tet_0 = th.stack(
            [edge_x_start_idx,
            edge_x_end_idx,
            edge_x_adj_cube_idx_0,
            edge_x_adj_cube_idx_1],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_x_tet_1 = th.stack(
            [edge_x_start_idx,
            edge_x_end_idx,
            edge_x_adj_cube_idx_0,
            edge_x_adj_cube_idx_2],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_x_tet_2 = th.stack(
            [edge_x_start_idx,
            edge_x_end_idx,
            edge_x_adj_cube_idx_1,
            edge_x_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_x_tet_3 = th.stack(
            [edge_x_start_idx,
            edge_x_end_idx,
            edge_x_adj_cube_idx_2,
            edge_x_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_x_tets = th.cat([edge_x_tet_0, edge_x_tet_1, edge_x_tet_2, edge_x_tet_3], dim=0)
        edge_x_tets_valid = (edge_x_tets[:, 2:] >= 0) & \
            (edge_x_tets[:, 2:] < th.tensor([num_grid_x, num_grid_y, num_grid_z], device=device, dtype=th.long).reshape(1, 1, 3))
        edge_x_tets_valid = edge_x_tets_valid.all(dim=-1).all(dim=-1)

        edge_x_tets = edge_x_tets[edge_x_tets_valid]

        # change to flattened indices
        f_edge_x_tets_0 = edge_x_tets[:, 0, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_x_tets[:, 0, 1] * (num_grid_z + 1) + edge_x_tets[:, 0, 2]
        f_edge_x_tets_1 = edge_x_tets[:, 1, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_x_tets[:, 1, 1] * (num_grid_z + 1) + edge_x_tets[:, 1, 2]
        f_edge_x_tets_2 = edge_x_tets[:, 2, 0] * ((num_grid_y) * (num_grid_z)) + edge_x_tets[:, 2, 1] * (num_grid_z) + edge_x_tets[:, 2, 2] + len_cube_lattice_verts
        f_edge_x_tets_3 = edge_x_tets[:, 3, 0] * ((num_grid_y) * (num_grid_z)) + edge_x_tets[:, 3, 1] * (num_grid_z) + edge_x_tets[:, 3, 2] + len_cube_lattice_verts
        f_edge_x_tets = th.stack([f_edge_x_tets_0, f_edge_x_tets_1, f_edge_x_tets_2, f_edge_x_tets_3], dim=-1)  # [num_edges, 4]

        return f_edge_x_tets
    
    def _y_edge_tets(self, num_grid_x, num_grid_y, num_grid_z, device, len_cube_lattice_verts):
        edge_y_start_idx = th.stack(
            th.meshgrid(
                th.arange(0, num_grid_x + 1, device=device, dtype=th.long),
                th.arange(0, num_grid_y, device=device, dtype=th.long),
                th.arange(0, num_grid_z + 1, device=device, dtype=th.long),
                indexing='ij'
            ),
            dim=-1
        )
        edge_y_end_idx = edge_y_start_idx + th.tensor([0, 1, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)

        edge_y_adj_cube_idx_0 = edge_y_start_idx - th.tensor([1, 0, 1], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        edge_y_adj_cube_idx_1 = edge_y_start_idx - th.tensor([1, 0, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        edge_y_adj_cube_idx_2 = edge_y_start_idx - th.tensor([0, 0, 1], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        edge_y_adj_cube_idx_3 = edge_y_start_idx - th.tensor([0, 0, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)

        edge_y_tet_0 = th.stack(
            [edge_y_start_idx,
            edge_y_end_idx,
            edge_y_adj_cube_idx_0,
            edge_y_adj_cube_idx_1],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_y_tet_1 = th.stack(
            [edge_y_start_idx,
            edge_y_end_idx,
            edge_y_adj_cube_idx_0,
            edge_y_adj_cube_idx_2],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_y_tet_2 = th.stack(
            [edge_y_start_idx,
            edge_y_end_idx,
            edge_y_adj_cube_idx_1,
            edge_y_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_y_tet_3 = th.stack(
            [edge_y_start_idx,
            edge_y_end_idx,
            edge_y_adj_cube_idx_2,
            edge_y_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_y_tets = th.cat([edge_y_tet_0, edge_y_tet_1, edge_y_tet_2, edge_y_tet_3], dim=0)
        edge_y_tets_valid = (edge_y_tets[:, 2:] >= 0) & \
            (edge_y_tets[:, 2:] < th.tensor([num_grid_x, num_grid_y, num_grid_z], device=device, dtype=th.long).reshape(1, 1, 3))
        edge_y_tets_valid = edge_y_tets_valid.all(dim=-1).all(dim=-1)

        edge_y_tets = edge_y_tets[edge_y_tets_valid]

        # change to flattened indices
        f_edge_y_tets_0 = edge_y_tets[:, 0, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_y_tets[:, 0, 1] * (num_grid_z + 1) + edge_y_tets[:, 0, 2]
        f_edge_y_tets_1 = edge_y_tets[:, 1, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_y_tets[:, 1, 1] * (num_grid_z + 1) + edge_y_tets[:, 1, 2]
        f_edge_y_tets_2 = edge_y_tets[:, 2, 0] * ((num_grid_y) * (num_grid_z)) + edge_y_tets[:, 2, 1] * (num_grid_z) + edge_y_tets[:, 2, 2] + len_cube_lattice_verts
        f_edge_y_tets_3 = edge_y_tets[:, 3, 0] * ((num_grid_y) * (num_grid_z)) + edge_y_tets[:, 3, 1] * (num_grid_z) + edge_y_tets[:, 3, 2] + len_cube_lattice_verts

        f_edge_y_tets = th.stack([f_edge_y_tets_0, f_edge_y_tets_1, f_edge_y_tets_2, f_edge_y_tets_3], dim=-1)  # [num_edges, 4]

        return f_edge_y_tets

    def _z_edge_tets(self, num_grid_x, num_grid_y, num_grid_z, device, len_cube_lattice_verts):
        edge_z_start_idx = th.stack(
            th.meshgrid(
                th.arange(0, num_grid_x + 1, device=device, dtype=th.long),
                th.arange(0, num_grid_y + 1, device=device, dtype=th.long),
                th.arange(0, num_grid_z, device=device, dtype=th.long),
                indexing='ij'
            ),
            dim=-1
        )
        edge_z_end_idx = edge_z_start_idx + th.tensor([0, 0, 1], device=device, dtype=th.long).reshape(1, 1, 1, 3)

        edge_z_adj_cube_idx_0 = edge_z_start_idx - th.tensor([1, 1, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        edge_z_adj_cube_idx_1 = edge_z_start_idx - th.tensor([1, 0, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        edge_z_adj_cube_idx_2 = edge_z_start_idx - th.tensor([0, 1, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        edge_z_adj_cube_idx_3 = edge_z_start_idx - th.tensor([0, 0, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)

        edge_z_tet_0 = th.stack(
            [edge_z_start_idx,
            edge_z_end_idx,
            edge_z_adj_cube_idx_0,
            edge_z_adj_cube_idx_1],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_z_tet_1 = th.stack(
            [edge_z_start_idx,
            edge_z_end_idx,
            edge_z_adj_cube_idx_0,
            edge_z_adj_cube_idx_2],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_z_tet_2 = th.stack(
            [edge_z_start_idx,
            edge_z_end_idx,
            edge_z_adj_cube_idx_1,
            edge_z_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_z_tet_3 = th.stack(
            [edge_z_start_idx,
            edge_z_end_idx,
            edge_z_adj_cube_idx_2,
            edge_z_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_z_tets = th.cat([edge_z_tet_0, edge_z_tet_1, edge_z_tet_2, edge_z_tet_3], dim=0)

        edge_z_tets_valid = (edge_z_tets[:, 2:] >= 0) & \
            (edge_z_tets[:, 2:] < th.tensor([num_grid_x, num_grid_y, num_grid_z], device=device, dtype=th.long).reshape(1, 1, 3))
        edge_z_tets_valid = edge_z_tets_valid.all(dim=-1).all(dim=-1)

        edge_z_tets = edge_z_tets[edge_z_tets_valid]

        # change to flattened indices
        f_edge_z_tets_0 = edge_z_tets[:, 0, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_z_tets[:, 0, 1] * (num_grid_z + 1) + edge_z_tets[:, 0, 2]
        f_edge_z_tets_1 = edge_z_tets[:, 1, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_z_tets[:, 1, 1] * (num_grid_z + 1) + edge_z_tets[:, 1, 2]
        f_edge_z_tets_2 = edge_z_tets[:, 2, 0] * ((num_grid_y) * (num_grid_z)) + edge_z_tets[:, 2, 1] * (num_grid_z) + edge_z_tets[:, 2, 2] + len_cube_lattice_verts
        f_edge_z_tets_3 = edge_z_tets[:, 3, 0] * ((num_grid_y) * (num_grid_z)) + edge_z_tets[:, 3, 1] * (num_grid_z) + edge_z_tets[:, 3, 2] + len_cube_lattice_verts
        
        f_edge_z_tets = th.stack([f_edge_z_tets_0, f_edge_z_tets_1, f_edge_z_tets_2, f_edge_z_tets_3], dim=-1)

        return f_edge_z_tets

    def _x_edge_tets_real(self, num_grid_x, num_grid_y, num_grid_z, device, len_cube_lattice_verts, verts_real):
        # select starting vertices with real 1.0
        edge_x_start_idx = th.stack(
            th.meshgrid(
                th.arange(0, num_grid_x, device=device, dtype=th.long),
                th.arange(0, num_grid_y + 1, device=device, dtype=th.long),
                th.arange(0, num_grid_z + 1, device=device, dtype=th.long),
                indexing='ij'
            ),
            dim=-1
        ).view(-1, 3)
        f_edge_x_start_idx = \
            edge_x_start_idx[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + \
            edge_x_start_idx[:, 1] * (num_grid_z + 1) + \
            edge_x_start_idx[:, 2]
        edge_x_start_idx = edge_x_start_idx[verts_real[f_edge_x_start_idx]]

        edge_x_end_idx = edge_x_start_idx + th.tensor([1, 0, 0], device=device, dtype=th.long).reshape(1, 3)

        edge_x_adj_cube_idx_0 = edge_x_start_idx - th.tensor([0, 1, 1], device=device, dtype=th.long).reshape(1, 3)
        edge_x_adj_cube_idx_1 = edge_x_start_idx - th.tensor([0, 1, 0], device=device, dtype=th.long).reshape(1, 3)
        edge_x_adj_cube_idx_2 = edge_x_start_idx - th.tensor([0, 0, 1], device=device, dtype=th.long).reshape(1, 3)
        edge_x_adj_cube_idx_3 = edge_x_start_idx - th.tensor([0, 0, 0], device=device, dtype=th.long).reshape(1, 3)

        edge_x_tet_0 = th.stack(
            [edge_x_start_idx,
            edge_x_end_idx,
            edge_x_adj_cube_idx_0,
            edge_x_adj_cube_idx_1],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_x_tet_1 = th.stack(
            [edge_x_start_idx,
            edge_x_end_idx,
            edge_x_adj_cube_idx_0,
            edge_x_adj_cube_idx_2],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_x_tet_2 = th.stack(
            [edge_x_start_idx,
            edge_x_end_idx,
            edge_x_adj_cube_idx_1,
            edge_x_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_x_tet_3 = th.stack(
            [edge_x_start_idx,
            edge_x_end_idx,
            edge_x_adj_cube_idx_2,
            edge_x_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_x_tets = th.cat([edge_x_tet_0, edge_x_tet_1, edge_x_tet_2, edge_x_tet_3], dim=0)
        edge_x_tets_valid = (edge_x_tets[:, 2:] >= 0) & \
            (edge_x_tets[:, 2:] < th.tensor([num_grid_x, num_grid_y, num_grid_z], device=device, dtype=th.long).reshape(1, 1, 3))
        edge_x_tets_valid = edge_x_tets_valid.all(dim=-1).all(dim=-1)

        edge_x_tets = edge_x_tets[edge_x_tets_valid]

        # change to flattened indices
        f_edge_x_tets_0 = edge_x_tets[:, 0, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_x_tets[:, 0, 1] * (num_grid_z + 1) + edge_x_tets[:, 0, 2]
        f_edge_x_tets_1 = edge_x_tets[:, 1, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_x_tets[:, 1, 1] * (num_grid_z + 1) + edge_x_tets[:, 1, 2]
        f_edge_x_tets_2 = edge_x_tets[:, 2, 0] * ((num_grid_y) * (num_grid_z)) + edge_x_tets[:, 2, 1] * (num_grid_z) + edge_x_tets[:, 2, 2] + len_cube_lattice_verts
        f_edge_x_tets_3 = edge_x_tets[:, 3, 0] * ((num_grid_y) * (num_grid_z)) + edge_x_tets[:, 3, 1] * (num_grid_z) + edge_x_tets[:, 3, 2] + len_cube_lattice_verts
        f_edge_x_tets = th.stack([f_edge_x_tets_0, f_edge_x_tets_1, f_edge_x_tets_2, f_edge_x_tets_3], dim=-1)  # [num_edges, 4]

        # select based on reals
        f_edge_x_tets_valid = verts_real[f_edge_x_tets].all(dim=-1)
        f_edge_x_tets = f_edge_x_tets[f_edge_x_tets_valid]

        return f_edge_x_tets
    
    def _y_edge_tets_real(self, num_grid_x, num_grid_y, num_grid_z, device, len_cube_lattice_verts, verts_real):
        edge_y_start_idx = th.stack(
            th.meshgrid(
                th.arange(0, num_grid_x + 1, device=device, dtype=th.long),
                th.arange(0, num_grid_y, device=device, dtype=th.long),
                th.arange(0, num_grid_z + 1, device=device, dtype=th.long),
                indexing='ij'
            ),
            dim=-1
        ).view(-1, 3)
        f_edge_y_start_idx = \
            edge_y_start_idx[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + \
            edge_y_start_idx[:, 1] * (num_grid_z + 1) + \
            edge_y_start_idx[:, 2]
        edge_y_start_idx = edge_y_start_idx[verts_real[f_edge_y_start_idx]]
        
        edge_y_end_idx = edge_y_start_idx + th.tensor([0, 1, 0], device=device, dtype=th.long).reshape(1, 3)

        edge_y_adj_cube_idx_0 = edge_y_start_idx - th.tensor([1, 0, 1], device=device, dtype=th.long).reshape(1, 3)
        edge_y_adj_cube_idx_1 = edge_y_start_idx - th.tensor([1, 0, 0], device=device, dtype=th.long).reshape(1, 3)
        edge_y_adj_cube_idx_2 = edge_y_start_idx - th.tensor([0, 0, 1], device=device, dtype=th.long).reshape(1, 3)
        edge_y_adj_cube_idx_3 = edge_y_start_idx - th.tensor([0, 0, 0], device=device, dtype=th.long).reshape(1, 3)

        edge_y_tet_0 = th.stack(
            [edge_y_start_idx,
            edge_y_end_idx,
            edge_y_adj_cube_idx_0,
            edge_y_adj_cube_idx_1],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_y_tet_1 = th.stack(
            [edge_y_start_idx,
            edge_y_end_idx,
            edge_y_adj_cube_idx_0,
            edge_y_adj_cube_idx_2],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_y_tet_2 = th.stack(
            [edge_y_start_idx,
            edge_y_end_idx,
            edge_y_adj_cube_idx_1,
            edge_y_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_y_tet_3 = th.stack(
            [edge_y_start_idx,
            edge_y_end_idx,
            edge_y_adj_cube_idx_2,
            edge_y_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_y_tets = th.cat([edge_y_tet_0, edge_y_tet_1, edge_y_tet_2, edge_y_tet_3], dim=0)
        edge_y_tets_valid = (edge_y_tets[:, 2:] >= 0) & \
            (edge_y_tets[:, 2:] < th.tensor([num_grid_x, num_grid_y, num_grid_z], device=device, dtype=th.long).reshape(1, 1, 3))
        edge_y_tets_valid = edge_y_tets_valid.all(dim=-1).all(dim=-1)

        edge_y_tets = edge_y_tets[edge_y_tets_valid]

        # change to flattened indices
        f_edge_y_tets_0 = edge_y_tets[:, 0, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_y_tets[:, 0, 1] * (num_grid_z + 1) + edge_y_tets[:, 0, 2]
        f_edge_y_tets_1 = edge_y_tets[:, 1, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_y_tets[:, 1, 1] * (num_grid_z + 1) + edge_y_tets[:, 1, 2]
        f_edge_y_tets_2 = edge_y_tets[:, 2, 0] * ((num_grid_y) * (num_grid_z)) + edge_y_tets[:, 2, 1] * (num_grid_z) + edge_y_tets[:, 2, 2] + len_cube_lattice_verts
        f_edge_y_tets_3 = edge_y_tets[:, 3, 0] * ((num_grid_y) * (num_grid_z)) + edge_y_tets[:, 3, 1] * (num_grid_z) + edge_y_tets[:, 3, 2] + len_cube_lattice_verts

        f_edge_y_tets = th.stack([f_edge_y_tets_0, f_edge_y_tets_1, f_edge_y_tets_2, f_edge_y_tets_3], dim=-1)  # [num_edges, 4]

        # select based on reals
        f_edge_y_tets_valid = verts_real[f_edge_y_tets].all(dim=-1)
        f_edge_y_tets = f_edge_y_tets[f_edge_y_tets_valid]

        return f_edge_y_tets

    def _z_edge_tets_real(self, num_grid_x, num_grid_y, num_grid_z, device, len_cube_lattice_verts, verts_real):
        edge_z_start_idx = th.stack(
            th.meshgrid(
                th.arange(0, num_grid_x + 1, device=device, dtype=th.long),
                th.arange(0, num_grid_y + 1, device=device, dtype=th.long),
                th.arange(0, num_grid_z, device=device, dtype=th.long),
                indexing='ij'
            ),
            dim=-1
        ).view(-1, 3)
        f_edge_z_start_idx = \
            edge_z_start_idx[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + \
            edge_z_start_idx[:, 1] * (num_grid_z + 1) + \
            edge_z_start_idx[:, 2]
        edge_z_start_idx = edge_z_start_idx[verts_real[f_edge_z_start_idx]]

        edge_z_end_idx = edge_z_start_idx + th.tensor([0, 0, 1], device=device, dtype=th.long).reshape(1, 3)

        edge_z_adj_cube_idx_0 = edge_z_start_idx - th.tensor([1, 1, 0], device=device, dtype=th.long).reshape(1, 3)
        edge_z_adj_cube_idx_1 = edge_z_start_idx - th.tensor([1, 0, 0], device=device, dtype=th.long).reshape(1, 3)
        edge_z_adj_cube_idx_2 = edge_z_start_idx - th.tensor([0, 1, 0], device=device, dtype=th.long).reshape(1, 3)
        edge_z_adj_cube_idx_3 = edge_z_start_idx - th.tensor([0, 0, 0], device=device, dtype=th.long).reshape(1, 3)

        edge_z_tet_0 = th.stack(
            [edge_z_start_idx,
            edge_z_end_idx,
            edge_z_adj_cube_idx_0,
            edge_z_adj_cube_idx_1],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_z_tet_1 = th.stack(
            [edge_z_start_idx,
            edge_z_end_idx,
            edge_z_adj_cube_idx_0,
            edge_z_adj_cube_idx_2],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_z_tet_2 = th.stack(
            [edge_z_start_idx,
            edge_z_end_idx,
            edge_z_adj_cube_idx_1,
            edge_z_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_z_tet_3 = th.stack(
            [edge_z_start_idx,
            edge_z_end_idx,
            edge_z_adj_cube_idx_2,
            edge_z_adj_cube_idx_3],
            dim=-2
        ).view(-1, 4, 3)    # [num_edges, 4, 3]

        edge_z_tets = th.cat([edge_z_tet_0, edge_z_tet_1, edge_z_tet_2, edge_z_tet_3], dim=0)

        edge_z_tets_valid = (edge_z_tets[:, 2:] >= 0) & \
            (edge_z_tets[:, 2:] < th.tensor([num_grid_x, num_grid_y, num_grid_z], device=device, dtype=th.long).reshape(1, 1, 3))
        edge_z_tets_valid = edge_z_tets_valid.all(dim=-1).all(dim=-1)

        edge_z_tets = edge_z_tets[edge_z_tets_valid]

        # change to flattened indices
        f_edge_z_tets_0 = edge_z_tets[:, 0, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_z_tets[:, 0, 1] * (num_grid_z + 1) + edge_z_tets[:, 0, 2]
        f_edge_z_tets_1 = edge_z_tets[:, 1, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + edge_z_tets[:, 1, 1] * (num_grid_z + 1) + edge_z_tets[:, 1, 2]
        f_edge_z_tets_2 = edge_z_tets[:, 2, 0] * ((num_grid_y) * (num_grid_z)) + edge_z_tets[:, 2, 1] * (num_grid_z) + edge_z_tets[:, 2, 2] + len_cube_lattice_verts
        f_edge_z_tets_3 = edge_z_tets[:, 3, 0] * ((num_grid_y) * (num_grid_z)) + edge_z_tets[:, 3, 1] * (num_grid_z) + edge_z_tets[:, 3, 2] + len_cube_lattice_verts
        
        f_edge_z_tets = th.stack([f_edge_z_tets_0, f_edge_z_tets_1, f_edge_z_tets_2, f_edge_z_tets_3], dim=-1)

        # select based on reals
        f_edge_z_tets_valid = verts_real[f_edge_z_tets].all(dim=-1)
        f_edge_z_tets = f_edge_z_tets[f_edge_z_tets_valid]

        return f_edge_z_tets

    def init(self, domain_min: tuple, domain_max: tuple, grid_size: float):
        '''
        Initialize [self.verts] so that this grid covers the given rectangular domain.
        '''
        self.grid_size = grid_size

        device = self.device

        '''
        1. Set verts
        '''

        self.domain_min = domain_min
        self.domain_max = domain_max

        x_min, y_min, z_min = domain_min
        x_max, y_max, z_max = domain_max

        x_length = x_max - x_min
        y_length = y_max - y_min
        z_length = z_max - z_min

        # compute cube grid size
        num_grid_x = int(x_length / grid_size) + 1
        num_grid_y = int(y_length / grid_size) + 1
        num_grid_z = int(z_length / grid_size) + 1

        self.num_grid_x = num_grid_x
        self.num_grid_y = num_grid_y
        self.num_grid_z = num_grid_z

        # compute cube grid lattice vertices
        cube_lattice_verts = th.stack(
            th.meshgrid(
                # +1 for the final group
                th.arange(0, num_grid_x + 1, device=device, dtype=th.float32),
                th.arange(0, num_grid_y + 1, device=device, dtype=th.float32),
                th.arange(0, num_grid_z + 1, device=device, dtype=th.float32),
                indexing='ij'
            ),
            dim=-1
        )
        cube_lattice_verts = cube_lattice_verts * grid_size
        cube_lattice_verts[..., 0] = cube_lattice_verts[..., 0] + x_min
        cube_lattice_verts[..., 1] = cube_lattice_verts[..., 1] + y_min
        cube_lattice_verts[..., 2] = cube_lattice_verts[..., 2] + z_min

        f_cube_lattice_verts = cube_lattice_verts.view(-1, 3)
        len_cube_lattice_verts = f_cube_lattice_verts.shape[0]

        # compute cube grid center vertices
        cube_center_verts = cube_lattice_verts[:-1, :-1, :-1] + (grid_size * 0.5)

        f_cube_center_verts = cube_center_verts.view(-1, 3)
        len_cube_center_verts = f_cube_center_verts.shape[0]

        f_grid_verts = th.cat([f_cube_lattice_verts, f_cube_center_verts], dim=0)

        self.verts = f_grid_verts

        del cube_lattice_verts, cube_center_verts, f_cube_lattice_verts, f_cube_center_verts

        '''
        2. Set tet_idx
        '''
        ### compute edge-wise

        ### edges along x-axis
        f_edge_x_tets = self._x_edge_tets(num_grid_x, num_grid_y, num_grid_z, device, len_cube_lattice_verts)
        
        ### edges along y-axis
        f_edge_y_tets = self._y_edge_tets(num_grid_x, num_grid_y, num_grid_z, device, len_cube_lattice_verts)
        
        ### edges along z-axis
        f_edge_z_tets = self._z_edge_tets(num_grid_x, num_grid_y, num_grid_z, device, len_cube_lattice_verts)
        
        ### combine all edges
        f_tets = th.cat([f_edge_x_tets, f_edge_y_tets, f_edge_z_tets], dim=0)

        self.tet_idx = th.sort(f_tets, dim=-1)[0]
        self.tet_idx = th.unique(self.tet_idx, dim=0)

        del f_edge_x_tets, f_edge_y_tets, f_edge_z_tets, f_tets

        # refresh cuda memory
        if device != 'cpu':
            th.cuda.empty_cache()

        '''
        3. Set tri_idx
        '''
        tri_combs = [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]
        tri_idx = self.tet_idx[:, tri_combs].view(-1, 3)
        # tri_idx = th.sort(tri_idx, dim=-1)[0]
        tri_idx = th.unique(tri_idx, dim=0)

        self.tri_idx = tri_idx

        # apex circumball dist
        self.apex_circumball_dist = ((np.sqrt(34) - np.sqrt(18)) / 8.0) * self.grid_size

    def subdivide(self, div: int, preal: th.Tensor):
        '''
        Subdivide the grid.
        Only consider the cells which satisfy [preal] condition.
        '''
        assert div > 1, 'div should be greater than 1'
        assert len(preal) == self.verts.shape[0], 'preal should have the same number of vertices as the grid'

        device = self.device
        n_grid = TetGrid(self.device)
        n_grid_size = self.grid_size / div

        n_grid.grid_size = n_grid_size

        '''
        1. Set verts
        '''

        n_grid.domain_min = self.domain_min
        n_grid.domain_max = self.domain_max

        x_min, y_min, z_min = self.domain_min
        x_max, y_max, z_max = self.domain_max

        x_length = x_max - x_min
        y_length = y_max - y_min
        z_length = z_max - z_min

        # compute original cube grid size
        num_grid_x = int(x_length / self.grid_size) + 1
        num_grid_y = int(y_length / self.grid_size) + 1
        num_grid_z = int(z_length / self.grid_size) + 1

        '''
        1. Find real cells in the original grid
        '''
        ### for each cell in the original grid, find which verts are included
        cell_vert_0 = th.stack(
            th.meshgrid(
                th.arange(0, num_grid_x, device=device, dtype=th.long),
                th.arange(0, num_grid_y, device=device, dtype=th.long),
                th.arange(0, num_grid_z, device=device, dtype=th.long),
                indexing='ij'
            ),
            dim=-1
        )
        cell_vert_1 = cell_vert_0 + th.tensor([1, 0, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        cell_vert_2 = cell_vert_0 + th.tensor([0, 1, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        cell_vert_3 = cell_vert_0 + th.tensor([0, 0, 1], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        cell_vert_4 = cell_vert_0 + th.tensor([1, 1, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        cell_vert_5 = cell_vert_0 + th.tensor([1, 0, 1], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        cell_vert_6 = cell_vert_0 + th.tensor([0, 1, 1], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        cell_vert_7 = cell_vert_0 + th.tensor([1, 1, 1], device=device, dtype=th.long).reshape(1, 1, 1, 3)
        cell_vert_8 = cell_vert_0 + th.tensor([0, 0, 0], device=device, dtype=th.long).reshape(1, 1, 1, 3)

        cell_vert_0 = cell_vert_0.view(-1, 3)
        cell_vert_1 = cell_vert_1.view(-1, 3)
        cell_vert_2 = cell_vert_2.view(-1, 3)
        cell_vert_3 = cell_vert_3.view(-1, 3)
        cell_vert_4 = cell_vert_4.view(-1, 3)
        cell_vert_5 = cell_vert_5.view(-1, 3)
        cell_vert_6 = cell_vert_6.view(-1, 3)
        cell_vert_7 = cell_vert_7.view(-1, 3)
        cell_vert_8 = cell_vert_8.view(-1, 3)

        cell_vert_0 = cell_vert_0[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + cell_vert_0[:, 1] * (num_grid_z + 1) + cell_vert_0[:, 2]
        cell_vert_1 = cell_vert_1[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + cell_vert_1[:, 1] * (num_grid_z + 1) + cell_vert_1[:, 2]
        cell_vert_2 = cell_vert_2[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + cell_vert_2[:, 1] * (num_grid_z + 1) + cell_vert_2[:, 2]
        cell_vert_3 = cell_vert_3[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + cell_vert_3[:, 1] * (num_grid_z + 1) + cell_vert_3[:, 2]
        cell_vert_4 = cell_vert_4[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + cell_vert_4[:, 1] * (num_grid_z + 1) + cell_vert_4[:, 2]
        cell_vert_5 = cell_vert_5[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + cell_vert_5[:, 1] * (num_grid_z + 1) + cell_vert_5[:, 2]
        cell_vert_6 = cell_vert_6[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + cell_vert_6[:, 1] * (num_grid_z + 1) + cell_vert_6[:, 2]
        cell_vert_7 = cell_vert_7[:, 0] * ((num_grid_y + 1) * (num_grid_z + 1)) + cell_vert_7[:, 1] * (num_grid_z + 1) + cell_vert_7[:, 2]
        cell_vert_8 = cell_vert_8[:, 0] * ((num_grid_y) * (num_grid_z)) + cell_vert_8[:, 1] * (num_grid_z) + cell_vert_8[:, 2] + ((num_grid_x + 1) * (num_grid_y + 1) * (num_grid_z + 1))
        cell_verts = th.stack([cell_vert_0, cell_vert_1, cell_vert_2, cell_vert_3, cell_vert_4, cell_vert_5, cell_vert_6, cell_vert_7, cell_vert_8], dim=-1)
        
        ### for each cell, determine it is real or not
        cell_real = preal[cell_verts].any(dim=-1)
        real_cell_idx = th.nonzero(cell_real, as_tuple=False).view(-1)
        real_cell_idx_x = real_cell_idx // (num_grid_y * num_grid_z)
        real_cell_idx_y = (real_cell_idx % (num_grid_y * num_grid_z)) // num_grid_z
        real_cell_idx_z = real_cell_idx % num_grid_z
        real_cell_idx = th.stack([real_cell_idx_x, real_cell_idx_y, real_cell_idx_z], dim=-1)

        '''
        2. Find vertices in the new grid that are included in the real cells in the original grid
        '''
        n_lattice_add = th.stack(
            th.meshgrid(
                th.arange(0, div + 1, device=device, dtype=th.long),
                th.arange(0, div + 1, device=device, dtype=th.long),
                th.arange(0, div + 1, device=device, dtype=th.long),
                indexing='ij'
            ),
            dim=-1
        ).view(-1, 3)
        n_real_lattice_vert_idx = real_cell_idx.view(-1, 1, 3) * div + n_lattice_add
        n_real_lattice_vert_idx = n_real_lattice_vert_idx.view(-1, 3)

        n_center_add = th.stack(
            th.meshgrid(
                th.arange(0, div, device=device, dtype=th.long),
                th.arange(0, div, device=device, dtype=th.long),
                th.arange(0, div, device=device, dtype=th.long),
                indexing='ij'
            ),
            dim=-1
        ).view(-1, 3)
        n_real_center_vert_idx = real_cell_idx.view(-1, 1, 3) * div + n_center_add 
        n_real_center_vert_idx = n_real_center_vert_idx.view(-1, 3)

        '''
        3. Compute grid info
        '''
        n_num_grid_x = num_grid_x * div
        n_num_grid_y = num_grid_y * div
        n_num_grid_z = num_grid_z * div

        # compute cube grid lattice vertices
        cube_lattice_verts = th.stack(
            th.meshgrid(
                # +1 for the final group
                th.arange(0, n_num_grid_x + 1, device=device, dtype=th.float32),
                th.arange(0, n_num_grid_y + 1, device=device, dtype=th.float32),
                th.arange(0, n_num_grid_z + 1, device=device, dtype=th.float32),
                indexing='ij'
            ),
            dim=-1
        )
        cube_lattice_verts = cube_lattice_verts * n_grid_size
        cube_lattice_verts[..., 0] = cube_lattice_verts[..., 0] + x_min
        cube_lattice_verts[..., 1] = cube_lattice_verts[..., 1] + y_min
        cube_lattice_verts[..., 2] = cube_lattice_verts[..., 2] + z_min

        f_cube_lattice_verts = cube_lattice_verts.view(-1, 3)
        len_cube_lattice_verts = f_cube_lattice_verts.shape[0]

        # compute cube grid center vertices
        cube_center_verts = cube_lattice_verts[:-1, :-1, :-1] + (n_grid_size * 0.5)

        f_cube_center_verts = cube_center_verts.view(-1, 3)
        len_cube_center_verts = f_cube_center_verts.shape[0]

        f_grid_verts = th.cat([f_cube_lattice_verts, f_cube_center_verts], dim=0)

        n_grid.verts = f_grid_verts

        del cube_lattice_verts, cube_center_verts, f_cube_lattice_verts, f_cube_center_verts

        ### flatten the indices of real vertices
        f_n_real_lattice_vert_idx = \
            n_real_lattice_vert_idx[:, 0] * ((n_num_grid_y + 1) * (n_num_grid_z + 1)) + \
            n_real_lattice_vert_idx[:, 1] * (n_num_grid_z + 1) + \
            n_real_lattice_vert_idx[:, 2]
        f_n_real_center_vert_idx = \
            n_real_center_vert_idx[:, 0] * ((n_num_grid_y) * (n_num_grid_z)) + \
            n_real_center_vert_idx[:, 1] * (n_num_grid_z) + \
            n_real_center_vert_idx[:, 2] + ((n_num_grid_x + 1) * (n_num_grid_y + 1) * (n_num_grid_z + 1))
        f_n_real_vert_idx = th.cat([f_n_real_lattice_vert_idx, f_n_real_center_vert_idx], dim=0)
        
        n_verts_real = th.zeros_like(f_grid_verts[:, 0], dtype=th.bool)
        n_verts_real[f_n_real_vert_idx] = True

        ### set tet_idx
        
        f_edge_x_tets = self._x_edge_tets_real(n_num_grid_x, n_num_grid_y, n_num_grid_z, device, len_cube_lattice_verts, n_verts_real)
        f_edge_y_tets = self._y_edge_tets_real(n_num_grid_x, n_num_grid_y, n_num_grid_z, device, len_cube_lattice_verts, n_verts_real)
        f_edge_z_tets = self._z_edge_tets_real(n_num_grid_x, n_num_grid_y, n_num_grid_z, device, len_cube_lattice_verts, n_verts_real)
        
        f_tets = th.cat([f_edge_x_tets, f_edge_y_tets, f_edge_z_tets], dim=0)

        n_grid.tet_idx = th.sort(f_tets, dim=-1)[0]
        n_grid.tet_idx = th.unique(n_grid.tet_idx, dim=0)

        del f_edge_x_tets, f_edge_y_tets, f_edge_z_tets, f_tets

        # refresh cuda memory
        if device != 'cpu':
            th.cuda.empty_cache()

        '''
        4. Set tri_idx
        '''
        tri_combs = [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]
        tri_idx = n_grid.tet_idx[:, tri_combs].view(-1, 3)
        # tri_idx = th.sort(tri_idx, dim=-1)[0]
        tri_idx = th.unique(tri_idx, dim=0)

        n_grid.tri_idx = tri_idx

        return n_grid

class EffTetGrid:
    '''
    Efficient TetGrid.

    Do not store every vertex and face in the grid.
    Only store those that we use during optimization.
    '''

    def __init__(self, device):
        self.device = device

        self.verts = None
        self.tri_idx = None

        self.grid_size = None
        self.verts_grid_idx = None      # grid index of each vertex

        self.domain_min = None
        self.domain_max = None

        self.num_grid_x = None
        self.num_grid_y = None
        self.num_grid_z = None

        self.apex_circumball_dist = None

    @staticmethod
    def extract(tgrid: TetGrid, preal: th.Tensor):
        '''
        For given tet grid and real values of vertices in the grid,
        extract an efficient tet grid.
        '''
        device = tgrid.device

        etgrid = EffTetGrid(device)
        etgrid.grid_size = tgrid.grid_size
        
        etgrid.domain_min = tgrid.domain_min
        etgrid.domain_max = tgrid.domain_max

        etgrid.num_grid_x = tgrid.num_grid_x
        etgrid.num_grid_y = tgrid.num_grid_y
        etgrid.num_grid_z = tgrid.num_grid_z

        assert len(preal) == tgrid.verts.shape[0], 'preal should have the same number of vertices as the grid'
        orig_tris_reality = preal[tgrid.tri_idx].all(dim=-1)
        orig_real_tris = tgrid.tri_idx[orig_tris_reality]

        verts_on_orig_real_tris, inv_verts_on_orig_real_tris = th.unique(orig_real_tris, return_inverse=True)

        n_verts = tgrid.verts[verts_on_orig_real_tris]
        n_tris = inv_verts_on_orig_real_tris

        etgrid.verts = n_verts
        etgrid.tri_idx = n_tris
        etgrid.verts_grid_idx = verts_on_orig_real_tris
        etgrid.apex_circumball_dist = tgrid.apex_circumball_dist

        return etgrid

    def subdivide(self, preal: th.Tensor):
        '''
        Subdivide this efficient tet grid and return a new efficient tet grid.
        Use [preal] to identify which vertices and faces should be subdivided.
        '''

        etgrid = EffTetGrid(self.device)
        etgrid.grid_size = self.grid_size * 0.5

        etgrid.domain_min = self.domain_min
        etgrid.domain_max = self.domain_max

        etgrid.num_grid_x = self.num_grid_x * 2
        etgrid.num_grid_y = self.num_grid_y * 2
        etgrid.num_grid_z = self.num_grid_z * 2

        etgrid.apex_circumball_dist = self.apex_circumball_dist * 0.5

        assert len(preal) == self.verts.shape[0], 'preal should have the same number of vertices as the grid'

        '''
        1. Find grid cells to subdivide based on [preal]
        '''
        num_lattice_verts = (self.num_grid_x + 1) * (self.num_grid_y + 1) * (self.num_grid_z + 1)
        # num_central_verts = (self.num_grid_x * self.num_grid_y * self.num_grid_z)

        real_verts_idx = self.verts_grid_idx[preal == 1.0]
        real_lattice_verts = real_verts_idx[real_verts_idx < num_lattice_verts]
        real_central_verts = real_verts_idx[real_verts_idx >= num_lattice_verts] - num_lattice_verts

        real_lattice_verts_x = real_lattice_verts // ((self.num_grid_y + 1) * (self.num_grid_z + 1))
        real_lattice_verts_y = (real_lattice_verts % ((self.num_grid_y + 1) * (self.num_grid_z + 1))) // (self.num_grid_z + 1)
        real_lattice_verts_z = real_lattice_verts % (self.num_grid_z + 1)

        real_central_verts_x = real_central_verts // (self.num_grid_y * self.num_grid_z)
        real_central_verts_y = (real_central_verts % (self.num_grid_y * self.num_grid_z)) // self.num_grid_z
        real_central_verts_z = real_central_verts % self.num_grid_z

        real_lattice_verts_xyz = th.stack([real_lattice_verts_x, real_lattice_verts_y, real_lattice_verts_z], dim=-1)
        real_central_verts_xyz = th.stack([real_central_verts_x, real_central_verts_y, real_central_verts_z], dim=-1)
    
        ### retrieve grid cells from [real_lattice_verts_xyz]
        real_cell_idx_0 = real_lattice_verts_xyz
        real_cell_idx_1 = real_lattice_verts_xyz - th.tensor([1, 0, 0], device=self.device, dtype=th.long).reshape(1, 3)
        real_cell_idx_2 = real_lattice_verts_xyz - th.tensor([0, 1, 0], device=self.device, dtype=th.long).reshape(1, 3)
        real_cell_idx_3 = real_lattice_verts_xyz - th.tensor([0, 0, 1], device=self.device, dtype=th.long).reshape(1, 3)
        real_cell_idx_4 = real_lattice_verts_xyz - th.tensor([1, 1, 0], device=self.device, dtype=th.long).reshape(1, 3)
        real_cell_idx_5 = real_lattice_verts_xyz - th.tensor([1, 0, 1], device=self.device, dtype=th.long).reshape(1, 3)
        real_cell_idx_6 = real_lattice_verts_xyz - th.tensor([0, 1, 1], device=self.device, dtype=th.long).reshape(1, 3)
        real_cell_idx_7 = real_lattice_verts_xyz - th.tensor([1, 1, 1], device=self.device, dtype=th.long).reshape(1, 3)
        real_cell_idx_A = th.cat([real_cell_idx_0, real_cell_idx_1, real_cell_idx_2, real_cell_idx_3, real_cell_idx_4, real_cell_idx_5, real_cell_idx_6, real_cell_idx_7], dim=0)
        
        ### retrieve grid cells from [real_central_verts_xyz]
        real_cell_idx_B = real_central_verts_xyz

        real_cell_idx = th.cat([real_cell_idx_A, real_cell_idx_B], dim=0)
        real_cell_idx = th.unique(real_cell_idx, dim=0)

        ### remove invalid cells
        real_cell_idx_valid = (real_cell_idx >= 0) & (real_cell_idx < th.tensor([self.num_grid_x, self.num_grid_y, self.num_grid_z], device=self.device, dtype=th.long).reshape(1, 3))
        real_cell_idx_valid = real_cell_idx_valid.all(dim=-1)
        real_cell_idx = real_cell_idx[real_cell_idx_valid]
    
        '''
        2. Find new real grid cells and derive vertices and faces from them
        '''
        ### find new real grid cells
        n_real_cell_idx_0 = real_cell_idx * 2 + th.tensor([0, 0, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_real_cell_idx_1 = real_cell_idx * 2 + th.tensor([1, 0, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_real_cell_idx_2 = real_cell_idx * 2 + th.tensor([0, 1, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_real_cell_idx_3 = real_cell_idx * 2 + th.tensor([0, 0, 1], device=self.device, dtype=th.long).reshape(1, 3)
        n_real_cell_idx_4 = real_cell_idx * 2 + th.tensor([1, 1, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_real_cell_idx_5 = real_cell_idx * 2 + th.tensor([1, 0, 1], device=self.device, dtype=th.long).reshape(1, 3)
        n_real_cell_idx_6 = real_cell_idx * 2 + th.tensor([0, 1, 1], device=self.device, dtype=th.long).reshape(1, 3)
        n_real_cell_idx_7 = real_cell_idx * 2 + th.tensor([1, 1, 1], device=self.device, dtype=th.long).reshape(1, 3)
        n_real_cell_idx = th.cat([n_real_cell_idx_0, n_real_cell_idx_1, n_real_cell_idx_2, n_real_cell_idx_3, n_real_cell_idx_4, n_real_cell_idx_5, n_real_cell_idx_6, n_real_cell_idx_7], dim=0)

        ### find vertices of new real grid cells
        # lattice verts
        n_verts_idx_0 = n_real_cell_idx
        n_verts_idx_1 = n_real_cell_idx + th.tensor([1, 0, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_2 = n_real_cell_idx + th.tensor([0, 1, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_3 = n_real_cell_idx + th.tensor([1, 1, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_4 = n_real_cell_idx + th.tensor([0, 0, 1], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_5 = n_real_cell_idx + th.tensor([1, 0, 1], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_6 = n_real_cell_idx + th.tensor([0, 1, 1], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_7 = n_real_cell_idx + th.tensor([1, 1, 1], device=self.device, dtype=th.long).reshape(1, 3)

        n_verts_idx_agg = th.cat([n_verts_idx_0, n_verts_idx_1, n_verts_idx_2, n_verts_idx_3, n_verts_idx_4, n_verts_idx_5, n_verts_idx_6, n_verts_idx_7], dim=0)
        n_verts_idx_agg = n_verts_idx_agg[:, 0] * ((etgrid.num_grid_y + 1) * (etgrid.num_grid_z + 1)) + n_verts_idx_agg[:, 1] * (etgrid.num_grid_z + 1) + n_verts_idx_agg[:, 2]

        n_verts_idx_0 = n_verts_idx_agg[:len(n_real_cell_idx)]
        n_verts_idx_1 = n_verts_idx_agg[len(n_real_cell_idx):2*len(n_real_cell_idx)]
        n_verts_idx_2 = n_verts_idx_agg[2*len(n_real_cell_idx):3*len(n_real_cell_idx)]
        n_verts_idx_3 = n_verts_idx_agg[3*len(n_real_cell_idx):4*len(n_real_cell_idx)]
        n_verts_idx_4 = n_verts_idx_agg[4*len(n_real_cell_idx):5*len(n_real_cell_idx)]
        n_verts_idx_5 = n_verts_idx_agg[5*len(n_real_cell_idx):6*len(n_real_cell_idx)]
        n_verts_idx_6 = n_verts_idx_agg[6*len(n_real_cell_idx):7*len(n_real_cell_idx)]
        n_verts_idx_7 = n_verts_idx_agg[7*len(n_real_cell_idx):8*len(n_real_cell_idx)]

        n_verts_idx_0_valid = th.ones_like(n_verts_idx_0, dtype=th.bool)
        n_verts_idx_1_valid = th.ones_like(n_verts_idx_1, dtype=th.bool)
        n_verts_idx_2_valid = th.ones_like(n_verts_idx_2, dtype=th.bool)
        n_verts_idx_3_valid = th.ones_like(n_verts_idx_3, dtype=th.bool)
        n_verts_idx_4_valid = th.ones_like(n_verts_idx_4, dtype=th.bool)
        n_verts_idx_5_valid = th.ones_like(n_verts_idx_5, dtype=th.bool)
        n_verts_idx_6_valid = th.ones_like(n_verts_idx_6, dtype=th.bool)
        n_verts_idx_7_valid = th.ones_like(n_verts_idx_7, dtype=th.bool)

        # central verts
        n_verts_idx_8 = n_real_cell_idx
        n_verts_idx_9 = n_real_cell_idx - th.tensor([1, 0, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_10 = n_real_cell_idx + th.tensor([1, 0, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_11 = n_real_cell_idx - th.tensor([0, 1, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_12 = n_real_cell_idx + th.tensor([0, 1, 0], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_13 = n_real_cell_idx - th.tensor([0, 0, 1], device=self.device, dtype=th.long).reshape(1, 3)
        n_verts_idx_14 = n_real_cell_idx + th.tensor([0, 0, 1], device=self.device, dtype=th.long).reshape(1, 3)

        n_verts_idx_agg = th.cat([n_verts_idx_8, n_verts_idx_9, n_verts_idx_10, n_verts_idx_11, n_verts_idx_12, n_verts_idx_13, n_verts_idx_14], dim=0)
        n_verts_idx_agg_valid = (n_verts_idx_agg >= 0) & (n_verts_idx_agg < th.tensor([etgrid.num_grid_x, etgrid.num_grid_y, etgrid.num_grid_z], device=self.device, dtype=th.long).reshape(1, 3))
        n_verts_idx_agg_valid = n_verts_idx_agg_valid.all(dim=-1)       # some central verts can go out of bound...
        n_verts_idx_agg = n_verts_idx_agg[:, 0] * ((etgrid.num_grid_y) * (etgrid.num_grid_z)) + n_verts_idx_agg[:, 1] * (etgrid.num_grid_z) + n_verts_idx_agg[:, 2] + ((etgrid.num_grid_x + 1) * (etgrid.num_grid_y + 1) * (etgrid.num_grid_z + 1))

        n_verts_idx_8 = n_verts_idx_agg[:len(n_real_cell_idx)]
        n_verts_idx_9 = n_verts_idx_agg[len(n_real_cell_idx):2*len(n_real_cell_idx)]
        n_verts_idx_10 = n_verts_idx_agg[2*len(n_real_cell_idx):3*len(n_real_cell_idx)]
        n_verts_idx_11 = n_verts_idx_agg[3*len(n_real_cell_idx):4*len(n_real_cell_idx)]
        n_verts_idx_12 = n_verts_idx_agg[4*len(n_real_cell_idx):5*len(n_real_cell_idx)]
        n_verts_idx_13 = n_verts_idx_agg[5*len(n_real_cell_idx):6*len(n_real_cell_idx)]
        n_verts_idx_14 = n_verts_idx_agg[6*len(n_real_cell_idx):7*len(n_real_cell_idx)]

        n_verts_idx_8_valid = n_verts_idx_agg_valid[:len(n_real_cell_idx)]
        n_verts_idx_9_valid = n_verts_idx_agg_valid[len(n_real_cell_idx):2*len(n_real_cell_idx)]
        n_verts_idx_10_valid = n_verts_idx_agg_valid[2*len(n_real_cell_idx):3*len(n_real_cell_idx)]
        n_verts_idx_11_valid = n_verts_idx_agg_valid[3*len(n_real_cell_idx):4*len(n_real_cell_idx)]
        n_verts_idx_12_valid = n_verts_idx_agg_valid[4*len(n_real_cell_idx):5*len(n_real_cell_idx)]
        n_verts_idx_13_valid = n_verts_idx_agg_valid[5*len(n_real_cell_idx):6*len(n_real_cell_idx)]
        n_verts_idx_14_valid = n_verts_idx_agg_valid[6*len(n_real_cell_idx):7*len(n_real_cell_idx)]

        n_verts_idx_agg = th.stack([
            n_verts_idx_0, n_verts_idx_1, n_verts_idx_2, n_verts_idx_3, n_verts_idx_4, n_verts_idx_5, n_verts_idx_6, n_verts_idx_7,
            n_verts_idx_8, n_verts_idx_9, n_verts_idx_10, n_verts_idx_11, n_verts_idx_12, n_verts_idx_13, n_verts_idx_14
        ], dim=1)       # [num_real_cells, 15]

        n_verts_idx_agg_valid = th.stack([
            n_verts_idx_0_valid, n_verts_idx_1_valid, n_verts_idx_2_valid, n_verts_idx_3_valid, n_verts_idx_4_valid, n_verts_idx_5_valid, n_verts_idx_6_valid, n_verts_idx_7_valid,
            n_verts_idx_8_valid, n_verts_idx_9_valid, n_verts_idx_10_valid, n_verts_idx_11_valid, n_verts_idx_12_valid, n_verts_idx_13_valid, n_verts_idx_14_valid
        ], dim=1)       # [num_real_cells, 15]

        ### find tets of new real grid cells
        tet_kernel = [0, 1, 8, 13, \
                      0, 1, 8, 11, \
                      2, 3, 8, 13, \
                      2, 3, 8, 12, \
                      4, 5, 8, 14, \
                      4, 5, 8, 11, \
                      6, 7, 8, 14, \
                      6, 7, 8, 12, \
                      
                      0, 2, 8, 13, \
                      0, 2, 8, 9, \
                      1, 3, 8, 10, \
                      1, 3, 8, 13, \
                      4, 6, 8, 14, \
                      4, 6, 8, 9, \
                      5, 7, 8, 14, \
                      5, 7, 8, 10, \
                        
                      0, 4, 8, 11, \
                      0, 4, 8, 9, \
                      1, 5, 8, 11, \
                      1, 5, 8, 10, \
                      2, 6, 8, 9, \
                      2, 6, 8, 12, \
                      3, 7, 8, 12, \
                      3, 7, 8, 10]
        n_tets_idx = n_verts_idx_agg[:, tet_kernel].reshape(-1, 4)
        n_tets_idx_valid = n_verts_idx_agg_valid[:, tet_kernel].reshape(-1, 4).all(dim=-1)
        n_tets_idx = n_tets_idx[n_tets_idx_valid]

        n_tets_idx = th.sort(n_tets_idx, dim=-1)[0]
        n_tets_idx = th.unique(n_tets_idx, dim=0)

        ### find tris of new real grid cells
        tri_kernel = [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]
        n_tris_idx = n_tets_idx[:, tri_kernel].view(-1, 3)
        n_tris_idx = th.unique(n_tris_idx, dim=0)

        ### find vertices that we need
        verts_on_real_tris, inv_verts_on_real_tris = th.unique(n_tris_idx, return_inverse=True)
        etgrid.verts_grid_idx = verts_on_real_tris

        # find coordinates of vertices
        n_num_lattice_verts = (etgrid.num_grid_x + 1) * (etgrid.num_grid_y + 1) * (etgrid.num_grid_z + 1)
        is_lattice_verts = verts_on_real_tris < n_num_lattice_verts
        is_central_verts = verts_on_real_tris >= n_num_lattice_verts

        lattice_vert_idx = verts_on_real_tris[is_lattice_verts]
        central_vert_idx = verts_on_real_tris[is_central_verts] - n_num_lattice_verts

        lattice_vert_x = lattice_vert_idx // ((etgrid.num_grid_y + 1) * (etgrid.num_grid_z + 1))
        lattice_vert_y = (lattice_vert_idx % ((etgrid.num_grid_y + 1) * (etgrid.num_grid_z + 1))) // (etgrid.num_grid_z + 1)
        lattice_vert_z = lattice_vert_idx % (etgrid.num_grid_z + 1)
        lattice_vert_xyz = th.stack([lattice_vert_x, lattice_vert_y, lattice_vert_z], dim=-1)
        lattice_vert_pos = (lattice_vert_xyz * etgrid.grid_size)
        lattice_vert_pos[:, 0] = lattice_vert_pos[:, 0] + etgrid.domain_min[0]
        lattice_vert_pos[:, 1] = lattice_vert_pos[:, 1] + etgrid.domain_min[1]
        lattice_vert_pos[:, 2] = lattice_vert_pos[:, 2] + etgrid.domain_min[2]

        central_vert_x = central_vert_idx // (etgrid.num_grid_y * etgrid.num_grid_z)
        central_vert_y = (central_vert_idx % (etgrid.num_grid_y * etgrid.num_grid_z)) // etgrid.num_grid_z
        central_vert_z = central_vert_idx % etgrid.num_grid_z
        central_vert_xyz = th.stack([central_vert_x, central_vert_y, central_vert_z], dim=-1)
        central_vert_pos = (central_vert_xyz * etgrid.grid_size) + (etgrid.grid_size * 0.5)
        central_vert_pos[:, 0] = central_vert_pos[:, 0] + etgrid.domain_min[0]
        central_vert_pos[:, 1] = central_vert_pos[:, 1] + etgrid.domain_min[1]
        central_vert_pos[:, 2] = central_vert_pos[:, 2] + etgrid.domain_min[2]

        verts_on_real_tris_pos = th.zeros((len(verts_on_real_tris), 3), device=self.device, dtype=th.float32)
        verts_on_real_tris_pos[is_lattice_verts] = lattice_vert_pos
        verts_on_real_tris_pos[is_central_verts] = central_vert_pos

        etgrid.verts = verts_on_real_tris_pos
        etgrid.tri_idx = inv_verts_on_real_tris

        return etgrid