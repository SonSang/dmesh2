import torch as th
from . import _C

class CGALDTStruct:
    def __init__(self):
        # points
        self.points: th.Tensor = None
        
        # [# simplex, d + 1], indices of points for each d-simplex (d = dimension)
        self.dsimp_point_id: th.Tensor = None

        # [# simplex, d], orthocenter (or dual point) of each d-simplex
        self.dsimp_cc: th.Tensor = None

        # float, computation time
        self.time_sec: float = -1.

    @staticmethod
    def forward(points: th.Tensor,
                compute_cc: bool=False):

        with th.no_grad():
            t_positions = points
            if t_positions.device != th.device('cpu'):
                t_positions = points.cpu()
            t_weights = th.zeros((points.size(0),), dtype=th.float32, device='cpu')

            result = _C.compute_wdt(t_positions, 
                                    t_weights,
                                    compute_cc)

            dsimp_point_id = result[0].to(points.device)
            dsimp_cc = result[1].to(points.device)

            ## sort dsimp_point_id
            prev_dsimp_count = len(dsimp_point_id)
            
            dsimp_point_id = th.sort(dsimp_point_id, dim=-1)[0]
            dsimp_info = th.cat([dsimp_point_id, dsimp_cc], dim=-1)
            dsimp_info = th.unique(dsimp_info, dim=0)

            dsimp_point_id = dsimp_info[:, :points.size(1) + 1].to(dtype=th.long)
            dsimp_cc = dsimp_info[:, points.size(1) + 1:].to(dtype=th.float32)

            after_dsimp_count = len(dsimp_point_id)
            assert prev_dsimp_count == after_dsimp_count, f"prev_dsimp_count={prev_dsimp_count}, after_dsimp_count={after_dsimp_count}"

            rstruct = CGALDTStruct()
            rstruct.points = points
            rstruct.dsimp_point_id = dsimp_point_id
            rstruct.dsimp_cc = dsimp_cc
            rstruct.time_sec = result[2]
            
        return rstruct
    
class CGALWDTStruct:
    def __init__(self):
        # points
        self.points: th.Tensor = None

        # weights
        self.weights: th.Tensor = None
        
        # [# simplex, d + 1], indices of points for each d-simplex (d = dimension)
        self.dsimp_point_id: th.Tensor = None

        # [# simplex, d], orthocenter (or dual point) of each d-simplex
        self.dsimp_cc: th.Tensor = None

        # float, computation time
        self.time_sec: float = -1.

    @staticmethod
    def forward(points: th.Tensor,
                weights: th.Tensor,
                compute_cc: bool=False):

        with th.no_grad():
            t_positions = points
            t_weights = weights
            if t_positions.device != th.device('cpu'):
                t_positions = points.cpu()
            if t_weights.device != th.device('cpu'):
                t_weights = weights.cpu()
            
            result = _C.compute_wdt(t_positions, 
                                    t_weights,
                                    compute_cc)

            dsimp_point_id = result[0].to(points.device)
            dsimp_cc = result[1].to(points.device)

            ## sort dsimp_point_id
            prev_dsimp_count = len(dsimp_point_id)
            
            dsimp_point_id = th.sort(dsimp_point_id, dim=-1)[0]
            dsimp_info = th.cat([dsimp_point_id, dsimp_cc], dim=-1)
            dsimp_info = th.unique(dsimp_info, dim=0)

            dsimp_point_id = dsimp_info[:, :points.size(1) + 1].to(dtype=th.long)
            dsimp_cc = dsimp_info[:, points.size(1) + 1:].to(dtype=th.float32)

            after_dsimp_count = len(dsimp_point_id)
            assert prev_dsimp_count == after_dsimp_count, f"prev_dsimp_count={prev_dsimp_count}, after_dsimp_count={after_dsimp_count}"

            rstruct = CGALWDTStruct()
            rstruct.points = points
            rstruct.weights = weights
            rstruct.dsimp_point_id = dsimp_point_id
            rstruct.dsimp_cc = dsimp_cc
            rstruct.time_sec = result[2]
            
        return rstruct