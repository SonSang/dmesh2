import torch as th
import numpy as np
import os
import time
import argparse
import yaml
import sys
from tqdm import tqdm

from exp.d2.renderer import *
from exp.utils.utils import *
from exp.utils.dmesh import *
from exp.utils.logging import get_logger
from exp.utils.common import *

from input.common import DOMAIN

from mindiffdt.qface import qface_knn_spatial, qedge_dt
from mindiffdt.projection import knn_search, projection
from mindiffdt.minball import MB2_V0
from mindiffdt.utils import tensor_subtract_1

from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
from matplotlib import collections as mc

from easydict import EasyDict as edict

from torch_scatter import scatter

DEVICE = 'cuda:0'

MAX_KNN_K = 40              # maximum k for knn in computing expected chamfer distance
NEAR_THRESH = 1e-3          # distance threshold to be used in preal initialization
PWEIGHT_INIT_VAL = 0.9      # initial value of pweight

### rendering settings
RENDERING = False
RENDER_FORMAT = 'pdf'
RENDER_GRID = False
LINE_WIDTH = 0.4

class PCRecon2D:

    def __init__(self, 

                logdir,
                logger,

                # target points;
                target_point_positions,

                # sample interval for our mesh;
                our_sample_interval,

                # init method;
                init_args,

                # lr;
                lr_settings,

                # init preal;
                init_preal_settings,

                # optimize ppos;
                optimize_ppos_settings,
                
                # optimize pweight;
                optimize_pweight_settings,):

        self.logger = logger

        self.target_point_positions = target_point_positions
        self.our_sample_interval = our_sample_interval
        self.init_args = init_args

        '''
        Grid
        '''
        self.tgrid = TriGrid(DEVICE)
        self.ppos: th.Tensor = None
        self.preal: th.Tensor = None
        self.dtedges: th.Tensor = None
        
        '''
        Logdir
        '''
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)

        '''
        LR
        '''
        self.lr = lr_settings
        self.ppos_lr = float(self.lr.pos)
        self.preal_lr = float(self.lr.real)
        self.pweight_lr = float(self.lr.weight)
        
        '''
        Init point reals
        '''
        self.init_preal_settings = init_preal_settings
        self.init_preal_settings.num_steps = int(float(self.init_preal_settings.num_steps))
        self.init_preal_settings.vis_steps = int(float(self.init_preal_settings.vis_steps))
        self.init_preal_settings.real_reg_weight = float(self.init_preal_settings.real_reg_weight)
        
        '''
        Optimize point positions
        '''
        self.optimize_ppos_settings = optimize_ppos_settings
        self.optimize_ppos_settings.num_steps = int(float(self.optimize_ppos_settings.num_steps))
        self.optimize_ppos_settings.vis_steps = int(float(self.optimize_ppos_settings.vis_steps))

        '''
        Optimize point weights
        '''
        self.optimize_pweight_settings = optimize_pweight_settings
        self.optimize_pweight_settings.num_reward_weight = float(self.optimize_pweight_settings.num_reward_weight)
        self.optimize_pweight_settings.num_epochs = int(float(self.optimize_pweight_settings.num_epochs))
        self.optimize_pweight_settings.num_steps = int(float(self.optimize_pweight_settings.num_steps))
        self.optimize_pweight_settings.vis_steps = int(float(self.optimize_pweight_settings.vis_steps))

        '''
        Etc
        '''
        self.global_optim_start_time = 0.0
        
    '''
    Initialization and refinement
    '''
    def init_grid(self):
        grid_size = self.init_args.get("grid_size", 1e-2)
        self.tgrid.init((-DOMAIN, -DOMAIN), (DOMAIN, DOMAIN), grid_size)

        self.ppos = self.tgrid.f_verts.clone()
        self.preal = th.zeros((self.ppos.shape[0],), dtype=th.float32, device=DEVICE)
    

    '''
    Saving
    '''
    @th.no_grad()
    def save_image(self, ppos, edges, path):

        os.makedirs(path, exist_ok=True)

        fig, ax = plt.subplots()
        ax.set_xlim(-DOMAIN, DOMAIN)
        ax.set_ylim(-DOMAIN, DOMAIN)

        t_verts0 = ppos[edges[:, 0]]                        # [# face, 2]
        t_verts1 = ppos[edges[:, 1]]                        # [# face, 2]
        t_verts2 = th.stack([t_verts0, t_verts1], dim=1)    # [# face, 2, 2]
        t_verts2 = t_verts2.cpu().numpy()
            
        lc = mc.LineCollection(t_verts2, colors='k', linewidths=LINE_WIDTH)
    
        ax.add_collection(lc)

        # set visibility of x-axis as False
        xax = ax.get_xaxis()
        xax = xax.set_visible(False)
        # set visibility of y-axis as False
        yax = ax.get_yaxis()
        yax = yax.set_visible(False)

        # set aspect of the plot to be equal
        ax.set_aspect('equal')
        # set figure size
        fig.set_size_inches(36, 36)
        # remove outer box
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.tight_layout()  
        plt.savefig(os.path.join(path, f"mesh.{RENDER_FORMAT}"))
        plt.close()

        ### save timestamp
        with open(os.path.join(path, "time_sec.txt"), "w") as f:
            f.write(str(time.time() - self.global_optim_start_time))

        ### save points and edges
        np.save(os.path.join(path, "points.npy"), ppos.cpu().numpy())
        np.save(os.path.join(path, "edges.npy"), edges.cpu().numpy())

        ### save num points and edges
        with open(os.path.join(path, "mesh_info.txt"), "w") as f:
            f.write(f"num_points: {ppos.shape[0]}\n")
            f.write(f"num_edges: {edges.shape[0]}\n")
    
    '''
    Losses
    '''
    def compute_topology_regularizer(self, preal: th.Tensor):
        '''
        Remove redundant faces by penalizing the existence of faces.
        '''
        reg = th.mean(preal)
        return reg
    
    def compute_geometry_regularizer(self, preal: th.Tensor, coef: float):
        '''
        Remove redundant faces by penalizing the existence of faces.
        '''
        reg = th.mean(preal) * coef
        return reg

    def compute_eval_loss(self, positions: th.Tensor, faces: th.Tensor):
        '''
        Compute evaluation loss.
        '''
        raise NotImplementedError()

    '''
    Updates during optimization
    '''
    
    def update_lr(self, lr: float, lr_schedule: str, step: int, num_steps: int, optimizer: th.optim.Optimizer):
        
        if lr_schedule == "linear":
            lr = lr * (1.0 - (step / num_steps))
        elif lr_schedule == "exp":
            min_log_lr = np.log(MIN_LR)
            max_log_lr = np.log(lr)
            curr_log_lr = max_log_lr + (min_log_lr - max_log_lr) * (step / num_steps)
            lr = np.exp(curr_log_lr)
        elif lr_schedule == "constant":
            lr = lr
        else:
            raise ValueError(f"Invalid lr schedule: {lr_schedule}")

        lr = max(lr, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr
    
    '''
    Loss
    '''
    def sample_points_from_edges(self, point_positions: th.Tensor, edge_idx: th.Tensor):

        ppos = point_positions

        edge_v0 = ppos[edge_idx[:, 0]]
        edge_v1 = ppos[edge_idx[:, 1]]
        edge_len = th.norm(edge_v1 - edge_v0, dim=-1)
        edge_len_cumsum = th.cumsum(edge_len, dim=0)
        edge_len_cumsum_beg = th.cat([th.zeros(1, dtype=th.float32, device=DEVICE), edge_len_cumsum[:-1]], dim=0)
        
        with th.no_grad():
            our_sample_points_pos = th.arange(0.0, edge_len_cumsum[-1].cpu().item(), self.our_sample_interval, device=DEVICE)
            our_sample_points_edge = th.searchsorted(edge_len_cumsum, our_sample_points_pos)

            our_sample_points_local_pos = our_sample_points_pos - edge_len_cumsum_beg[our_sample_points_edge]
            our_sample_points_local_pos = our_sample_points_local_pos / edge_len[our_sample_points_edge]
            our_sample_points_local_pos = th.clamp(our_sample_points_local_pos, 0.0, 1.0)
            
        our_sample_points_pos = \
            edge_v0[our_sample_points_edge] * (1.0 - our_sample_points_local_pos.unsqueeze(-1)) + \
            edge_v1[our_sample_points_edge] * our_sample_points_local_pos.unsqueeze(-1)
        
        return our_sample_points_pos

    '''
    Main Optimization
    '''
    def optimize(self):

        self.global_optim_start_time = time.time()

        self.init_grid()
        self.logger.info(f"Initialized grid with {self.ppos.shape[0]} points.")

        self.logger.info(f"Start preal initialization.")
        self.init_preal()

        self.logger.info(f"Start ppos optimization.")
        self.optimize_ppos()

        if self.optimize_pweight_settings.num_epochs > 0:

            self.logger.info(f"Start pweight optimization.")
            for epoch in range(self.optimize_pweight_settings.num_epochs):
                self.logger.info(f"[pweight optimization] Epoch {epoch + 1}/{self.optimize_pweight_settings.num_epochs}")
                self.optimize_pweight(epoch + 1)

        ### save final result
        save_dir = os.path.join(self.writer.log_dir, f"result")
        self.save_image(self.ppos, self.dtedges, save_dir)

    '''
    Point-wise real initialization (fixed point-positions).
    '''
    def _refresh_preal_optimizer(self):

        preal = self.preal.clone()
        preal.requires_grad = True

        preal_lr = self.preal_lr
        optimizer = th.optim.Adam([preal], lr=preal_lr)
        
        return optimizer, preal

    def init_preal(self):

        '''
        Real settings
        '''
        target_sample_points = self.target_point_positions

        '''
        Refresh optimizer and variables
        '''
        optimizer, preal = self._refresh_preal_optimizer()
        ppos = self.ppos.clone()

        '''
        Gather fixed faces.
        '''
        edge_idx = self.tgrid.f_edge_idx
        # apex_idx = self.tgrid.f_apex_idx

        '''
        Gather point idx of which reals are fixed to 0.
        Those points that are located far from the target points are fixed to 0 real value.
        '''
        with th.no_grad():
            edge_v0 = ppos[edge_idx[:, 0]]
            edge_v1 = ppos[edge_idx[:, 1]]
            edge_ball_cen = (edge_v0 + edge_v1) * 0.5
            edge_ball_rad = th.norm(edge_v0 - edge_v1, dim=-1) * 0.5

            edge_ball_cen_nn_dist = run_knn(edge_ball_cen, self.target_point_positions, 1)[1]
            edge_ball_cen_nn_dist = edge_ball_cen_nn_dist.squeeze(-1)
            edge_ball_nn_dist = edge_ball_cen_nn_dist - edge_ball_rad

            possible_edges = edge_ball_nn_dist <= (NEAR_THRESH * 2 * DOMAIN)
            possible_edge_idx = edge_idx[possible_edges]
            # possible_apex_idx = apex_idx[possible_edges]
            possible_edge_verts = possible_edge_idx.unique()

            # @bugfix: have to update [possible_edge_idx] and [possible_apex_idx] again
            edge_comprised_of_possible_edge_verts = th.all(th.isin(edge_idx, possible_edge_verts), dim=-1)
            possible_edge_idx = edge_idx[edge_comprised_of_possible_edge_verts]
            # possible_apex_idx = apex_idx[edge_comprised_of_possible_edge_verts]

            fixed_zero_idx = th.ones_like(ppos[:, 0], dtype=th.bool)
            fixed_zero_idx[possible_edge_verts] = False

            preal[fixed_zero_idx] = 0.0
            preal[~fixed_zero_idx] = 1.0

        '''
        Sample points from edges that possibly exist.
        '''
        with th.no_grad():
            possible_edge_v0 = ppos[possible_edge_idx[:, 0]]
            possible_edge_v1 = ppos[possible_edge_idx[:, 1]]
            possible_edge_len = th.norm(possible_edge_v1 - possible_edge_v0, dim=-1)
            possible_edge_len_cumsum = th.cumsum(possible_edge_len, dim=0)
            possible_edge_len_cumsum_beg = th.cat([th.zeros(1, dtype=th.float32, device=DEVICE), possible_edge_len_cumsum[:-1]], dim=0)
            
            our_sample_points_pos = th.arange(0.0, possible_edge_len_cumsum[-1] - (self.our_sample_interval * 0.5), self.our_sample_interval, device=DEVICE)
            our_sample_points_edge = th.searchsorted(possible_edge_len_cumsum, our_sample_points_pos)

            our_sample_points_local_pos = our_sample_points_pos - possible_edge_len_cumsum_beg[our_sample_points_edge]
            our_sample_points_local_pos = our_sample_points_local_pos / possible_edge_len[our_sample_points_edge]
            our_sample_points_local_pos = th.clamp(our_sample_points_local_pos, 0.0, 1.0)

            our_sample_points_pos = \
                possible_edge_v0[our_sample_points_edge] * (1.0 - our_sample_points_local_pos.unsqueeze(-1)) + \
                possible_edge_v1[our_sample_points_edge] * our_sample_points_local_pos.unsqueeze(-1)

        '''
        Find K nearest points.
        '''
        with th.no_grad():
            num_knn = MAX_KNN_K
            num_knn = min(num_knn, len(our_sample_points_pos))
            tgt_to_our_knn_idx, tgt_to_our_knn_dist = run_knn(target_sample_points, our_sample_points_pos, num_knn)
            tgt_to_our_edge_knn_idx = our_sample_points_edge[tgt_to_our_knn_idx]

            num_knn = 1
            our_to_tgt_knn_idx, our_to_tgt_knn_dist = run_knn(our_sample_points_pos, target_sample_points, num_knn)
            our_to_tgt_knn_dist = our_to_tgt_knn_dist.squeeze(-1)

        '''
        Warmup steps: Exclude edges that are not close to the target points using direct differentiation.
        '''
        num_steps = self.init_preal_settings.num_steps
        vis_steps = self.init_preal_settings.vis_steps
        init_lr = self.preal_lr
        lr_schedule = self.init_preal_settings.lr_schedule
        real_reg_weight = self.init_preal_settings.real_reg_weight
        
        start_event = th.cuda.Event(enable_timing=True)
        end_event = th.cuda.Event(enable_timing=True)

        bar = tqdm(range(num_steps))
        for step in bar:

            curr_lr = self.update_lr(init_lr, lr_schedule, step, num_steps, optimizer)
            
            '''
            Evaluate probability based on preals
            '''
            possible_edge_preal = preal[possible_edge_idx.to(dtype=th.long)]
            possible_edge_prob = dmin(possible_edge_preal, k=DMIN_K)
            our_sample_points_prob = possible_edge_prob[our_sample_points_edge]

            '''
            2. Compute losses
            '''

            '''
            2-1. CD loss from GT to OURS
            '''

            start_event.record()
            if True:
            
                dist = tgt_to_our_knn_dist                                      # [# gt sample points, # k]
                prob_mat = possible_edge_prob[tgt_to_our_edge_knn_idx]          # [# tgt sample points, # k]
                
                '''
                Sorting: If a sample point from gt mesh finds a near point from a certain face,
                we do not consider another point from the same face in computing the loss.
                '''

                # =========

                sorted_indices = th.argsort(tgt_to_our_edge_knn_idx, dim=1, stable=True)

                # Step 2: Rearrange A using sorted indices
                sorted_A = th.gather(prob_mat, 1, sorted_indices)

                # Step 3: Identify duplicates in sorted B
                sorted_B = th.gather(tgt_to_our_edge_knn_idx, 1, sorted_indices)
                duplicate_mask = sorted_B[:, 1:] == sorted_B[:, :-1]
                # Pad the mask to match the shape of A and B
                padded_mask = th.cat([th.zeros(duplicate_mask.shape[0], 1, dtype=th.bool, device=DEVICE), duplicate_mask], dim=1)

                # Step 4: Revert A to the original order, applying the duplicate mask
                # First, set duplicates in sorted_A to 0
                sorted_A[padded_mask] = 0.0
                
                # Then, invert the sorted indices to get the original order
                inverse_indices = th.argsort(sorted_indices, dim=1)
                original_order_A = th.gather(sorted_A, 1, inverse_indices)

                prob_mat = original_order_A
                
                # =========

                # append one more column: fall back for all miss case;
                dist_n_col = th.ones((dist.shape[0], 1), dtype=th.float32, device=DEVICE) * (DOMAIN * 10)
                prob_n_col = th.ones((dist.shape[0], 1), dtype=th.float32, device=DEVICE)

                dist = th.cat([dist, dist_n_col], dim=-1)
                prob_mat = th.cat([prob_mat, prob_n_col], dim=-1)

                # =========

                n_prob_mat = 1.0 - prob_mat
                n_prob_mat_prod = th.cumprod(n_prob_mat, dim=-1)

                prob_mat[:, 1:] = prob_mat[:, 1:].clone() * n_prob_mat_prod[:, :-1]

                loss_0 = th.sum(prob_mat * dist, dim=-1)         # [# gt sample points,]
                loss_0 = loss_0.mean()

            '''
            2-2. CD loss from OURS to GT.
            '''
            if True:
                dist = our_to_tgt_knn_dist
                loss_1 = (our_sample_points_prob * dist).mean()
                loss_1 = loss_1.mean()

            end_event.record()
            th.cuda.synchronize()
            recon_loss_time = start_event.elapsed_time(end_event) / 1000.0

            '''
            2-3. Regularizers.
            '''
            start_event.record()

            real_regularizer = self.compute_topology_regularizer(
                preal
            )
            
            end_event.record()
            th.cuda.synchronize()
            real_loss_time = start_event.elapsed_time(end_event) / 1000.0

            recon_loss = loss_0 + loss_1
            loss = recon_loss + (real_regularizer * real_reg_weight)
            
            '''
            Update points.
            '''
            with th.no_grad():
                prev_preal = preal.clone()
                
            start_event.record()
            
            optimizer.zero_grad()
            loss.backward()
            
            end_event.record()
            th.cuda.synchronize()
            loss_backward_time = start_event.elapsed_time(end_event) / 1000.0

            # clip grads;
            with th.no_grad():
                preal_grad = preal.grad if preal.grad is not None else th.zeros_like(preal)
                
                # fix for nan grads;
                preal_grad_nan_idx = th.isnan(preal_grad)
                preal_grad[preal_grad_nan_idx] = 0.0

                if preal.grad is not None:
                    preal.grad.data = preal_grad
                
                preal_nan_grad_ratio = th.count_nonzero(preal_grad_nan_idx) / preal_grad_nan_idx.shape[0]
                
            optimizer.step()

            '''
            Prev mesh we got.
            '''
            with th.no_grad():
                # previous (non-differentiable) mesh we got;
                prev_mesh_faces = edge_idx[prev_preal[edge_idx].min(dim=-1).values > INIT_PREAL_THRESH]
                
                prev_num_points_on_mesh = th.unique(prev_mesh_faces).shape[0]
                prev_num_faces_on_mesh = prev_mesh_faces.shape[0]

            '''
            Bounding.
            '''
            with th.no_grad():
                preal.data = th.clamp(preal.data, min=0.0, max=1.0)
                preal.data[fixed_zero_idx] = 0.0

                # update points;
                self.preal = preal.clone()
                
                assert th.any(th.isnan(preal)) == False, "point real contains nan."
                assert th.any(th.isinf(preal)) == False, "point real contains inf."

            '''
            Logging
            '''
            with th.no_grad():
                self.writer.add_scalar(f"init_preal/loss", loss, step)
                self.writer.add_scalar(f"init_preal/recon_loss", recon_loss, step)
                self.writer.add_scalar(f"init_preal/real_regularizer", real_regularizer, step)
                
                self.writer.add_scalar(f"init_preal_info/num_faces_on_mesh", prev_num_faces_on_mesh, step)
                self.writer.add_scalar(f"init_preal_info/num_points_on_mesh", prev_num_points_on_mesh, step)

                # nan grad;
                self.writer.add_scalar(f"init_preal_nan/nan_grad_ratio", preal_nan_grad_ratio, step)
                
                # time;
                self.writer.add_scalar(f"init_preal_time/recon_loss_time", recon_loss_time, step)
                self.writer.add_scalar(f"init_preal_time/real_loss_time", real_loss_time, step)
                self.writer.add_scalar(f"init_preal_time/loss_backward_time", loss_backward_time, step)
                
                bar.set_description("loss: {:.4f}".format(loss))

            '''
            Saving
            '''
            if step % vis_steps == 0 or step == num_steps - 1:

                save_dir = os.path.join(self.writer.log_dir, f"save/init_preal")

                if RENDERING:
                    
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_image(
                        ppos,
                        prev_mesh_faces,
                        os.path.join(
                            save_dir, 
                            f"step_{step}"
                        )
                    )

        # change preal to 0 or 1
        with th.no_grad():
            preal.data[preal > INIT_PREAL_THRESH] = 1.0
            preal.data[preal <= INIT_PREAL_THRESH] = 0.0
            self.preal = preal.detach().clone()

        # remove unnecessary points
        # only points with preal == 1.0 or adjacent to points with preal == 1.0 are kept
        max_adjacency = 2
        real_verts = th.where(preal == 1.0)[0]
        adjacency_counter = th.full_like(preal, max_adjacency, dtype=th.long)
        adjacency_counter[real_verts] = 0
        for _ in range(max_adjacency - 1):
            edge_adj = adjacency_counter[self.tgrid.f_edge_idx]
            edge_adj_0 = edge_adj[:, 0]
            edge_adj_1 = edge_adj[:, 1]

            edge_vid_0 = self.tgrid.f_edge_idx[:, 0]
            edge_vid_1 = self.tgrid.f_edge_idx[:, 1]

            case_0 = (edge_adj_0 < edge_adj_1)
            case_1 = (edge_adj_0 > edge_adj_1)

            # case 0: set adjacency counter of edge_vid_1 to that of edge_vid_0 + 1
            tmp_counter = adjacency_counter[edge_vid_0] + 1
            tmp_counter[~case_0] = max_adjacency
            adjacency_counter = scatter(tmp_counter, edge_vid_1, out=adjacency_counter, dim=0, reduce='min')

            # case 1: set adjacency counter of edge_vid_0 to that of edge_vid_1 + 1
            tmp_counter = adjacency_counter[edge_vid_1] + 1
            tmp_counter[~case_1] = max_adjacency
            adjacency_counter = scatter(tmp_counter, edge_vid_0, out=adjacency_counter, dim=0, reduce='min')

        valid_verts = adjacency_counter < max_adjacency
        valid_verts_idx = th.where(valid_verts)[0]

        ppos = ppos[valid_verts_idx]
        preal = preal[valid_verts_idx]

        self.ppos = ppos.detach().clone()
        self.preal = preal.detach().clone()

        valid_verts_ratio = valid_verts.sum() / valid_verts.shape[0]
        self.logger.info(f"Point-wise real value initialization done: {ppos.shape[0]} points remain ({valid_verts_ratio * 100:.2f} % remain).")

    '''
    Point-wise position optimization with Minimum-Ball algorithm (fixed point-reals).
    '''
    def _refresh_ppos_optimizer(self):
        ppos = self.ppos.clone()
        ppos.requires_grad = True

        ppos_lr = self.ppos_lr
        optimizer = th.optim.Adam([
            {'params': [ppos], 'lr': ppos_lr},
        ])

        return optimizer, ppos

    def _geometry_sdist_to_prob(self, sdist: th.Tensor, sdist_unit: float, sigmoid_T: float):
        normalized_sdist = sdist / (sdist_unit)                     # [-sdist_unit, sdist_unit] -> [-1.0, 1.0]
        return th.sigmoid(normalized_sdist / sigmoid_T)
    
    def optimize_ppos(self):
        
        '''
        Refresh optimizer and variables
        '''
        optimizer, ppos = self._refresh_ppos_optimizer()

        num_steps = self.optimize_ppos_settings.num_steps
        vis_steps = self.optimize_ppos_settings.vis_steps

        '''
        Thresholds for signed distance used for probability computation
        '''
        sdist_unit = self.tgrid.grid_tri_height_length - (self.tgrid.grid_tri_edge_length * 0.5)
        assert sdist_unit > 0, "[sdist_unit] should be positive."

        # if [sdist] is equal to [sdist_unit], it corresponds to sigmoid(MAX_INPUT) probability
        ppos_sigmoid_max_input = PPOS_SIGMOID_MAX_INPUT
        ppos_sigmoid_T = 1.0 / ppos_sigmoid_max_input   # temperature parameter for sigmoid function
        ppos_update_thresh = ((sdist_unit / 3.0) / ppos_sigmoid_max_input)

        '''
        Find query edges
        '''
        is_real_point = (self.preal == 1.0)
        real_points_idx = th.where(is_real_point)[0]
        real_points = ppos[real_points_idx]
        
        # we only care about these edges...
        with th.no_grad():
            qedges_0 = qface_knn_spatial(real_points, QFACE_KNN_SPATIAL_K, 1)
            qedges_0 = real_points_idx[qedges_0]
            qedges_1 = qedge_dt(ppos, is_real_point)

            qedges = th.cat([qedges_0, qedges_1], dim=0)
            qedges = th.sort(qedges, dim=-1)[0]
            qedges = th.unique(qedges, dim=0)
            qedges_nearest = th.zeros_like(qedges[:, 0], dtype=th.long, device=DEVICE)
            prev_qedges_nearest = None

        start_event = th.cuda.Event(enable_timing=True)
        end_event = th.cuda.Event(enable_timing=True)

        bar = tqdm(range(num_steps))
        for step in bar:

            '''
            Evaluate probability of query edges.
            '''
            start_event.record()

            qedges_minball = MB2_V0.forward(ppos[qedges[:, 0]], ppos[qedges[:, 1]])
            if prev_qedges_nearest is None:
                qedges_nearest, qedges_nearest_info = knn_search(qedges, qedges_minball.center, qedges_minball.radius, ppos)
                prev_qedges_nearest = qedges_nearest
            else:
                # first use prev nearest to cull out edges with very low probability;
                with th.no_grad():
                    qedges_sdist = projection(qedges, qedges_minball.center, qedges_minball.radius, ppos, prev_qedges_nearest)
                    qedges_probs = self._geometry_sdist_to_prob(qedges_sdist, sdist_unit, ppos_sigmoid_T)
                    likely_qedges = qedges_probs > PROB_THRESH

                    likely_qedges_nearest, likely_qedges_nearest_info = knn_search(
                        qedges[likely_qedges], 
                        qedges_minball.center[likely_qedges], 
                        qedges_minball.radius[likely_qedges], 
                        ppos
                    )
                    prev_qedges_nearest[likely_qedges] = likely_qedges_nearest

            qedges_sdist = projection(qedges, qedges_minball.center, qedges_minball.radius, ppos, prev_qedges_nearest)
            qedges_probs = self._geometry_sdist_to_prob(qedges_sdist, sdist_unit, ppos_sigmoid_T)

            curr_edges = qedges[qedges_probs > PROB_THRESH]
            curr_edge_probs = qedges_probs[qedges_probs > PROB_THRESH]

            end_event.record()
            th.cuda.synchronize()
            prob_time = start_event.elapsed_time(end_event) / 1000.0
            
            '''
            Sample points from the edges.
            '''
            start_event.record()

            curr_edge_v0 = ppos[curr_edges[:, 0]]
            curr_edge_v1 = ppos[curr_edges[:, 1]]
            
            with th.no_grad():
                ### randomly sample points based on edge lengths
                curr_edge_len = th.norm(curr_edge_v1 - curr_edge_v0, dim=-1)
                curr_edge_len_cumsum = th.cumsum(curr_edge_len, dim=0)
                curr_edge_len_cumsum_beg = th.cat([th.zeros(1, dtype=th.float32, device=DEVICE), curr_edge_len_cumsum[:-1]], dim=0)
                
                our_sample_points_pos = th.arange(0.0, curr_edge_len_cumsum[-1] - (self.our_sample_interval * 0.5), self.our_sample_interval, device=DEVICE)
                our_sample_points_edge = th.searchsorted(curr_edge_len_cumsum, our_sample_points_pos)

                our_sample_points_local_pos = our_sample_points_pos - curr_edge_len_cumsum_beg[our_sample_points_edge]
                our_sample_points_local_pos = our_sample_points_local_pos / curr_edge_len[our_sample_points_edge]
                our_sample_points_local_pos = th.clamp(our_sample_points_local_pos, 0.0, 1.0)

                ### select middle point of every edge
                our_sample_points_local_pos_mid = th.rand((len(curr_edges),), device=DEVICE)
                our_sample_points_edge_mid = th.arange(0, len(curr_edges), device=DEVICE)

                ### merge
                our_sample_points_local_pos = th.cat([our_sample_points_local_pos, our_sample_points_local_pos_mid], dim=0)
                our_sample_points_edge = th.cat([our_sample_points_edge, our_sample_points_edge_mid], dim=0)
                
            our_sample_points_pos = \
                curr_edge_v0[our_sample_points_edge] * (1.0 - our_sample_points_local_pos.unsqueeze(-1)) + \
                curr_edge_v1[our_sample_points_edge] * our_sample_points_local_pos.unsqueeze(-1)
            our_sample_points_prob = curr_edge_probs[our_sample_points_edge]

            end_event.record()
            th.cuda.synchronize()
            sample_time = start_event.elapsed_time(end_event) / 1000.0

            '''
            Compute CD loss.
            '''
            start_event.record()

            '''
            2-1. CD loss from GT to OURS
            '''

            num_knn = MAX_KNN_K if len(our_sample_points_pos) > MAX_KNN_K else len(our_sample_points_pos)
            tgt_to_our_knn_idx, tgt_to_our_knn_dist = run_knn(
                self.target_point_positions,
                our_sample_points_pos,
                num_knn
            )
            tgt_to_our_edge_knn_idx = our_sample_points_edge[tgt_to_our_knn_idx]

            dist = tgt_to_our_knn_dist                                      # [# gt sample points, # k]
            prob_mat = curr_edge_probs[tgt_to_our_edge_knn_idx]             # [# tgt sample points, # k]
            
            '''
            Sorting: If a sample point from gt mesh finds a near point from a certain face,
            we do not consider another point from the same face in computing the loss.
            '''

            # =========

            sorted_indices = th.argsort(tgt_to_our_edge_knn_idx, dim=1, stable=True)

            # Step 2: Rearrange A using sorted indices
            sorted_A = th.gather(prob_mat, 1, sorted_indices)

            # Step 3: Identify duplicates in sorted B
            sorted_B = th.gather(tgt_to_our_edge_knn_idx, 1, sorted_indices)
            duplicate_mask = sorted_B[:, 1:] == sorted_B[:, :-1]
            # Pad the mask to match the shape of A and B
            padded_mask = th.cat([th.zeros(duplicate_mask.shape[0], 1, dtype=th.bool, device=DEVICE), duplicate_mask], dim=1)

            # Step 4: Revert A to the original order, applying the duplicate mask
            # First, set duplicates in sorted_A to 0
            sorted_A[padded_mask] = 0.0
            
            # Then, invert the sorted indices to get the original order
            inverse_indices = th.argsort(sorted_indices, dim=1)
            original_order_A = th.gather(sorted_A, 1, inverse_indices)

            prob_mat = original_order_A
            
            # =========

            # append one more column: fall back for all miss case;
            dist_n_col = th.ones((dist.shape[0], 1), dtype=th.float32, device=DEVICE) * (DOMAIN * 10)
            prob_n_col = th.ones((dist.shape[0], 1), dtype=th.float32, device=DEVICE)

            dist = th.cat([dist, dist_n_col], dim=-1)
            prob_mat = th.cat([prob_mat, prob_n_col], dim=-1)

            # =========

            n_prob_mat = 1.0 - prob_mat
            n_prob_mat_prod = th.cumprod(n_prob_mat, dim=-1)

            prob_mat[:, 1:] = prob_mat[:, 1:].clone() * n_prob_mat_prod[:, :-1]

            loss_0 = th.sum(prob_mat * dist, dim=-1)         # [# gt sample points,]
            loss_0 = loss_0.mean()

            '''
            2-2. CD loss from OURS to GT.
            '''
            num_knn = 1
            our_to_tgt_knn_idx, our_to_tgt_knn_dist = run_knn(
                our_sample_points_pos,
                self.target_point_positions,
                num_knn
            )

            dist = our_to_tgt_knn_dist.reshape(our_sample_points_prob.shape)
            loss_1 = (our_sample_points_prob * dist).mean()
            loss_1 = loss_1.mean()

            recon_loss = loss_0 + loss_1

            end_event.record()
            th.cuda.synchronize()
            recon_loss_time = start_event.elapsed_time(end_event) / 1000.0

            loss = recon_loss
            
            '''
            Update points.
            '''
            with th.no_grad():
                prev_ppos = ppos.clone()
                
            optimizer.zero_grad()

            start_event.record()
            loss.backward()
            end_event.record()
            th.cuda.synchronize()
            loss_backward_time = start_event.elapsed_time(end_event) / 1000.0
            
            # clip grads;
            with th.no_grad():
                ppos_grad = ppos.grad if ppos.grad is not None else th.zeros_like(ppos)
                
                # fix for nan grads;
                ppos_grad_nan_idx = th.any(th.isnan(ppos_grad), dim=-1)
                ppos_grad[ppos_grad_nan_idx] = 0.0
                
                if ppos.grad is not None:
                    ppos.grad.data = ppos_grad
                    
                ppos_nan_grad_ratio = th.count_nonzero(ppos_grad_nan_idx) / ppos_grad_nan_idx.shape[0]
                
            optimizer.step()

            '''
            Bounding.
            '''
            start_event.record()
            with th.no_grad():
                # ppos
                ppos_curr_perturb = ppos - prev_ppos
                ppos_curr_perturb_len = th.norm(ppos_curr_perturb, dim=-1)
                ppos_curr_perturb_dir = ppos_curr_perturb / (ppos_curr_perturb_len.unsqueeze(-1) + 1e-6)

                ppos_to_bound = ppos_curr_perturb_len > ppos_update_thresh
                ppos_safe_perturb = ppos_curr_perturb
                ppos_safe_perturb[ppos_to_bound] = \
                    ppos_curr_perturb_dir[ppos_to_bound] * ppos_update_thresh
                
                ppos_safe_perturb_len = th.norm(ppos_safe_perturb, dim=-1)
                assert th.all(ppos_safe_perturb_len <= ppos_update_thresh), "Safe perturbation range is not satisfied."

                ppos.data = prev_ppos + ppos_safe_perturb

                self.ppos = ppos.clone()

            end_event.record()
            th.cuda.synchronize()
            bound_time = start_event.elapsed_time(end_event) / 1000.0

            '''
            Logging
            '''

            with th.no_grad():
                self.writer.add_scalar(f"update_ppos_loss/loss", loss, step)
                self.writer.add_scalar(f"update_ppos_loss/recon_loss", recon_loss, step)
                self.writer.add_scalar(f"update_ppos_loss/loss_0", loss_0, step)
                self.writer.add_scalar(f"update_ppos_loss/loss_1", loss_1, step)
                
                # nan grad;
                self.writer.add_scalar(f"update_ppos_nan/ppos_nan_grad_ratio", ppos_nan_grad_ratio, step)
                
                # time;
                self.writer.add_scalar(f"update_ppos_time/recon_loss_time", recon_loss_time, step)
                self.writer.add_scalar(f"update_ppos_time/loss_backward_time", loss_backward_time, step)
                self.writer.add_scalar(f"update_ppos_time/bound_time", bound_time, step)
                self.writer.add_scalar(f"update_ppos_time/prob_time", prob_time, step)
                self.writer.add_scalar(f"update_ppos_time/sample_time", sample_time, step)
                
                bar.set_description("loss: {:.4f}".format(loss))

            '''
            Saving
            '''
            if step % vis_steps == 0 or step == num_steps - 1:
                
                if RENDERING:
                    save_dir = os.path.join(self.writer.log_dir, f"save/optimize_ppos")
                    os.makedirs(save_dir, exist_ok=True)

                    # extract edges
                    valid_qedges = qedges_sdist > 0
                    vis_edges = qedges[valid_qedges]

                    self.save_image(
                        prev_ppos,
                        vis_edges,
                        os.path.join(
                            save_dir, 
                            f"step_{step}"
                        )
                    )

        # extract edges / save final edges
        valid_qedges = qedges_sdist > 0
        vis_edges = qedges[valid_qedges]
        self.logger.info(f"Point-wise position optimization done: {vis_edges.shape[0]} edges remain.")
        self.dtedges = vis_edges.clone()

        # save final points
        ppos = prev_ppos.detach().clone()
        self.ppos = ppos

        save_dir = os.path.join(self.writer.log_dir, f"save/optimize_ppos")
        self.save_image(
            ppos,
            vis_edges,
            os.path.join(save_dir, "final")
        )

        # set [preal] based on [vis_edges]
        points_on_vis_edges = th.unique(vis_edges)
        preal = th.zeros_like(ppos[:, 0])
        preal[points_on_vis_edges] = 1.0
        self.preal = preal

    '''
    Point-wise probability optimization with Reinforce-Ball algorithm.
    '''
    def _compute_edgewise_incompatible_points(self, ppos, edges, edges_ccen, edges_crad):

        num_knn = 10

        edges_found_every_incompatible_points = th.zeros((edges.shape[0],), dtype=th.bool, device=DEVICE)

        edges_incompatible_points_idx = []          # [edge id, point id]
        while True:

            complete_ratio = th.sum(edges_found_every_incompatible_points).cpu().item() / edges.shape[0]
            
            curr_edges_idx = th.where(~edges_found_every_incompatible_points)[0]
            curr_edges, curr_edges_ccen, curr_edges_crad = \
                edges[curr_edges_idx], edges_ccen[curr_edges_idx], edges_crad[curr_edges_idx]
            
            ### find nearest neighbors in [ppos] from [edges_ccen]
            curr_edges_ccen_knn_idx, curr_edges_ccen_knn_dist = run_knn(curr_edges_ccen, ppos, num_knn)

            ### edges that found all the incompatible points in their minballs
            curr_edges_found_every_incompatible_points = (curr_edges_ccen_knn_dist[:, -1] > curr_edges_crad)
            curr_edges_found_every_incompatible_points_idx = th.where(curr_edges_found_every_incompatible_points)[0]
            curr_edges_found_every_incompatible_points_idx = curr_edges_idx[curr_edges_found_every_incompatible_points_idx]

            ### for edges that found all the incompatible points, add them in the list
            if len(curr_edges_found_every_incompatible_points_idx) == 0:
                continue

            for ki in range(num_knn):
                curr_edges_incompatible_points_idx = th.stack([
                    curr_edges_found_every_incompatible_points_idx,
                    curr_edges_ccen_knn_idx[curr_edges_found_every_incompatible_points, ki]
                ], dim=-1)

                # check validity, because incompatible points could include points on the edge
                curr_edges_incompatible_points_is_valid_0 = (
                    curr_edges[curr_edges_found_every_incompatible_points] != \
                    curr_edges_incompatible_points_idx[:, [1]]
                ).all(dim=-1)
                curr_edges_incompatible_points_is_valid_1 = (
                    curr_edges_ccen_knn_dist[curr_edges_found_every_incompatible_points, ki] <= \
                    curr_edges_crad[curr_edges_found_every_incompatible_points]
                )
                curr_edges_incompatible_points_is_valid = \
                    curr_edges_incompatible_points_is_valid_0 & curr_edges_incompatible_points_is_valid_1

                # if there is no valid incompatible points anymore, break
                if th.all(~curr_edges_incompatible_points_is_valid):
                    break
                
                curr_edges_incompatible_points_idx = curr_edges_incompatible_points_idx[curr_edges_incompatible_points_is_valid]

                edges_incompatible_points_idx.append(curr_edges_incompatible_points_idx)

            edges_found_every_incompatible_points[curr_edges_found_every_incompatible_points_idx] = True

            if th.all(edges_found_every_incompatible_points):
                break

            num_knn += 10

        edges_incompatible_points_idx = th.cat(edges_incompatible_points_idx, dim=0)
        edges_incompatible_points_idx = th.unique(edges_incompatible_points_idx, dim=0)
        u_edges, u_edges_cnt = th.unique(edges_incompatible_points_idx[:, 0], return_counts=True)
        u_edges_cnt_cumsum = th.cumsum(u_edges_cnt, dim=0)
        u_edges_cnt_cumsum_beg = th.cat([th.zeros(1, dtype=th.long, device=DEVICE), u_edges_cnt_cumsum[:-1]], dim=0)
        u_edges_cnt_cumsum_end = u_edges_cnt_cumsum

        edges_incompatible_points_beg = th.zeros((edges.shape[0],), dtype=th.long, device=DEVICE) - 1
        edges_incompatible_points_beg[u_edges] = u_edges_cnt_cumsum_beg
        edges_incompatible_points_end = th.zeros((edges.shape[0],), dtype=th.long, device=DEVICE) - 1
        edges_incompatible_points_end[u_edges] = u_edges_cnt_cumsum_end

        return edges_incompatible_points_idx, edges_incompatible_points_beg, edges_incompatible_points_end

    def _compute_point_edge_knn(self, from_points, to_points, to_edges, knn_k):

        MAX_ELEMENTS = 200_000_000
        num_from_points = from_points.shape[0]
        num_to_edges = to_edges.shape[0]
        num_from_points_to_process_per_batch = (MAX_ELEMENTS // num_to_edges) + 1
        num_batches = (num_from_points // num_from_points_to_process_per_batch) + 1

        to_edges_v0 = to_points[to_edges[:, 0]]         # [num to edges, 2]
        to_edges_v1 = to_points[to_edges[:, 1]]         # [num to edges, 2]
        to_edges_dir = to_edges_v1 - to_edges_v0        # [num to edges, 2]

        knn_idx = None
        knn_dist = None
        for bi in range(num_batches):

            from_points_beg = bi * num_from_points_to_process_per_batch
            if from_points_beg >= num_from_points:
                break
            from_points_end = min(from_points_beg + num_from_points_to_process_per_batch, num_from_points)

            b_from_points = from_points[from_points_beg:from_points_end]        # [num from points, 2]

            ### compute min distance from [b_from_points] to [to_edges]
            b_from_points = b_from_points[:, None, :]                            # [num from points, 1, 2]
            b_to_edges_v0 = to_edges_v0[None, :, :]                              # [1, num to edges, 2]
            b_to_edges_dir = to_edges_dir[None, :, :]                            # [1, num to edges, 2]
            b_foot_t = th.sum((b_from_points - b_to_edges_v0) * (b_to_edges_dir), dim=-1) / th.sum(b_to_edges_dir * b_to_edges_dir, dim=-1)     # [num from points, num to edges]
            b_foot_t = th.clamp(b_foot_t, 0.0, 1.0)
            b_foot_point = b_to_edges_v0 + b_to_edges_dir * b_foot_t[:, :, None]  # [num from points, num to edges, 2]

            b_dist = th.norm(b_foot_point - b_from_points, dim=-1)               # [num from points, num to edges]

            b_knn_idx = th.argsort(b_dist, dim=1)[:, :knn_k]                     # [num from points, knn_k]
            b_knn_dist = th.gather(b_dist, 1, b_knn_idx)                         # [num from points, knn_k]

            if knn_idx is None:
                knn_idx = b_knn_idx
                knn_dist = b_knn_dist
            else:
                knn_idx = th.cat([knn_idx, b_knn_idx], dim=0)
                knn_dist = th.cat([knn_dist, b_knn_dist], dim=0)
            
        return knn_idx, knn_dist

    def _compute_edge_point_cd(self, from_points, from_edges, to_points):

        '''
        Sample points on [from_edges]
        '''
        our_sample_interval = self.our_sample_interval

        from_edge_v0 = from_points[from_edges[:, 0]]
        from_edge_v1 = from_points[from_edges[:, 1]]

        # randomly sample points based on edge lengths
        from_edge_len = th.norm(from_edge_v1 - from_edge_v0, dim=-1)
        from_edge_len_cumsum = th.cumsum(from_edge_len, dim=0)
        from_edge_len_cumsum_beg = th.cat([th.zeros(1, dtype=th.float32, device=DEVICE), from_edge_len_cumsum[:-1]], dim=0)
        
        our_sample_points_pos = th.arange(0.0, from_edge_len_cumsum[-1] - (our_sample_interval * 0.5), our_sample_interval, device=DEVICE)
        our_sample_points_edge = th.searchsorted(from_edge_len_cumsum, our_sample_points_pos)

        our_sample_points_local_pos = our_sample_points_pos - from_edge_len_cumsum_beg[our_sample_points_edge]
        our_sample_points_local_pos = our_sample_points_local_pos / from_edge_len[our_sample_points_edge]
        our_sample_points_local_pos = th.clamp(our_sample_points_local_pos, 0.0, 1.0)

        # select middle point of every edge
        our_sample_points_local_pos_mid = th.rand((len(from_edges),), device=DEVICE)
        our_sample_points_edge_mid = th.arange(0, len(from_edges), device=DEVICE)

        ### merge
        our_sample_points_local_pos = th.cat([our_sample_points_local_pos, our_sample_points_local_pos_mid], dim=0)
        our_sample_points_edge = th.cat([our_sample_points_edge, our_sample_points_edge_mid], dim=0)
                
        our_sample_points_pos = \
            from_edge_v0[our_sample_points_edge] * (1.0 - our_sample_points_local_pos.unsqueeze(-1)) + \
            from_edge_v1[our_sample_points_edge] * our_sample_points_local_pos.unsqueeze(-1)
        
        '''
        Find nearest neighbors in [to_points] from sample points on [from_edges]
        '''
        num_knn = 1
        _, our_to_tgt_knn_dist = run_knn(
            our_sample_points_pos,
            to_points,
            num_knn
        )
        our_to_tgt_knn_dist = our_to_tgt_knn_dist.squeeze(-1)

        '''
        For each edge, get CD by averaging the distance to the nearest neighbors from the sample points
        '''
        our_to_tgt_knn_info = th.stack([our_sample_points_edge, our_to_tgt_knn_dist], dim=-1)
        our_to_tgt_knn_info = th.unique(our_to_tgt_knn_info, dim=0)
        _, u_edge_cnt = th.unique(our_to_tgt_knn_info[:, 0], return_counts=True)
        u_edge_cnt_cumsum = th.cumsum(u_edge_cnt, dim=0)
        u_edge_cnt_cumsum_beg = th.cat([th.zeros(1, dtype=th.long, device=DEVICE), u_edge_cnt_cumsum[:-1]], dim=0)
        u_edge_cnt_cumsum_end = u_edge_cnt_cumsum

        our_to_tgt_knn_dist_cumsum = th.cumsum(our_to_tgt_knn_info[:, 1], dim=0)
        u_edge_dist_cumsum_beg = our_to_tgt_knn_dist_cumsum[u_edge_cnt_cumsum_beg - 1]
        u_edge_dist_cumsum_end = our_to_tgt_knn_dist_cumsum[u_edge_cnt_cumsum_end - 1]
        u_edge_dist_cumsum_beg[0] = 0.0

        u_edge_dist_cumsum = u_edge_dist_cumsum_end - u_edge_dist_cumsum_beg
        u_edge_mean_dist = u_edge_dist_cumsum / u_edge_cnt

        assert u_edge_mean_dist.shape[0] == from_edges.shape[0], "Number of edges should be the same."

        return u_edge_mean_dist

    def _refresh_pweight_optimizer(self):
        pweight = th.full_like(self.preal, PWEIGHT_INIT_VAL, device=DEVICE, dtype=th.float32)
        pweight.requires_grad = True

        pweight_lr = self.pweight_lr
        optimizer = th.optim.Adam([pweight], lr=pweight_lr)

        return optimizer, pweight

    def optimize_pweight(self, epoch: int):
        '''
        Initialize weight for each point.
        '''
        ppos = self.ppos.clone()
        preal = self.preal.clone()
        existing_edges = self.dtedges.clone()

        init_num_points = ppos.shape[0]
        init_num_edges = existing_edges.shape[0]

        optimizer, pweight = self._refresh_pweight_optimizer()
        
        '''
        Collect list of edges that possibly exist using KNN.
        '''
        verts_is_real = preal > 0.5
        real_verts_idx = th.where(verts_is_real)[0]
        real_verts_pos = ppos[real_verts_idx]
        num_real_verts = len(real_verts_idx)
        
        pweight_neighbor_k = self.optimize_pweight_settings.possible_edge_k
        knn_k = pweight_neighbor_k if num_real_verts > pweight_neighbor_k else num_real_verts

        real_verts_knn = run_knn(real_verts_pos, real_verts_pos, knn_k)[0]
        real_verts_knn = real_verts_knn[:, 1:]                                              # [num real verts, knn_k - 1]
        real_verts_knn_idx = real_verts_idx[real_verts_knn]                                         # [num real verts, knn_k - 1]
        real_verts_cen_idx = real_verts_idx.unsqueeze(-1).expand(-1, knn_k - 1)                     # [num real verts, knn_k - 1]
        possible_edges = th.stack([real_verts_cen_idx, real_verts_knn_idx], dim=-1).reshape(-1, 2)  # [num real verts * (knn_k - 1), 2]
        possible_edges = th.cat([possible_edges, existing_edges])
        possible_edges = th.sort(possible_edges, dim=-1)[0]
        possible_edges = th.unique(possible_edges, dim=0)                                   # [num possible edges, 2]

        # find min ball for each possible edge
        possible_edge_minball = MB2_V0.forward(ppos[possible_edges[:, 0]], ppos[possible_edges[:, 1]])
        possible_edge_ccen = possible_edge_minball.center
        possible_edge_crad = possible_edge_minball.radius

        # check existence of [posisble_edge], and if there is an existing possible edge that was not in [existing_edges], remove it
        possible_edge_nearest, _ = knn_search(
            possible_edges,
            possible_edge_ccen,
            possible_edge_crad,
            ppos
        )
        possible_edge_sdist = projection(
            possible_edges,
            possible_edge_ccen,
            possible_edge_crad,
            ppos,
            possible_edge_nearest
        )
        possible_edge_exist = (possible_edge_sdist > 0)
        existing_possible_edges = possible_edges[possible_edge_exist]
        possible_edges_to_remove = tensor_subtract_1(existing_possible_edges, existing_edges)

        possible_edges = tensor_subtract_1(possible_edges, possible_edges_to_remove)
        possible_edge_minball = MB2_V0.forward(ppos[possible_edges[:, 0]], ppos[possible_edges[:, 1]])
        possible_edge_ccen = possible_edge_minball.center
        possible_edge_crad = possible_edge_minball.radius
        
        # for each ball, find out which points are included in the ball
        # @NOTE: requires O(num possible edges * num points) memory
        possible_edge_incompatible_points_info, possible_edge_incomp_beg, possible_edge_incomp_end = \
            self._compute_edgewise_incompatible_points(
                ppos,
                possible_edges,
                possible_edge_ccen,
                possible_edge_crad
            )

        points_for_optim = th.unique(possible_edge_incompatible_points_info[:, 1])
        is_optimizing_points = th.full_like(pweight, False, device=DEVICE, dtype=th.bool)
        is_optimizing_points[points_for_optim] = True
        
        '''
        Set up information for loss computation
        '''
        # for each gt point, find knn edges
        num_knn = self.optimize_pweight_settings.chamfer_k
        target_points = self.target_point_positions
        tgt_to_edge_knn_idx, tgt_to_edge_knn_dist = self._compute_point_edge_knn(
            target_points,
            ppos,
            possible_edges,
            num_knn
        )

        # for each edge, find chamfer distance to gt points
        edge_to_tgt_dist = self._compute_edge_point_cd(
            ppos,
            possible_edges,
            target_points
        )

        '''
        Optimization loop
        '''
        num_steps = self.optimize_pweight_settings.num_steps
        vis_steps = self.optimize_pweight_settings.vis_steps
        points_num_reward_weight = self.optimize_pweight_settings.num_reward_weight
        num_batch = self.optimize_pweight_settings.batch_size

        save_dir = os.path.join(self.writer.log_dir, f"save/optimize_pweight_epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        
        bar = tqdm(range(num_steps))
        for step in bar:

            '''
            Sample batch of points
            '''
            num_points = len(pweight)
            batch_sampled_points = th.rand((num_batch, num_points), device=DEVICE) <= pweight.unsqueeze(0)              # [num batch, num points]
            batch_sampled_points[:, ~is_optimizing_points] = True
            batch_sampled_points_logp = th.where(
                batch_sampled_points,
                th.log(pweight),
                th.log(1.0 - pweight)
            ).sum(dim=-1)
            batch_sampled_points_neglogp = -batch_sampled_points_logp

            '''
            Find out existing edges for each edge
            '''
            batch_edges = possible_edges[None, :, :].expand(num_batch, -1, -1)                                          # [num batch, num possible edges, 2]
            
            # cond 1. end points exist
            batch_sampled_edges_0 = th.gather(batch_sampled_points, 1, batch_edges[:, :, 0])                            # [num batch, num possible edges]
            batch_sampled_edges_1 = th.gather(batch_sampled_points, 1, batch_edges[:, :, 1])                            # [num batch, num possible edges]
            batch_sampled_edges_2 = batch_sampled_edges_0 * batch_sampled_edges_1                                       # [num batch, num possible edges]

            batch_possible_edge_exist_0 = batch_sampled_edges_2

            # cond 2. incompatible points in ball do not exist
            batch_possible_edge_incompatible_points_exist = th.index_select(
                batch_sampled_points,
                1,
                possible_edge_incompatible_points_info[:, 1]
            )       # [num batch, num incompatible points]
            batch_possible_edge_incompatible_points_exist_cumsum = th.cumsum(
                batch_possible_edge_incompatible_points_exist,
                dim=-1
            )       # [num batch, num incompatible points]
            batch_possible_edge_incompatible_points_exist_cumsum_beg = th.index_select(
                batch_possible_edge_incompatible_points_exist_cumsum,
                1,
                th.clamp(possible_edge_incomp_beg - 1, min=0)
            )       # [num batch, num possible edges]
            batch_possible_edge_incompatible_points_exist_cumsum_beg[:, 0] = 0
            batch_possible_edge_incompatible_points_exist_cumsum_end = th.index_select(
                batch_possible_edge_incompatible_points_exist_cumsum,
                1,
                th.clamp(possible_edge_incomp_end - 1, min=0)
            )       # [num batch, num possible edges]

            # [num batch, num possible edges]
            batch_possible_edge_exist_1 = ( \
                batch_possible_edge_incompatible_points_exist_cumsum_end - \
                batch_possible_edge_incompatible_points_exist_cumsum_beg) == 0
            batch_possible_edge_exist_1[:, possible_edge_incomp_beg == -1] = True

            ### final: [num batch, num possible edges]
            batch_possible_edge_exist = batch_possible_edge_exist_0 * batch_possible_edge_exist_1

            '''
            Compute CD reward
            '''
            ### from tgt to ours
            num_tgt_points = len(target_points)
            tmp_tgt_to_edge_knn_idx = tgt_to_edge_knn_idx.flatten()
            tgt_to_edge_knn_exist = th.index_select(
                batch_possible_edge_exist,
                1,
                tmp_tgt_to_edge_knn_idx
            )                                                                                       # [num batch, num tgt points * knn_k]
            tgt_to_edge_knn_exist = tgt_to_edge_knn_exist.reshape(num_batch, num_tgt_points, -1)    # [num batch, num tgt points, knn_k]
            tmp_tgt_to_edge_knn_dist = tgt_to_edge_knn_dist.unsqueeze(0)                            # [1, num tgt points, knn_k]
            tmp_tgt_to_edge_knn_dist = tmp_tgt_to_edge_knn_dist.expand(num_batch, -1, -1)           # [num batch, num tgt points, knn_k]

            tmp_tgt_to_edge_knn_dist = th.where(
                tgt_to_edge_knn_exist,
                tmp_tgt_to_edge_knn_dist,
                th.full_like(tmp_tgt_to_edge_knn_dist, DOMAIN * 10)
            )
            tmp_tgt_to_edge_min_dist = th.min(tmp_tgt_to_edge_knn_dist, dim=-1)[0]                  # [num batch, num tgt points]
            b_tgt_to_edge_cd = th.mean(tmp_tgt_to_edge_min_dist, dim=-1)                              # [num batch]

            ### from ours to gt
            tmp_edge_to_tgt_dist = edge_to_tgt_dist.unsqueeze(0)                                      # [1, num possible edges]
            tmp_edge_to_tgt_dist = tmp_edge_to_tgt_dist.expand(num_batch, -1)                         # [num batch, num possible edges]
            b_edge_to_tgt_dist = th.where(
                batch_possible_edge_exist,
                tmp_edge_to_tgt_dist,
                th.zeros_like(tmp_edge_to_tgt_dist)
            )                                                                                # [num batch, num possible edges]
            b_edge_to_tgt_dist = b_edge_to_tgt_dist.sum(dim=-1)                             # [num batch]
            b_edge_to_tgt_cd = b_edge_to_tgt_dist / (th.count_nonzero(batch_possible_edge_exist, dim=-1).float() + 1e-5)  # [num batch]

            batch_cd_reward = -(b_tgt_to_edge_cd + b_edge_to_tgt_cd)        # smaller the better

            '''
            Compute point num reward
            '''

            # compute point num reward
            batch_points_num = th.count_nonzero(batch_sampled_points, dim=-1) # / len(ppos)  # [num batch]
            batch_points_num_reward = -batch_points_num # smaller the better
            
            # final reward
            batch_reward = batch_cd_reward + (batch_points_num_reward * points_num_reward_weight)

            # actor loss
            advantages = batch_reward
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            actor_loss = (normalized_advantages * batch_sampled_points_neglogp).mean()
            
            # optimize
            optimizer.zero_grad()
            actor_loss.backward()

            with th.no_grad():
                pweight_grad = pweight.grad if pweight.grad is not None else th.zeros_like(pweight)
                
                # fix for nan grads;
                pweight_grad_nan_idx = th.isnan(pweight_grad)
                pweight_grad[pweight_grad_nan_idx] = 0.0

                if pweight.grad is not None:
                    pweight.grad.data = pweight_grad
                
                pweight_nan_grad_ratio = th.count_nonzero(pweight_grad_nan_idx) / pweight_grad_nan_idx.shape[0]
            
            optimizer.step()

            with th.no_grad():
                pweight.data[~is_optimizing_points] = 1.0
                pweight.data = th.clamp(pweight.data, 0.01, 0.99)
                
            # logging
            with th.no_grad():
                num_submerged_points = th.count_nonzero(pweight < 0.5)          # discarded points
                num_points = batch_points_num.float().mean().item()
                num_edges = th.count_nonzero(batch_possible_edge_exist, dim=-1).float().mean().item()

                self.writer.add_scalar(f"optimize_pweight_epoch_{epoch}_loss/actor_loss", actor_loss, step)
                self.writer.add_scalar(f"optimize_pweight_epoch_{epoch}_loss/reward", batch_reward.mean(), step)
                self.writer.add_scalar(f"optimize_pweight_epoch_{epoch}_loss/cd_reward", batch_cd_reward.mean(), step)
                self.writer.add_scalar(f"optimize_pweight_epoch_{epoch}_loss/points_num_reward", batch_points_num_reward.float().mean(), step)
                
                self.writer.add_scalar(f"optimize_pweight_epoch_{epoch}_info/num_submerged_points", num_submerged_points, step)
                self.writer.add_scalar(f"optimize_pweight_epoch_{epoch}_info/num_points", num_points, step)
                self.writer.add_scalar(f"optimize_pweight_epoch_{epoch}_info/num_edges", num_edges, step)
                self.writer.add_scalar(f"optimize_pweight_epoch_{epoch}_info/nan_grad_ratio", pweight_nan_grad_ratio, step)
                
                bar.set_description("loss: {:.4f}".format(actor_loss))

            # visualization
            if step % vis_steps == 0 or step == num_steps - 1:
                
                batch_sampled_points = (pweight > 0.5).unsqueeze(0)

                batch_edges = possible_edges[None, :, :].expand(1, -1, -1)                                          # [num batch, num possible edges, 2]
            
                # cond 1. end points exist
                batch_sampled_edges_0 = th.gather(batch_sampled_points, 1, batch_edges[:, :, 0])                            # [num batch, num possible edges]
                batch_sampled_edges_1 = th.gather(batch_sampled_points, 1, batch_edges[:, :, 1])                            # [num batch, num possible edges]
                batch_sampled_edges_2 = batch_sampled_edges_0 * batch_sampled_edges_1                                       # [num batch, num possible edges]

                batch_possible_edge_exist_0 = batch_sampled_edges_2

                # cond 2. incompatible points in ball do not exist
                batch_possible_edge_incompatible_points_exist = th.index_select(
                    batch_sampled_points,
                    1,
                    possible_edge_incompatible_points_info[:, 1]
                )       # [num batch, num incompatible points]
                batch_possible_edge_incompatible_points_exist_cumsum = th.cumsum(
                    batch_possible_edge_incompatible_points_exist,
                    dim=-1
                )       # [num batch, num incompatible points]
                batch_possible_edge_incompatible_points_exist_cumsum_beg = th.index_select(
                    batch_possible_edge_incompatible_points_exist_cumsum,
                    1,
                    th.clamp(possible_edge_incomp_beg - 1, min=0)
                )       # [num batch, num possible edges]
                batch_possible_edge_incompatible_points_exist_cumsum_beg[:, 0] = 0
                batch_possible_edge_incompatible_points_exist_cumsum_end = th.index_select(
                    batch_possible_edge_incompatible_points_exist_cumsum,
                    1,
                    th.clamp(possible_edge_incomp_end - 1, min=0)
                )       # [num batch, num possible edges]

                # [num batch, num possible edges]
                batch_possible_edge_exist_1 = ( \
                    batch_possible_edge_incompatible_points_exist_cumsum_end - \
                    batch_possible_edge_incompatible_points_exist_cumsum_beg) == 0
                batch_possible_edge_exist_1[:, possible_edge_incomp_beg == -1] = True

                ### final: [num batch, num possible edges]
                batch_possible_edge_exist = batch_possible_edge_exist_0 * batch_possible_edge_exist_1

                # extract edges
                vis_edges = possible_edges[batch_possible_edge_exist[0]]

                if RENDERING:
                    self.save_image(
                        ppos,
                        vis_edges,
                        os.path.join(
                            save_dir,
                            f"step_{step}"
                        )
                    )
                
        # save final connectivity
        with th.no_grad():
            translator = th.full((len(ppos), ), -1, dtype=th.long, device=DEVICE)
            translator[pweight > 0.5] = th.arange(0, (pweight > 0.5).sum(), device=DEVICE)
            dtedges = translator[vis_edges].clone()

            ppos = ppos[pweight > 0.5].clone()
            preal = preal[pweight > 0.5].clone()

            self.ppos = ppos
            self.preal = preal
            self.dtedges = dtedges

        ### save image
        self.save_image(
            ppos,
            dtedges,
            os.path.join(
                save_dir,
                f"final"
            )
        )

        ### log info
        final_num_points = ppos.shape[0]
        final_num_edges = dtedges.shape[0]
        self.logger.info(f"Point-wise weight optimization done; # points: {init_num_points} ---> {final_num_points} / # edges: {init_num_edges} ---> {final_num_edges}.")


def subsample_point_cloud_with_grid(pc: th.Tensor, grid_cell_size: int):
    '''
    From input point cloud (pc), subsample subset of points using a regular grid.
    For each cell in the grid, select only one point among the points in the cell.
    '''
    # grid_cell_size = init_tgrid_size / 1.1
    pc_x = th.floor((pc[:, 0] - (-DOMAIN)) / grid_cell_size).long()
    pc_y = th.floor((pc[:, 1] - (-DOMAIN)) / grid_cell_size).long()
    pc_cell_id = th.stack([pc_x, pc_y], dim=-1)
    pc_cell_id_pos = th.cat([pc_cell_id, pc], dim=-1)
    pc_cell_id_pos_unique = th.unique(pc_cell_id_pos, dim=0)
    _, u_pc_cell_id_cnt = th.unique(pc_cell_id_pos_unique[:, :2], return_counts=True, dim=0)
    u_pc_cell_id_cnt_cumsum = th.cumsum(u_pc_cell_id_cnt, dim=0)
    
    pc = pc_cell_id_pos_unique[u_pc_cell_id_cnt_cumsum - 1][:, 2:]

    return pc

def load_input_point_cloud(init_tgrid_size: float, input_path: str, logdir: str):

    pc = np.load(input_path)
    pc = th.tensor(pc, device=DEVICE, dtype=th.float32)
    pc_abs_max = th.max(th.abs(pc))
    if pc_abs_max >= DOMAIN:
        raise ValueError("Input point cloud should be within the domain. Please use [input/generate_pcrecon_2d_input.py] script to generate input point clouds.")
        
    # reduce number of points using grid for faster optimization;
    # set the grid cell size to be slightly smaller than the initial grid size;
    subsample_grid_cell_size = init_tgrid_size / 1.1
    pc = subsample_point_cloud_with_grid(pc, subsample_grid_cell_size)
    
    return pc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp/config/d2/pcrecon_svg.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-path", type=str, default="input/2d/pcrecon/botanical_1.npy")
    parser.add_argument("--no-log-time", action='store_true')
    parser.add_argument("--render", action='store_true')
    args = parser.parse_args()

    # load settings from yaml file;
    with open(args.config, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    DEVICE = settings['device']
    settings['args']['seed'] = args.seed
    
    if args.render:
        RENDERING = True
    
    '''
    Setup log dir and logger.
    '''
    # setup logdir;
    logdir = settings['log_dir']
    if not args.no_log_time:
        logdir = logdir + time.strftime("/%Y_%m_%d_%H_%M_%S")
    logdir = setup_logdir(logdir)

    # setup logger;
    logger = get_logger("pcrecon_2d", os.path.join(logdir, "run.log"))
    
    # save settings;
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        yaml.dump(settings, f)
    th.random.manual_seed(args.seed)

    '''
    Arguments
    '''
    # initial grid size;
    init_tgrid_size = float(settings['args']['init_args']['grid_size'])

    # the unit interval to sample points from DMesh;
    # set this to be shorter than the grid size to guarantee at least one sample per DMesh edge as much as possible;
    our_sample_points_interval = init_tgrid_size * 0.5

    '''
    Input point cloud
    '''
    input_path = args.input_path
    try:
        gt_pc = load_input_point_cloud(init_tgrid_size, input_path, logdir)
    except ValueError as e:
        logger.exception(str(e))
        sys.exit(1)
    
    logger.info(f"Num. Input Point Cloud: {len(gt_pc)}")
    logger.info(f"Init. Triangle Grid Size: {init_tgrid_size}")

    '''
    Initialize optimizer
    '''
    init_args = settings['args']['init_args']
    lr_settings = edict(settings['args']['lr'])
    init_preal_settings = edict(settings['args']['init_preal'])
    optimize_ppos_settings = edict(settings['args']['optimize_ppos'])
    optimize_pweight_settings = edict(settings['args']['optimize_pweight'])
    
    optimizer = PCRecon2D(
        logdir,
        logger,

        gt_pc,
        our_sample_points_interval,

        init_args,

        lr_settings,

        init_preal_settings,
        optimize_ppos_settings,
        optimize_pweight_settings
    )

    try:
        optimizer.optimize()
    except Exception as e:
        logger.exception(str(e))
        sys.exit(1)