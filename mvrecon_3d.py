'''
Impl that is free of nvdiffrast (for testing non-manifoldness removal)
'''

import torch as th
import numpy as np
import argparse
import yaml
import os
import sys
import trimesh
import time
from tqdm import tqdm
from easydict import EasyDict as edict
from copy import deepcopy
from PIL import Image
import igl
from torch_scatter import scatter

from torch.utils.tensorboard import SummaryWriter

from exp.utils.utils import *
from exp.utils.dmesh import *
from exp.utils.logging import get_logger
from exp.utils.common import *

from input.common import DOMAIN, LIGHT_DIR

from dmesh2_renderer import Renderer, LayeredRenderer

from mindiffdt.tgrid import TetGrid, EffTetGrid
from mindiffdt.qface import qface_knn_spatial, qface_dt
from mindiffdt.minball import MB3_V0, Ball
from mindiffdt.projection import knn_search, projection, knn_search_multi, projection_multi
from mindiffdt.cgaldt import CGALDTStruct
from mindiffdt.utils import tensor_subtract_1
from mindiffdt.tetra import TetraSet
from mindiffdt._C import remove_non_manifold as cpp_remove_non_manifold
from mindiffdt.manifold import RenderLayersManager

### whether or not to use cache for nearest neighbors
USE_NN_CACHE = True
NN_CACHE_DEBUG = False

### number of maximum faces that we want to keep
TRI_SUBDIV_MAX_NUM = int(1e6)  

class MVRecon():

    def __init__(self, 

                logdir: str,
                logger,

                remove_nonmanifold: bool,
                optimize_color: bool,

                # gt images;
                mv: th.Tensor,
                proj: th.Tensor,
                image_size: int,
                
                gt_diffuse_map: th.Tensor,
                gt_depth_map: th.Tensor,
                
                # tetrahedral grid init method;
                init_args: dict,

                # init preal;
                init_preal_args: dict,

                num_epochs: int,

                # default epoch args;
                default_epoch_args: dict,

                # epoch-specific args;
                epoch_args: dict,):
        '''
        Rendering settings.
        '''
        self.mv = mv
        self.proj = proj
        self.image_size = image_size
        
        self.gt_diffuse_map = gt_diffuse_map
        self.gt_depth_map = gt_depth_map
        
        '''
        Init (grid)
        '''
        self.init_args = init_args
        self.tgrid = None
        
        ### point features and faces on the mesh
        self.ppos: th.Tensor = None
        self.preal: th.Tensor = None
        self.pcolor = None
        self.dtfaces: th.Tensor = None
        self.freal: th.Tensor = None
        
        '''
        Init preal
        '''
        self.init_preal_settings = init_preal_args
        self.init_preal_settings.real_lr = float(self.init_preal_settings.real_lr)
        self.init_preal_settings.color_lr = float(self.init_preal_settings.color_lr)
        self.init_preal_settings.image_size = int(float(self.init_preal_settings.image_size))
        self.init_preal_settings.patch_size = int(float(self.init_preal_settings.patch_size))
        self.init_preal_settings.num_views = int(float(self.init_preal_settings.num_views))
        self.init_preal_settings.num_steps = int(float(self.init_preal_settings.num_steps))
        self.init_preal_settings.vis_steps = int(float(self.init_preal_settings.vis_steps))
        self.init_preal_settings.refresh_steps = int(float(self.init_preal_settings.refresh_steps))
        self.init_preal_settings.real_reg_weight = float(self.init_preal_settings.real_reg_weight)
        
        '''
        Logdir
        '''
        self.logdir = logdir
        self.logger = logger
        self.writer = SummaryWriter(logdir)

        '''
        Epoch args
        '''
        self.default_epoch_args = edict(default_epoch_args)
        self.num_epochs = num_epochs

        ### optimize preal
        self.default_optimize_preal = self.default_epoch_args.optimize_preal
        self.default_optimize_preal.real_lr = float(self.default_optimize_preal.real_lr)
        self.default_optimize_preal.real_reg_weight = float(self.default_optimize_preal.real_reg_weight)
        self.default_optimize_preal.image_size = int(float(self.default_optimize_preal.image_size))
        self.default_optimize_preal.patch_size = int(float(self.default_optimize_preal.patch_size))
        self.default_optimize_preal.num_views = int(float(self.default_optimize_preal.num_views))
        self.default_optimize_preal.num_steps = int(float(self.default_optimize_preal.num_steps))
        self.default_optimize_preal.vis_steps = int(float(self.default_optimize_preal.vis_steps))
        self.optimize_preal_settings = deepcopy(self.default_optimize_preal)

        ### optimize ppos
        self.default_optimize_ppos = self.default_epoch_args.optimize_ppos
        self.default_optimize_ppos.pos_lr = float(self.default_optimize_ppos.pos_lr)
        self.default_optimize_ppos.color_lr = float(self.default_optimize_ppos.color_lr)
        self.default_optimize_ppos.quality_reg_weight = float(self.default_optimize_ppos.quality_reg_weight)
        self.default_optimize_ppos.image_size = int(float(self.default_optimize_ppos.image_size))
        self.default_optimize_ppos.patch_size = int(float(self.default_optimize_ppos.patch_size))
        self.default_optimize_ppos.num_views = int(float(self.default_optimize_ppos.num_views))
        self.default_optimize_ppos.num_steps = int(float(self.default_optimize_ppos.num_steps))
        self.default_optimize_ppos.vis_steps = int(float(self.default_optimize_ppos.vis_steps))
        self.default_optimize_ppos.nn_cache_size = int(float(self.default_optimize_ppos.nn_cache_size))
        self.default_optimize_ppos.nn_cache_steps = int(float(self.default_optimize_ppos.nn_cache_steps))
        self.default_optimize_ppos.qface_update_steps = int(float(self.default_optimize_ppos.qface_update_steps))
        self.optimize_ppos_settings = deepcopy(self.default_optimize_ppos)

        ### epoch-specific args
        self.epoch_args = epoch_args

        '''
        Etc
        '''
        self.remove_nm = remove_nonmanifold
        self.optimize_color = optimize_color
        self.global_optim_start_time = 0.0

    '''
    Others
    '''

    def init_grid(self):
        grid_size = self.init_args.get("grid_size", 1e-2)

        tgrid = TetGrid(DEVICE)
        tgrid.init(
            (-DOMAIN, -DOMAIN, -DOMAIN), 
            (DOMAIN, DOMAIN, DOMAIN), 
            grid_size
        )
        num_verts = tgrid.verts.shape[0]
        tmp_preal = th.ones((num_verts,), dtype=th.float32, device=DEVICE)

        self.tgrid = EffTetGrid.extract(tgrid, tmp_preal)

        self.ppos = self.tgrid.verts.clone()
        self.preal = th.full((self.ppos.shape[0],), 0.5, dtype=th.float32, device=DEVICE)
        self.dtfaces = self.tgrid.tri_idx.clone()
        self.pcolor = th.ones((self.ppos.shape[0], 3), dtype=th.float32, device=DEVICE)
        
    def prepare_target_images(self, target_image_size: int, target_patch_size: int):
        '''
        Prepare target images for supervision in the optimization.
        '''

        '''
        1. Downsample [gt_image] to size of [target_image_size x target_image_size].
        '''
        gt_image_size = self.image_size
        assert gt_image_size % target_image_size == 0, "Please adjust target_image_size to divide gt_image_size."

        gt_diffuse = self.gt_diffuse_map.permute(0, 3, 1, 2)        # [B, C, H, W]
        gt_depth = self.gt_depth_map.permute(0, 3, 1, 2)            # [B, C, H, W]
        
        if gt_image_size != target_image_size:
            tgt_diffuse = th.nn.functional.interpolate(
                gt_diffuse,
                size=(target_image_size, target_image_size),
                mode='bilinear',
                align_corners=False
            )
            tgt_depth = th.nn.functional.interpolate(
                gt_depth,
                size=(target_image_size, target_image_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            tgt_diffuse = gt_diffuse
            tgt_depth = gt_depth

        tgt_diffuse = tgt_diffuse.permute(0, 2, 3, 1)        # [B, H, W, C]
        tgt_depth = tgt_depth.permute(0, 2, 3, 1)            # [B, H, W, C]
        
        '''
        2. Extract patches of size [target_patch_size x target_patch_size] from downsampled images.
        Then, return the images and the patch information.
        '''
        num_images = tgt_diffuse.shape[0]
        num_patches = target_image_size // target_patch_size
        assert target_image_size % target_patch_size == 0, "Please adjust target_patch_size to divide target_image_size."

        image_patches = th.stack(th.meshgrid(
            [th.arange(num_images, device='cpu'),
            th.arange(num_patches, device='cpu'),
            th.arange(num_patches, device='cpu')],
            indexing='ij'
        ), dim=-1).reshape(-1, 3)
        image_patches_have_object = []
        for i in range(len(image_patches)):
            image_id = image_patches[i, 0]

            patch_size_original = target_patch_size * (gt_image_size // target_image_size)
            patch_x_min = image_patches[i, 1] * patch_size_original
            patch_y_min = image_patches[i, 2] * patch_size_original
            patch_x_max = th.clamp(patch_x_min + patch_size_original, max=gt_image_size)
            patch_y_max = th.clamp(patch_y_min + patch_size_original, max=gt_image_size)
            patch_depth = gt_depth[
                image_id, patch_y_min:patch_y_max, patch_x_min:patch_x_max
            ]
            image_patches_have_object.append(th.any(patch_depth > 0.0))
        image_patches = image_patches[image_patches_have_object]

        return tgt_diffuse, tgt_depth, image_patches
    
    def prepare_settings(self, epoch: int):
        self.optimize_ppos_settings = deepcopy(self.default_optimize_ppos)
        self.optimize_preal_settings = deepcopy(self.default_optimize_preal)
        
        epoch_setting = self.epoch_args.get(f"epoch_{epoch}", None)
        if epoch_setting is None:
            self.logger.info(f"Epoch {epoch} settings not found. Use default settings.")
            self.logger.info(f"Optimize ppos settings: {self.optimize_ppos_settings}")
            self.logger.info(f"Optimize preal settings: {self.optimize_preal_settings}")
            
            return

        optimize_ppos_settings = epoch_setting.get("optimize_ppos", None)
        if optimize_ppos_settings is not None:
            for k, v in optimize_ppos_settings.items():
                if k == "pos_lr":
                    self.optimize_ppos_settings.real_lr = float(v)
                elif k == "color_lr":
                    self.optimize_ppos_settings.color_lr = float(v)
                elif k == "quality_reg_weight":
                    self.optimize_ppos_settings.quality_reg_weight = float(v)
                else:
                    setattr(self.optimize_ppos_settings, k, v)
        
        optimize_preal_settings = epoch_setting.get("optimize_preal", None)
        if optimize_preal_settings is not None:
            for k, v in optimize_preal_settings.items():
                if k == "real_lr":
                    self.optimize_preal_settings.real_lr = float(v)
                elif k == "real_reg_weight":
                    self.optimize_preal_settings.real_reg_weight = float(v)
                else:
                    setattr(self.optimize_preal_settings, k, v)

        self.logger.info(f"Optimize ppos settings: {self.optimize_ppos_settings}")
        self.logger.info(f"Optimize preal settings: {self.optimize_preal_settings}")
        
    def optimize(self):

        self.global_optim_start_time = time.time()

        self.init_grid()
        self.logger.info(f"Initialized grid with {self.ppos.shape[0]} points.")

        self.logger.info(f"Start preal initialization.")
        self.init_preal()
        self.after_init_preal()

        for ei in range(self.num_epochs):
            self.logger.info(f"Start epoch {ei}.")
            self.prepare_settings(ei)

            if self.optimize_ppos_settings.num_steps > 0:
                self.logger.info(f"Optimize point-wise positions.")
                self.optimize_ppos(ei)

            if self.optimize_preal_settings.num_steps > 0:
                self.logger.info(f"Optimize point-wise real values.")
                self.optimize_preal(ei)

            if ei == self.num_epochs - 1:
                # manifold mesh
                if self.remove_nm:
                    self.logger.info("Remove non-manifoldness.")
                    self.remove_nonmanifold()

                # save final mesh
                self.save(
                    os.path.join(self.logdir, "result"),
                    self.ppos, 
                    self.dtfaces, 
                    self.pcolor, 
                    time.time() - self.global_optim_start_time, 
                    True,
                )
                self.logger.info(f"Finish optimization.")
                break

            self.logger.info("Subvidivde mesh.")
            self.subdivide(ei)

    '''
    Helpers
    '''
    def _get_overlapping_faces_in_image(
            self,
            verts: th.Tensor, 
            faces: th.Tensor, 
            mv_mats: th.Tensor, 
            proj_mats: th.Tensor, 
            image_shapes: th.Tensor,
            tiles_min: th.Tensor,
            tiles_max: th.Tensor,
            padding: float=1.0):
        '''
        @ verts: (N, 3) tensor of vertex coordinates
        @ faces: (F, 3) tensor of face indices
        @ mv_mats: (B, 4, 4) tensor of model-view matrices
        @ proj_mats: (B, 4, 4) tensor of projection matrices
        @ image_shapes: (B, 2) tensor of image shapes, (H, W)
        @ tiles_min: (B, 2) tensor of minimum tile coordinates (x, y) (pixel coordinates)
        @ tiles_max: (B, 2) tensor of maximum tile coordinates (x, y) (pixel coordinates)
        Returns:
        @ intersect: (B, F) boolean tensor indicating whether each face overlaps with the tile
        '''

        '''
        Transform verts to NDC space: Now every coordinates should reside in [-1, 1] (if in viewing frustum)
        '''
        mvp = proj_mats @ mv_mats
        verts_hom = th.cat([verts, th.ones(verts.shape[0], 1, device=verts.device)], dim=1)     # (N, 4)
        verts_proj = verts_hom @ mvp.transpose(1, 2)                                            # (B, N, 4)
        p_w = 1.0 / th.clamp(verts_proj[:, :, 3], min=1e-6)                                     # (B, N)
        verts_ndc = verts_proj[:, :, :3] * p_w.unsqueeze(2)                                     # (B, N, 3)
        verts_ndc_xy = verts_ndc[:, :, :2]                                                      # (B, N, 2)

        '''
        Get pixel coordinates of verts and faces
        '''
        verts_pixel = (verts_ndc_xy + 1) * 0.5 * image_shapes.unsqueeze(1)                      # (B, N, 2)
        face_verts_pixel_0 = verts_pixel[:, faces[:, 0], :]                                     # (B, F, 2)
        face_verts_pixel_1 = verts_pixel[:, faces[:, 1], :]                                     # (B, F, 2)
        face_verts_pixel_2 = verts_pixel[:, faces[:, 2], :]                                     # (B, F, 2)
        face_verts_pixel = th.stack([face_verts_pixel_0, face_verts_pixel_1, face_verts_pixel_2], dim=2)  # (B, F, 3, 2)

        '''
        Cull using bounding box of faces
        '''
        face_verts_pixel_min = th.min(face_verts_pixel, dim=2)[0]                               # (B, F, 2)
        face_verts_pixel_max = th.max(face_verts_pixel, dim=2)[0]                               # (B, F, 2)
        face_verts_pixel_min = face_verts_pixel_min - padding
        face_verts_pixel_max = face_verts_pixel_max + padding

        ### find overlapping faces with tiles
        not_intersect_0 = face_verts_pixel_max[..., 0] < tiles_min[:, 0].unsqueeze(1)            # (B, F)
        not_intersect_1 = face_verts_pixel_max[..., 1] < tiles_min[:, 1].unsqueeze(1)            # (B, F)
        not_intersect_2 = face_verts_pixel_min[..., 0] > tiles_max[:, 0].unsqueeze(1)            # (B, F)
        not_intersect_3 = face_verts_pixel_min[..., 1] > tiles_max[:, 1].unsqueeze(1)            # (B, F)
        not_intersect = not_intersect_0 | not_intersect_1 | not_intersect_2 | not_intersect_3    # (B, F)
        intersect = ~not_intersect                                                               # (B, F)

        return intersect

    def _is_faces_visible(self, 
                        verts: th.Tensor, 
                        faces: th.Tensor, 
                        image_size: int, 
                        patch_size: int,
                        image_patch: th.Tensor):

        if patch_size == image_size:
            return th.ones((faces.shape[0],), dtype=th.bool, device=DEVICE)

        t_mv = self.mv[image_patch[:, 0]]
        t_proj = self.proj[image_patch[:, 0]]

        t_image_shapes = th.tensor([image_size, image_size], dtype=th.long, device=DEVICE)
        t_image_shapes = t_image_shapes.unsqueeze(0).expand((t_mv.shape[0], -1))
        t_tiles_min = (image_patch[:, 1:] * patch_size).to(device=DEVICE)
        t_tiles_max = th.clamp(t_tiles_min + patch_size, max=image_size)

        is_overlap_faces = self._get_overlapping_faces_in_image(
            verts,
            faces,
            t_mv,
            t_proj,
            t_image_shapes,
            t_tiles_min,
            t_tiles_max
        )
        is_overlap_faces = is_overlap_faces.any(dim=0)
        
        return is_overlap_faces

    def _compute_faces_normals(
        self,
        vertices:th.Tensor, #V,3 first vertex may be unreferenced
        faces:th.Tensor, #F,3 long, first face may be all zero
        normalize:bool=False,
        )->th.Tensor: #F,3
        '''
        Code from [Continous Remeshing For Inverse Rendering](https://github.com/Profactor/continuous-remeshing).
        '''
        full_vertices = vertices[faces] #F,C=3,3
        v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
        face_normals = th.cross(v1-v0,v2-v0, dim=1) #F,3
        if normalize:
            face_normals = th.nn.functional.normalize(face_normals, eps=1e-6, dim=1) #TODO inplace?
        return face_normals #F,3

    def _compute_faces_view_normals(self, verts: th.Tensor, faces: th.Tensor, mv: th.Tensor):
        '''
        Compute face normals in the view space using [mv].
        @ verts: [# point, 3]
        @ faces: [# face, 3]
        @ mv: [# batch, 4, 4]
        '''
        faces_normals = self._compute_faces_normals(verts, faces, True)           # [F, 3]
        faces_normals_hom = th.cat((faces_normals, th.zeros_like(faces_normals[:, [1]])), dim=-1)   # [F, 4]
        faces_normals_hom = faces_normals_hom.unsqueeze(0).unsqueeze(-1)                    # [1, F, 4, 1]
        e_mv = mv.unsqueeze(1)                                                              # [B, 1, 4, 4]
        faces_normals_view = e_mv @ faces_normals_hom                                       # [B, F, 4, 1]
        faces_normals_view = faces_normals_view[:, :, :3, 0]                                # [B, F, 3]
        faces_normals_view[faces_normals_view[..., 2] > 0] = \
            -faces_normals_view[faces_normals_view[..., 2] > 0]                               # [B, F, 3]

        return faces_normals_view

    def _compute_faces_intense(self, verts: th.Tensor, faces: th.Tensor, mv: th.Tensor, lightdir: th.Tensor):
        '''
        Compute face intense using [mv] and [lightdir].
        @ verts: [# point, 3]
        @ faces: [# face, 3]
        @ mv: [# batch, 4, 4]
        @ lightdir: [# batch, 3]
        '''
        faces_normals_view = self._compute_faces_view_normals(verts, faces, mv)                        # [B, F, 3]
        faces_attr = th.sum(lightdir.unsqueeze(1) * faces_normals_view, -1, keepdim=True)       # [B, F, 1]
        faces_attr = th.clamp(faces_attr, min=0.0, max=1.0)                                     # [B, F, 1]
        faces_intense = faces_attr[..., 0]                                                    # [B, F]

        return faces_intense

    def preal_to_faces_prob(self, preal: th.Tensor, faces: th.Tensor, dmin_k: float=-1):
        '''
        Convert preal to faces probability.
        '''
        if dmin_k > 0:
            return dmin(preal[faces], k=dmin_k)
        else:
            return preal[faces].min(dim=-1)[0]

    def get_neighbor_verts(self, num_verts: int, conn: th.Tensor, targ_verts_idx: th.Tensor, k: int):
        '''
        Get neighbors of [targ_verts_idx] in [verts].
        [k] stands for the degree of neighbors.
        [conn] stands for connectivity (# elem, # connected verts).
        '''
        verts_degree = th.full((num_verts,), 1000, dtype=th.long, device=DEVICE)
        verts_degree[targ_verts_idx] = 0

        assert k >= 0 and k < 1000, "Please adjust k to a proper value."

        for ki in range(k):
            conn_verts_degree = verts_degree[conn]                  # [# elem, # connected verts]
            conn_verts_degree_min = conn_verts_degree.min(dim=-1)[0] + 1
            
            for ei in range(conn.shape[1]):
                vidx = conn[:, ei]
                verts_degree = scatter(conn_verts_degree_min, vidx, out=verts_degree, dim=0, reduce='min')

        n_verts_idx = th.where(verts_degree <= k)[0]

        return n_verts_idx

    '''
    Save
    '''
    def save(self, 
            save_dir: str,
            verts: th.Tensor,
            faces: th.Tensor,
            vcolors: th.Tensor,
            time: float,
            orient: bool = False,
            file_name: str = "mesh"):
        
        with th.no_grad():
            save_path = save_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if orient:
                o_faces = faces.cpu().numpy()
                o_faces = np.array(igl.bfs_orient(o_faces)[0])
            else:
                o_faces = faces.cpu().numpy()

            save_mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=o_faces, vertex_colors=vcolors.cpu().numpy() * 255.0, process=True)
            save_mesh.export(os.path.join(save_path, f"{file_name}.obj"))

            time_path = os.path.join(save_path, "time_sec.txt")
            with open(time_path, "w") as f:
                f.write("{}".format(time))

    def save_step(self,
                label: str,
                step: int, 
                verts: th.Tensor,
                faces: th.Tensor,
                vcolors: th.Tensor,
                time: float,
                orient: bool = False):

        save_path = os.path.join(self.writer.log_dir, "save/{}/step_{}".format(label, step))
        self.save(save_path, verts, faces, vcolors, time, orient)

    '''
    Losses
    '''

    def _image_loss(self, pred: th.Tensor, target: th.Tensor):
        return th.mean(th.abs(pred - target))

    def _extract_target_image_tiles(self, tgt_diffuse: th.Tensor, tgt_depth: th.Tensor, image_patches: th.Tensor, image_size: int, patch_size: int):
        patch_x_min = image_patches[:, 1] * patch_size
        patch_y_min = image_patches[:, 2] * patch_size
        patch_min = th.stack([patch_x_min, patch_y_min], dim=-1).to(device=DEVICE)
        
        image_idx = image_patches[:, 0]

        if patch_size == image_size:

            b_target_diffuse_tile = tgt_diffuse[image_idx]
            b_target_depth_tile = tgt_depth[image_idx]

        else:

            patch_x_indices = th.arange(patch_size).unsqueeze(0) + patch_x_min.unsqueeze(1) # [B, patch_size]
            patch_y_indices = th.arange(patch_size).unsqueeze(0) + patch_y_min.unsqueeze(1) # [B, patch_size]
            patch_x_indices = patch_x_indices.unsqueeze(-1)
            patch_y_indices = patch_y_indices.unsqueeze(1)

            b_target_diffuse_tile = tgt_diffuse[image_idx.unsqueeze(-1).unsqueeze(-1), patch_x_indices, patch_y_indices]
            b_target_depth_tile = tgt_depth[image_idx.unsqueeze(-1).unsqueeze(-1), patch_x_indices, patch_y_indices]

        b_target_diffuse_tile = b_target_diffuse_tile.to(device=DEVICE)
        b_target_depth_tile = b_target_depth_tile.to(device=DEVICE)

        return b_target_diffuse_tile, b_target_depth_tile, patch_min

    def compute_rendering_loss(self,
                            step: int, 
                            num_steps: int, 
                            renderer: Renderer,
                            tgt_diffuse: th.Tensor,
                            tgt_depth: th.Tensor,
                            image_patches: th.Tensor,
                            image_size: int,
                            patch_size: int,
                            ppos: th.Tensor,
                            pcolor: th.Tensor,
                            faces: th.Tensor,
                            faces_prob: th.Tensor,
                            aa_temperature: float):

        '''
        1. Ready for ground truth.
        '''
        image_idx = image_patches[:, 0]
        num_batch = image_patches.shape[0]
        b_target_diffuse_tile, b_target_depth_tile, patch_min = self._extract_target_image_tiles(
            tgt_diffuse, tgt_depth, image_patches, image_size, patch_size
        )
        b_mv = self.mv[image_idx]
        
        '''
        Setup rendering.
        '''

        # verts
        ren_verts = ppos

        # faces
        ren_faces = faces

        # verts color
        ren_verts_color = pcolor                                    # [V, 3]

        # faces opacity
        ren_faces_opacity = faces_prob                              # [F]

        # faces intense;
        lightdir = th.tensor([LIGHT_DIR], dtype=th.float32, device=DEVICE)                  # [1, 3] 
        lightdir = lightdir.expand((num_batch, -1))                                         # [B, 3]
        ren_faces_intense = self._compute_faces_intense(ren_verts, ren_faces.to(dtype=th.long), b_mv, lightdir)       # [B, F]

        # bg;
        bg = th.zeros((3,), dtype=th.float32, device=DEVICE)

        '''
        Render.
        '''
        
        try:
            diffuse, depth = renderer.forward(
                image_idx,
                patch_min,
                patch_size,
                patch_size,

                ren_verts,
                ren_faces,
                ren_verts_color,
                ren_faces_opacity,
                ren_faces_intense,

                bg,
                aa_temperature,
            )
        except:
            raise ValueError("Error in rendering.")

        depth = depth.unsqueeze(-1)
        
        ### compare to gt
        diffuse_loss = self._image_loss(diffuse, b_target_diffuse_tile)
        depth_loss = self._image_loss(depth, b_target_depth_tile)

        loss = diffuse_loss + depth_loss
        
        with th.no_grad():
            log = {
                "diffuse_loss": diffuse_loss.item(),
                "depth_loss": depth_loss.item(),
            }

            log["diffuse_image"] = diffuse[0]
            log["depth_image"] = depth[0]

            log["gt_diffuse_image"] = b_target_diffuse_tile[0]
            log["gt_depth_image"] = b_target_depth_tile[0]

        return loss, log

    '''
    Remove unnecessary points
    '''
    def _adjacency_check(self, save_dir: str):
        '''
        Remove points that are not adjacent to the real points.
        '''
        ppos = self.ppos
        preal = self.preal
        pcolor = self.pcolor
        faces = self.dtfaces

        max_adjacency = 2
        real_verts = th.where(preal == 1.0)[0]
        
        adjacency_counter = th.full_like(preal, max_adjacency, dtype=th.long)
        adjacency_counter[real_verts] = 0
        for _ in range(max_adjacency - 1):
            tri_adj = adjacency_counter[faces]
            min_tri_adj = th.min(tri_adj, dim=-1)[0]

            tri_vid_0 = faces[:, 0]
            tri_vid_1 = faces[:, 1]
            tri_vid_2 = faces[:, 2]

            adjacency_counter = scatter(min_tri_adj + 1, tri_vid_0, out=adjacency_counter, dim=0, reduce='min')
            adjacency_counter = scatter(min_tri_adj + 1, tri_vid_1, out=adjacency_counter, dim=0, reduce='min')
            adjacency_counter = scatter(min_tri_adj + 1, tri_vid_2, out=adjacency_counter, dim=0, reduce='min')

        valid_verts = adjacency_counter < max_adjacency
        valid_verts_idx = th.where(valid_verts)[0]

        ppos = ppos[valid_verts_idx]
        preal = preal[valid_verts_idx]
        pcolor = pcolor[valid_verts_idx]

        self.ppos = ppos.detach().clone()
        self.preal = preal.detach().clone()
        self.pcolor = pcolor.detach().clone()

        valid_verts_ratio = valid_verts.sum() / valid_verts.shape[0]
        self.logger.info(f"Init. preal done: {valid_verts.sum()} points remain ({valid_verts_ratio * 100:.2f} % remain).")

        '''
        Extract mesh
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dt_result = CGALDTStruct.forward(ppos)
        tets = dt_result.dsimp_point_id.to(dtype=th.long)
        dt_face_combs = [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]
        dt_faces = tets[:, dt_face_combs].reshape(-1, 3)
        dt_face_real = preal[dt_faces].all(dim=-1)
        real_faces = dt_faces[dt_face_real]
        imag_faces = dt_faces[~dt_face_real]

        real_mesh = trimesh.Trimesh(vertices=ppos.cpu().numpy(), faces=real_faces.cpu().numpy(), vertex_colors=pcolor.cpu().numpy())
        imag_mesh = trimesh.Trimesh(vertices=ppos.cpu().numpy(), faces=imag_faces.cpu().numpy(), vertex_colors=pcolor.cpu().numpy())

        real_mesh.export(os.path.join(save_dir, "real_mesh.obj"))
        imag_mesh.export(os.path.join(save_dir, "imag_mesh.obj"))

    '''
    Initialize preal
    '''
    def refresh_init_preal_optimizer(self):

        preal = self.preal.clone()
        pcolor = self.pcolor.clone()
        preal.requires_grad = True
        pcolor.requires_grad = True
        
        preal_lr = self.init_preal_settings.real_lr
        pcolor_lr = self.init_preal_settings.color_lr
        optimizer = th.optim.Adam([
            {'params': [preal], 'lr': preal_lr},
            {'params': [pcolor], 'lr': pcolor_lr},
        ])
            
        return optimizer, preal, pcolor 
    
    def compute_preal_regularizer(self, preal: th.Tensor):
        
        return th.mean(preal)
    
    def init_preal(self):

        label = f"init_preal"
        
        image_size = self.init_preal_settings.image_size
        patch_size = self.init_preal_settings.patch_size

        '''
        Init renderer
        '''
        renderer = Renderer(
            self.mv, 
            self.proj, 
            image_size, 
            image_size, 
            DEVICE
        )

        '''
        Refresh optimizer and variables
        '''
        optimizer, preal, pcolor = self.refresh_init_preal_optimizer()
        ppos = self.ppos.clone()
        faces = self.dtfaces.clone()
        
        '''
        Optimization.
        '''
        num_steps = self.init_preal_settings.num_steps
        vis_steps = self.init_preal_settings.vis_steps
        refresh_steps = self.init_preal_settings.refresh_steps
        num_views = self.init_preal_settings.num_views

        '''
        Prepare target images
        '''
        tgt_diffuse, tgt_depth, image_patches = self.prepare_target_images(
            image_size,
            patch_size
        )
        rand_image_patch_idx = th.randperm(image_patches.shape[0])
        batch_start = 0

        ### timers
        start_event = th.cuda.Event(enable_timing=True)
        end_event = th.cuda.Event(enable_timing=True)

        bar = tqdm(range(num_steps))
        for step in bar:

            '''
            Select images and patches to use in this iteration
            '''
            batch_end = batch_start + num_views
            if batch_end > rand_image_patch_idx.shape[0]:
                batch_end = rand_image_patch_idx.shape[0]

            ### (image id, patch id x, patch id y)
            t_image_patches = image_patches[rand_image_patch_idx[batch_start:batch_end]]
            if batch_end == rand_image_patch_idx.shape[0]:
                rand_image_patch_idx = th.randperm(image_patches.shape[0])
                batch_start = 0
            else:
                batch_start = batch_end

            '''
            Select faces to use in this iteration
            '''
            ### cull faces with reals first
            faces_prob = self.preal_to_faces_prob(preal, faces, DMIN_K)
            t_faces = faces[faces_prob > PROB_THRESH]
            t_faces_prob = faces_prob[faces_prob > PROB_THRESH]

            is_overlap_faces = self._is_faces_visible(
                ppos, t_faces, image_size, patch_size, t_image_patches
            )
            curr_faces = t_faces[is_overlap_faces]
            curr_faces_prob = t_faces_prob[is_overlap_faces]

            '''
            Compute losses
            '''

            ### recon loss
            start_event.record()
            recon_loss, recon_loss_info = self.compute_rendering_loss(
                step, num_steps, renderer, tgt_diffuse, tgt_depth, t_image_patches, image_size, patch_size, ppos, pcolor, curr_faces, curr_faces_prob, 0.0
            )
            end_event.record()
            th.cuda.synchronize()
            recon_loss_time = start_event.elapsed_time(end_event) / 1000.0

            ### real regularizer
            start_event.record()
            real_reg_weight = self.init_preal_settings.real_reg_weight
            preal_regularizer = self.compute_preal_regularizer(
                preal
            )
            end_event.record()
            th.cuda.synchronize()
            real_loss_time = start_event.elapsed_time(end_event) / 1000.0

            loss = recon_loss + (preal_regularizer * real_reg_weight)
            
            '''
            Update points.
            '''
            with th.no_grad():
                prev_preal = preal.clone()
                prev_pcolor = pcolor.clone()
                
            start_event.record()
            optimizer.zero_grad()
            loss.backward()
            end_event.record()
            th.cuda.synchronize()
            loss_backward_time = start_event.elapsed_time(end_event) / 1000.0
            
            # clip grads;
            with th.no_grad():
                preal_grad = preal.grad if preal.grad is not None else th.zeros_like(preal)
                pcolor_grad = pcolor.grad if pcolor.grad is not None else th.zeros_like(pcolor)
                
                # fix for nan grads;
                preal_grad_nan_idx = th.isnan(preal_grad)
                preal_grad[preal_grad_nan_idx] = 0.0

                pcolor_grad_nan_idx = th.isnan(pcolor_grad).any(dim=-1)
                pcolor_grad[pcolor_grad_nan_idx] = 0.0

                if preal.grad is not None:
                    preal.grad.data = preal_grad
                if pcolor.grad is not None:
                    pcolor.grad.data = pcolor_grad
                    
                preal_nan_grad_ratio = th.count_nonzero(preal_grad_nan_idx) / preal_grad_nan_idx.shape[0]
                pcolor_nan_grad_ratio = th.count_nonzero(pcolor_grad_nan_idx) / pcolor_grad_nan_idx.shape[0]
                
            optimizer.step()

            '''
            Prev mesh we got.
            '''
            with th.no_grad():
                # previous (non-differentiable) mesh we got;
                prev_faces_prob = self.preal_to_faces_prob(prev_preal, faces)
                existing_faces = prev_faces_prob > INIT_PREAL_THRESH
                prev_mesh_faces = faces[existing_faces]
                
                prev_num_points_on_mesh = th.unique(prev_mesh_faces).shape[0]
                prev_num_faces_on_mesh = prev_mesh_faces.shape[0]

            '''
            Bounding.
            '''
            with th.no_grad():
                preal.data = th.clamp(preal, min=0, max=1)
                pcolor.data = th.clamp(pcolor, min=0, max=1)

                if not self.optimize_color:
                    pcolor.data = prev_pcolor
                
                if step % refresh_steps == 0:
                    # add 0.2 to face reals that are adjacent to real faces
                    real_faces = th.where(preal > INIT_PREAL_THRESH)[0]
                    verts_on_real_faces = th.unique(faces[real_faces])

                    tmp_preal = th.zeros_like(self.preal)
                    tmp_preal[verts_on_real_faces] = 1.0

                    face_tmp_real = tmp_preal[faces.to(dtype=th.long)].max(dim=-1)[0]
                    adj_faces = th.where(face_tmp_real == 1.0)[0]
                    adj_points = th.unique(faces[adj_faces])
                    
                    preal[adj_points] = th.clamp(preal[adj_points] + 0.2, max=1.0)
                    
                # update points;
                self.preal = preal.clone()
                self.pcolor = pcolor.clone()
                
                assert th.any(th.isnan(preal)) == False, "point real contains nan."
                assert th.any(th.isinf(preal)) == False, "point real contains inf."
                assert th.any(th.isnan(pcolor)) == False, "point color contains nan."
                assert th.any(th.isinf(pcolor)) == False, "point color contains inf."

            '''
            Logging
            '''
            with th.no_grad():
                self.writer.add_scalar(f"{label}_loss/loss", loss, step)
                self.writer.add_scalar(f"{label}_loss/recon_loss", recon_loss, step)
                self.writer.add_scalar(f"{label}_loss/real_regularizer", preal_regularizer, step)
                
                self.writer.add_scalar(f"{label}_mesh/num_faces_on_mesh", prev_num_faces_on_mesh, step)
                self.writer.add_scalar(f"{label}_mesh/num_points_on_mesh", prev_num_points_on_mesh, step)

                # nan grad;
                self.writer.add_scalar(f"{label}_nan/p_r_nan_grad_ratio", preal_nan_grad_ratio, step)
                self.writer.add_scalar(f"{label}_nan/p_c_nan_grad_ratio", pcolor_nan_grad_ratio, step)
                
                # time;
                self.writer.add_scalar(f"{label}_time/recon_loss_time", recon_loss_time, step)
                self.writer.add_scalar(f"{label}_time/loss_backward_time", loss_backward_time, step)

                # etc;
                for k, v in recon_loss_info.items():
                    if 'loss' in k:
                        self.writer.add_scalar(f"{label}_loss/{k}", v, step)
                    elif 'image' in k:
                        if step % 50 == 0:
                            self.writer.add_image(f"{label}_image/{k}", v, step, dataformats='HWC')
                
                bar.set_description("loss: {:.4f}".format(loss))

            '''
            Saving
            '''
            if (step % vis_steps == 0 or step == num_steps - 1) and (step > 0):     # skip first step, meaningless

                self.save_step(
                    label,
                    step,
                    ppos,
                    prev_mesh_faces,
                    pcolor,
                    time.time() - self.global_optim_start_time
                )

        # change preal to 0 or 1
        with th.no_grad():
            preal.data[preal > INIT_PREAL_THRESH] = 1.0
            preal.data[preal <= INIT_PREAL_THRESH] = 0.0

        # remove dangling points of preal 1
        faces_reality = preal[faces.to(dtype=th.long)].all(dim=-1)
        real_faces = faces[faces_reality]
        verts_on_real_faces = th.unique(real_faces)
        with th.no_grad():
            preal = th.zeros_like(preal)
            preal[verts_on_real_faces] = 1.0

        self.preal = preal.detach().clone()
        self.pcolor = pcolor.detach().clone()

    def after_init_preal(self):
        '''
        Remove redundant faces and points.
        '''
        ### remove unnecessary points
        save_dir = os.path.join(self.logdir, 'save', f'after_init_preal_adjcheck')
        self._adjacency_check(save_dir)
        
    '''
    Point-wise position optimization
    '''
    def refresh_ppos_optimizer(self):

        ppos = self.ppos.clone()
        pcolor = self.pcolor.clone()
        ppos.requires_grad = True
        pcolor.requires_grad = True
        
        ppos_lr = self.optimize_ppos_settings.pos_lr
        pcolor_lr = self.optimize_ppos_settings.color_lr
        optimizer = th.optim.Adam([
            {'params': [ppos], 'lr': ppos_lr},
            {'params': [pcolor], 'lr': pcolor_lr},
        ])

        return optimizer, ppos, pcolor

    def _sdist_to_prob(self, sdist: th.Tensor, sdist_unit: float, sigmoid_T: float):
        normalized_sdist = sdist / (sdist_unit)                     # [-sdist_unit, sdist_unit] -> [-1.0, 1.0]
        return th.sigmoid(normalized_sdist / sigmoid_T)

    def compute_triangle_quality_loss(self, ppos: th.Tensor, faces: th.Tensor, face_prob: th.Tensor):

        faces_aspect_ratio = triangle_aspect_ratio(ppos, faces)
        loss = (face_prob * faces_aspect_ratio).sum() / (face_prob.sum().detach() + 1e-6)

        return loss

    def optimize_ppos(self, epoch: int):
        '''
        Optimize point-wise positions and colors.
        '''

        label = f"{epoch}_optimize_ppos"
        
        '''
        Refresh optimizer and variables
        '''
        optimizer, ppos, pcolor = self.refresh_ppos_optimizer()
        
        num_steps = self.optimize_ppos_settings.num_steps
        vis_steps = self.optimize_ppos_settings.vis_steps
        num_views = self.optimize_ppos_settings.num_views

        '''
        Thresholds for signed distance used for probability computation
        If the given signed distance is larger than (+unit), we try to consider the probability as 1.0
        If the given signed distance is smaller than (-unit), we try to consider the probability as 0.0
        '''
        sdist_unit = self.tgrid.apex_circumball_dist / (2 ** epoch)
        assert sdist_unit > 0, "[sdist_unit] should be positive."

        '''
        Temperature parameter for probability computation based on sigmoid
        Assume that the input range is [-1.0, 1.0]
        '''
        ppos_sigmoid_max_input = PPOS_SIGMOID_MAX_INPUT
        ppos_sigmoid_T = 1.0 / ppos_sigmoid_max_input   # temperature parameter for sigmoid function
        ppos_update_thresh = ((sdist_unit / 4) / ppos_sigmoid_max_input)

        '''
        Find query faces
        '''
        is_real_point = (self.preal == 1.0)
        real_points_idx = th.where(is_real_point)[0]
        real_points = ppos[real_points_idx]
        
        # we only care about these faces...
        with th.no_grad():
            qfaces_0 = qface_knn_spatial(real_points, QFACE_KNN_SPATIAL_K, 2)
            qfaces_0 = real_points_idx[qfaces_0]
            
            qfaces_1 = qface_dt(ppos, is_real_point)

            qfaces = th.cat([qfaces_0, qfaces_1], dim=0)
            qfaces = th.sort(qfaces, dim=-1)[0]
            qfaces = th.unique(qfaces, dim=0)
            qfaces_nearest = th.zeros_like(qfaces[:, 0], dtype=th.long, device=DEVICE)

        ### cache to store nearest points for each query face
        if USE_NN_CACHE:
            NN_CACHE_SIZE = self.optimize_ppos_settings.nn_cache_size
            NN_CACHE_STEP = self.optimize_ppos_settings.nn_cache_steps
        else:
            NN_CACHE_SIZE = 1
            NN_CACHE_STEP = 1
        likely_qfaces = None                    # [Q, 3]
        likely_qfaces_nearest = None            # [LQ, NN_CACHE_SIZE]
        likely_qfaces_nearest_valid = None      # [LQ, NN_CACHE_SIZE]
            
        '''
        Prepare renderer
        '''
        image_size = self.optimize_ppos_settings.image_size
        patch_size = self.optimize_ppos_settings.patch_size
        renderer = Renderer(
            self.mv,
            self.proj,
            image_size,
            image_size,
            DEVICE,
        )

        '''
        Prepare target images
        '''
        tgt_diffuse, tgt_depth, image_patches = self.prepare_target_images(
            image_size,
            patch_size
        )
        rand_image_patch_idx = th.randperm(image_patches.shape[0])
        batch_start = 0

        '''
        Prepare timer
        '''
        start_event = th.cuda.Event(enable_timing=True)
        end_event = th.cuda.Event(enable_timing=True)

        qface_update_steps = self.optimize_ppos_settings.qface_update_steps
        quality_reg_weight = self.optimize_ppos_settings.quality_reg_weight
        bar = tqdm(range(num_steps))
        for step in bar:

            '''
            Set aa temperature
            '''
            aa_temperature = 1.0

            '''
            Add query faces
            '''
            start_event.record()
            added_new_query_faces = False
            if step % qface_update_steps == 0:
                with th.no_grad():
                    ### add query faces
                    if step > 0:
                        bar.set_description("Adding query faces...")
                        real_points = ppos[real_points_idx]
                        n_qfaces_0 = qface_knn_spatial(real_points, QFACE_KNN_SPATIAL_K, 2)
                        n_qfaces_0 = real_points_idx[n_qfaces_0]
                        n_qfaces_1 = qface_dt(ppos, is_real_point)
                    
                        n_qfaces = th.cat([n_qfaces_0, n_qfaces_1], dim=0)
                        n_qfaces = th.sort(n_qfaces, dim=-1)[0]
                        n_qfaces = th.unique(n_qfaces, dim=0)

                        n_qfaces = tensor_subtract_1(n_qfaces, qfaces)
                        n_qfaces_nearest = th.zeros_like(n_qfaces[:, 0], dtype=th.long, device=DEVICE)
                        num_n_qfaces = n_qfaces.shape[0]

                        if num_n_qfaces > 0:
                            qfaces = th.cat([qfaces, n_qfaces], dim=0)
                            qfaces_nearest = th.cat([qfaces_nearest, n_qfaces_nearest], dim=0)
                            added_new_query_faces = True
            end_event.record()
            th.cuda.synchronize()
            add_qface_time = start_event.elapsed_time(end_event) * 0.001

            '''
            Update nearest cache for query faces
            '''
            start_event.record()
            if step % NN_CACHE_STEP == 0 or added_new_query_faces:
                with th.no_grad():
                    qfaces_minball, stable_mask = MB3_V0.forward(ppos[qfaces[:, 0]], ppos[qfaces[:, 1]], ppos[qfaces[:, 2]])
                    qfaces_minball_center = qfaces_minball.center
                    qfaces_minball_radius = qfaces_minball.radius

                    ### among [qfaces], cull out faces that are already unlikely using [qfaces_nearest]
                    qfaces_sdist = projection(
                        qfaces, 
                        qfaces_minball_center, 
                        qfaces_minball_radius, 
                        ppos, 
                        qfaces_nearest
                    )
                    qfaces_probs = self._sdist_to_prob(qfaces_sdist, sdist_unit, ppos_sigmoid_T)
                    qfaces_is_likely_0 = (qfaces_probs > PROB_THRESH)

                    ### for [qfaces] that passed the previous threshold, find the true nearest points
                    likely_qfaces_0 = qfaces[qfaces_is_likely_0]
                    likely_qfaces_0_minball_center = qfaces_minball_center[qfaces_is_likely_0]
                    likely_qfaces_0_minball_radius = qfaces_minball_radius[qfaces_is_likely_0]

                    likely_qfaces_0_nearest, _ = knn_search(
                        likely_qfaces_0,
                        likely_qfaces_0_minball_center,
                        likely_qfaces_0_minball_radius,
                        ppos
                    )
                    qfaces_nearest[qfaces_is_likely_0] = likely_qfaces_0_nearest

                    likely_qfaces_0_sdist = projection(
                        likely_qfaces_0,
                        likely_qfaces_0_minball_center,
                        likely_qfaces_0_minball_radius,
                        ppos,
                        likely_qfaces_0_nearest
                    )
                    likely_qfaces_0_probs = self._sdist_to_prob(likely_qfaces_0_sdist, sdist_unit, ppos_sigmoid_T)
                    likely_qfaces_0_is_likely = (likely_qfaces_0_probs > PROB_THRESH)
                    
                    ### we set the likely qfaces that passed the previous threshold, and only care about them until the next cache update
                    likely_qfaces = likely_qfaces_0[likely_qfaces_0_is_likely]
                    likely_qfaces_minball_center = likely_qfaces_0_minball_center[likely_qfaces_0_is_likely]
                    likely_qfaces_minball_radius = likely_qfaces_0_minball_radius[likely_qfaces_0_is_likely]

                    if NN_CACHE_SIZE > 1 and NN_CACHE_STEP > 1:
                        likely_qfaces_nearest, likely_qfaces_nearest_valid = knn_search_multi(
                            likely_qfaces,
                            likely_qfaces_minball_center,
                            likely_qfaces_minball_radius,
                            ppos,
                            NN_CACHE_SIZE
                        )
                    else:
                        likely_qfaces_nearest = likely_qfaces_0_nearest[likely_qfaces_0_is_likely].unsqueeze(-1)
                        likely_qfaces_nearest_valid = th.ones_like(likely_qfaces_nearest, dtype=th.bool, device=DEVICE)
            
            end_event.record()
            th.cuda.synchronize()
            nn_cache_time = start_event.elapsed_time(end_event) * 0.001

            '''
            Select images and patches to use in this iteration
            '''
            batch_end = batch_start + num_views
            if batch_end > rand_image_patch_idx.shape[0]:
                batch_end = rand_image_patch_idx.shape[0]

            ### (image id, patch id x, patch id y)
            t_image_patches = image_patches[rand_image_patch_idx[batch_start:batch_end]]
            if batch_end == rand_image_patch_idx.shape[0]:
                rand_image_patch_idx = th.randperm(image_patches.shape[0])
                batch_start = 0
            else:
                batch_start = batch_end

            '''
            Evaluate probability of query faces.
            '''
            start_event.record()

            num_likely_qfaces = likely_qfaces.shape[0]

            likely_qfaces_minball, stable_mask = MB3_V0.forward(ppos[likely_qfaces[:, 0]], ppos[likely_qfaces[:, 1]], ppos[likely_qfaces[:, 2]])

            stable_qfaces = likely_qfaces[stable_mask]
            stable_qfaces_minball_center = likely_qfaces_minball.center[stable_mask]
            stable_qfaces_minball_radius = likely_qfaces_minball.radius[stable_mask]
            stable_qfaces_nearest = likely_qfaces_nearest[stable_mask]
            stable_qfaces_nearest_is_valid = likely_qfaces_nearest_valid[stable_mask]
            
            ### among stable qfaces, select those that would be rendered;
            is_overlap_faces = self._is_faces_visible(
                ppos, stable_qfaces, image_size, patch_size, t_image_patches
            )
            # is_overlap_faces = th.ones(likely_stable_qfaces.shape[0], dtype=th.bool, device=DEVICE)
            
            target_qfaces = stable_qfaces[is_overlap_faces]
            target_qfaces_minball_center = stable_qfaces_minball_center[is_overlap_faces]
            target_qfaces_minball_radius = stable_qfaces_minball_radius[is_overlap_faces]
            target_qfaces_nearest = stable_qfaces_nearest[is_overlap_faces]
            target_qfaces_nearest_is_valid = stable_qfaces_nearest_is_valid[is_overlap_faces]

            if NN_CACHE_DEBUG:
                target_qfaces_nearest_debug, _ = knn_search(
                    target_qfaces,
                    target_qfaces_minball_center,
                    target_qfaces_minball_radius,
                    ppos
                )
                target_qfaces_hit = th.any(target_qfaces_nearest == target_qfaces_nearest_debug.unsqueeze(-1), dim=-1)
                target_qfaces_hit_rate = th.count_nonzero(target_qfaces_hit) / (target_qfaces.shape[0] + 1)
            else:
                target_qfaces_hit_rate = 0.0
            
            ### compute probs for target qfaces;
            target_qfaces_sdist = projection_multi(
                target_qfaces,
                target_qfaces_minball_center,
                target_qfaces_minball_radius,
                ppos,
                target_qfaces_nearest,
                target_qfaces_nearest_is_valid
            )
            
            target_qfaces_probs = self._sdist_to_prob(target_qfaces_sdist, sdist_unit, ppos_sigmoid_T)
            valid_target_qfaces = (target_qfaces_probs > PROB_THRESH)
            
            curr_faces = target_qfaces[valid_target_qfaces]
            curr_face_probs = target_qfaces_probs[valid_target_qfaces]
            num_faces_to_render = curr_faces.shape[0]

            end_event.record()
            th.cuda.synchronize()
            prob_time = start_event.elapsed_time(end_event) * 0.001
            
            '''
            Compute recon loss.
            '''
            start_event.record()
            recon_loss, recon_loss_info = self.compute_rendering_loss(
                step, num_steps, renderer, tgt_diffuse, tgt_depth, t_image_patches, image_size, patch_size, ppos, pcolor, curr_faces, curr_face_probs, aa_temperature
            )
            end_event.record()
            th.cuda.synchronize()
            recon_loss_time = start_event.elapsed_time(end_event) * 0.001

            '''
            Compute triangle quality loss.
            '''
            start_event.record()
            quality_loss = self.compute_triangle_quality_loss(
                ppos, curr_faces, curr_face_probs,
            )
            end_event.record()
            th.cuda.synchronize()
            quality_loss_time = start_event.elapsed_time(end_event) * 0.001

            loss = recon_loss + (quality_reg_weight * quality_loss)

            
            '''
            Update points.
            '''
            with th.no_grad():
                prev_ppos = ppos.clone()
                prev_pcolor = pcolor.clone()
                
            optimizer.zero_grad()

            start_event.record()
            loss.backward()
            end_event.record()
            th.cuda.synchronize()
            loss_backward_time = start_event.elapsed_time(end_event) * 0.001

            # clip grads;
            with th.no_grad():
                ppos_grad = ppos.grad if ppos.grad is not None else th.zeros_like(ppos)
                pcolor_grad = pcolor.grad if pcolor.grad is not None else th.zeros_like(pcolor)
                
                # fix for nan grads;
                ppos_grad_nan_idx = th.any(th.isnan(ppos_grad), dim=-1)
                pcolor_grad_nan_idx = th.any(th.isnan(pcolor_grad), dim=-1)
                ppos_grad[ppos_grad_nan_idx] = 0.0
                pcolor_grad[pcolor_grad_nan_idx] = 0.0
                
                if ppos.grad is not None:
                    ppos.grad.data = ppos_grad
                if pcolor.grad is not None:
                    pcolor.grad.data = pcolor_grad
                    
                ppos_nan_grad_ratio = th.count_nonzero(ppos_grad_nan_idx) / ppos_grad_nan_idx.shape[0]
                pcolor_nan_grad_ratio = th.count_nonzero(pcolor_grad_nan_idx) / pcolor_grad_nan_idx.shape[0]
                
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

                # pcolor
                pcolor.data = th.clamp(pcolor, min=0, max=1)
                if not self.optimize_color:
                    pcolor.data = prev_pcolor

                self.pcolor = pcolor.clone()
                
            end_event.record()
            th.cuda.synchronize()
            bound_time = start_event.elapsed_time(end_event) * 0.001

            '''
            Logging
            '''

            with th.no_grad():
                self.writer.add_scalar(f"{label}_loss/loss", loss, step)
                self.writer.add_scalar(f"{label}_loss/recon_loss", recon_loss, step)
                self.writer.add_scalar(f"{label}_loss/quality_loss", quality_loss, step)

                # info;
                self.writer.add_scalar(f"{label}_info/pos_sigmoid_temperature", ppos_sigmoid_T, step)
                self.writer.add_scalar(f"{label}_info/num_faces_to_render", num_faces_to_render, step)
                self.writer.add_scalar(f"{label}_info/num_likely_qfaces", num_likely_qfaces, step)
                self.writer.add_scalar(f"{label}_info/target_qfaces_hit_rate", target_qfaces_hit_rate, step)

                # nan grad;
                self.writer.add_scalar(f"{label}_nan/ppos_nan_grad_ratio", ppos_nan_grad_ratio, step)
                self.writer.add_scalar(f"{label}_nan/pcolor_nan_grad_ratio", pcolor_nan_grad_ratio, step)
                
                # time;
                self.writer.add_scalar(f"{label}_time/prob_time", prob_time, step)
                self.writer.add_scalar(f"{label}_time/recon_loss_time", recon_loss_time, step)
                self.writer.add_scalar(f"{label}_time/loss_backward_time", loss_backward_time, step)
                self.writer.add_scalar(f"{label}_time/bound_time", bound_time, step)
                self.writer.add_scalar(f"{label}_time/nn_cache_time", nn_cache_time, step)
                self.writer.add_scalar(f"{label}_time/add_qface_time", add_qface_time, step)
                self.writer.add_scalar(f"{label}_time/quality_loss_time", quality_loss_time, step)

                # etc;
                for k, v in recon_loss_info.items():
                    if 'loss' in k:
                        self.writer.add_scalar(f"{label}_loss/{k}", v, step)
                    elif 'image' in k:
                        if step % 50 == 0:
                            self.writer.add_image(f"{label}_image/{k}", v, step, dataformats='HWC')
                
                bar.set_description("loss: {:.4f}".format(loss))

            '''
            Saving
            '''
            if step % vis_steps == 0 or step == num_steps - 1:

                with th.no_grad():

                    ### real faces
                    target_qfaces_valid = target_qfaces_probs > 0.5
                    vis_faces = target_qfaces[target_qfaces_valid]
                    
                    self.save_step(
                        label,
                        step,
                        prev_ppos,
                        vis_faces,
                        prev_pcolor,
                        time.time() - self.global_optim_start_time
                    )


        with th.no_grad():
            ### run dt, and select faces in dt that satisfy the min-ball constraint
            ### this is because dt is more numerically stable
            ppos = self.ppos.clone()
            dt_result = CGALDTStruct.forward(ppos)
            tets = dt_result.dsimp_point_id.to(dtype=th.long)
            
            tetra_set = TetraSet(ppos, tets)
            faces = tetra_set.faces
            faces_is_real = self.preal[faces].all(dim=-1)
            qfaces = faces[faces_is_real]

            ### min-ball constraint
            qfaces_minball, _ = MB3_V0.forward(ppos[qfaces[:, 0]], ppos[qfaces[:, 1]], ppos[qfaces[:, 2]])
            qfaces_minball_center = qfaces_minball.center
            qfaces_minball_radius = qfaces_minball.radius

            qfaces_nearest, _ = knn_search(
                qfaces,
                qfaces_minball_center,
                qfaces_minball_radius,
                ppos
            )

            qfaces_sdist = projection(
                qfaces,
                qfaces_minball_center,
                qfaces_minball_radius,
                ppos,
                qfaces_nearest
            )

            qfaces_satisfying_min_ball = qfaces[qfaces_sdist > 0]
                
            vis_faces = qfaces_satisfying_min_ball
            self.logger.info(f"Num. faces after point-wise position optimization: {vis_faces.shape[0]}")

            save_path = os.path.join(self.writer.log_dir, f"save/{label}/final")

            self.save(
                save_path,
                self.ppos,
                vis_faces,
                self.pcolor,
                time.time() - self.global_optim_start_time
            )

            self.dtfaces = vis_faces.clone()

    '''
    Point-wise real value optimization
    '''
        
    def refresh_preal_optimizer(self, num_faces: int):
        ### optimize face-wise real values, instead of point-wise values
        freal = th.full((num_faces,), 1.0, dtype=th.float32, device=DEVICE, requires_grad=True)
        freal_lr = self.optimize_preal_settings.real_lr

        optimizer = th.optim.Adam([freal,], lr=freal_lr)
        return optimizer, freal
    
    def optimize_preal(self, epoch: int):

        label = f"{epoch}_optimize_preal"

        '''
        Run DT and construct tets
        '''
        ppos = self.ppos.clone()
        pcolor = self.pcolor.clone()
        opt_faces = self.dtfaces.clone()        # optimized faces that we currently have

        ### sort and unique opt_faces
        opt_faces = th.sort(opt_faces, dim=-1)[0]
        opt_faces = th.unique(opt_faces, dim=0)

        ### run dt
        dt_result = CGALDTStruct.forward(ppos)
        tets = dt_result.dsimp_point_id.to(dtype=th.long)
        
        ### construct tetra set
        tetra_set = TetraSet(ppos, tets)

        '''
        1. Visibility check: remove faces that are not visible
        '''
        dt_points = ppos
        dt_faces = tetra_set.faces
        dt_tets = tetra_set.tets
        dt_face_tet = tetra_set.face_tet
        dt_tet_face = tetra_set.tet_faces

        ### find out if each of [dt_faces] is in [opt_faces] or not
        dt_faces_with_id = th.cat((dt_faces, th.arange(dt_faces.shape[0], device=DEVICE).unsqueeze(-1)), dim=-1)
        opt_faces_with_id = th.cat((opt_faces, th.full((opt_faces.shape[0],), -1, device=DEVICE).unsqueeze(-1)), dim=-1)
        tmp_faces_with_id = th.cat((dt_faces_with_id, opt_faces_with_id), dim=0)
        tmp_faces_with_id = th.unique(tmp_faces_with_id, dim=0)
        _, tmp_faces_cnt = th.unique(tmp_faces_with_id[:, :-1], dim=0, return_counts=True)
        tmp_faces_cnt_cumsum = th.cumsum(tmp_faces_cnt, dim=0)
        tmp_faces_end = tmp_faces_cnt_cumsum
        tmp_faces_beg = th.cat((th.zeros(1, device=DEVICE), tmp_faces_cnt_cumsum[:-1]), dim=0)
        target_tmp_faces_end = tmp_faces_end[tmp_faces_cnt == 2]
        dt_faces_is_in_opt_faces = tmp_faces_with_id[target_tmp_faces_end - 1, -1]

        assert len(dt_faces_is_in_opt_faces) == opt_faces.shape[0], "Something wrong with the face existence check."

        dt_faces_existence = th.zeros_like(dt_faces[:, 0], dtype=th.bool)
        dt_faces_existence[dt_faces_is_in_opt_faces] = True

        ### get render layers
        mv, proj = self.mv, self.proj
        v_image_size = 1024         # some large enough resolution...
        v_num_layers = 1

        t_render_layers = RenderLayersManager(dt_points, opt_faces, mv, proj, v_image_size, v_num_layers, DEVICE)
        t_render_layers.generate()
        
        rendered_faces_cnt_map = t_render_layers.get_first_pixel_layer_face_cnt()
        rendered_faces_idx = th.tensor(list(rendered_faces_cnt_map.keys()), device=DEVICE, dtype=th.long)
        
        ### save before & after mesh
        save_path = os.path.join(self.writer.log_dir, "save/{}/before_visibility_check".format(label))
        self.save(save_path, ppos, opt_faces, pcolor, time.time() - self.global_optim_start_time, True)
        self.logger.info(f"Num. faces before visibility check: {opt_faces.shape[0]}")

        opt_faces = dt_faces[rendered_faces_idx]
        save_path = os.path.join(self.writer.log_dir, "save/{}/after_visibility_check".format(label))
        self.save(save_path, ppos, opt_faces, pcolor, time.time() - self.global_optim_start_time, True)
        self.logger.info(f"Num. faces before visibility check: {opt_faces.shape[0]}")

        dt_faces_existence = th.zeros_like(dt_faces[:, 0], dtype=th.bool)
        dt_faces_existence[rendered_faces_idx] = True

        '''
        2. Optimize point-wise real values
        '''
        image_size = self.optimize_preal_settings.image_size
        patch_size = self.optimize_preal_settings.patch_size

        renderer = Renderer(
            self.mv,
            self.proj,
            image_size,
            image_size,
            DEVICE,
        )

        optimizer, freal = self.refresh_preal_optimizer(opt_faces.shape[0])
        with th.no_grad():
            freal.data[:] = 1.0
            
        '''
        Prepare target images
        '''
        target_diffuse, target_depth, image_patches = self.prepare_target_images(
            image_size,
            patch_size
        )
        rand_image_patch_idx = th.randperm(image_patches.shape[0])
        batch_start = 0

        num_steps = self.optimize_preal_settings.num_steps
        vis_steps = self.optimize_preal_settings.vis_steps
        num_views = self.optimize_preal_settings.num_views

        start_event = th.cuda.Event(enable_timing=True)
        end_event = th.cuda.Event(enable_timing=True)

        real_reg_weight = self.optimize_preal_settings.real_reg_weight

        bar = tqdm(range(num_steps))
        for step in bar:

            '''
            Select images and patches to use in this iteration
            '''
            batch_end = batch_start + num_views
            if batch_end > rand_image_patch_idx.shape[0]:
                batch_end = rand_image_patch_idx.shape[0]

            ### (image id, patch id x, patch id y)
            t_image_patches = image_patches[rand_image_patch_idx[batch_start:batch_end]]
            if batch_end == rand_image_patch_idx.shape[0]:
                rand_image_patch_idx = th.randperm(image_patches.shape[0])
                batch_start = 0
            else:
                batch_start = batch_end

            '''
            Compute losses
            '''
            start_event.record()
            ### recon loss
            start_event.record()
            recon_loss, recon_loss_info = self.compute_rendering_loss(
                step, num_steps, renderer, target_diffuse, target_depth, t_image_patches, image_size, patch_size, ppos, pcolor, opt_faces, freal, 0.0
            )
            end_event.record()
            th.cuda.synchronize()
            recon_loss_time = start_event.elapsed_time(end_event) * 0.001

            start_event.record()
            real_regularizer = freal.mean()
            end_event.record()
            th.cuda.synchronize()
            
            loss = recon_loss + (real_regularizer * real_reg_weight)
            
            '''
            Update points.
            '''
            with th.no_grad():
                prev_freal = freal.clone()
                
            start_event.record()
            optimizer.zero_grad()
            loss.backward()
            end_event.record()
            th.cuda.synchronize()
            loss_backward_time = start_event.elapsed_time(end_event) * 0.001
 
            optimizer.step()

            '''
            Bounding.
            '''
            with th.no_grad():
                freal.data = th.clamp(freal, min=0, max=1)
                
                assert th.any(th.isnan(freal)) == False, "face real contains nan."
                assert th.any(th.isinf(freal)) == False, "face real contains inf."

            '''
            Logging
            '''
            with th.no_grad():
                prev_num_faces_on_mesh = (prev_freal > 0.5).sum()

                self.writer.add_scalar(f"{label}_loss/loss", loss, step)
                self.writer.add_scalar(f"{label}_loss/recon_loss", recon_loss, step)
                self.writer.add_scalar(f"{label}_loss/real_regularizer", real_regularizer, step)
                
                self.writer.add_scalar(f"{label}_mesh/num_faces_on_mesh", prev_num_faces_on_mesh, step)
                
                # time;
                self.writer.add_scalar(f"{label}_time/recon_loss_time", recon_loss_time, step)
                self.writer.add_scalar(f"{label}_time/loss_backward_time", loss_backward_time, step)

                # etc;
                for k, v in recon_loss_info.items():
                    if 'loss' in k:
                        self.writer.add_scalar(f"{label}_loss/{k}", v, step)
                    elif 'image' in k:
                        if step % 50 == 0:
                            self.writer.add_image(f"{label}_image/{k}", v, step, dataformats='HWC')
                
                bar.set_description("loss: {:.4f}".format(loss))

            '''
            Saving
            '''
            if step % vis_steps == 0 or step == num_steps - 1:

                mesh_faces = opt_faces[freal > 0.5]

                if len(mesh_faces) > 0:
                    self.save_step(
                        label,
                        step,
                        ppos,
                        mesh_faces,
                        pcolor,
                        time.time() - self.global_optim_start_time,
                        True
                    )

        opt_faces = opt_faces[freal > 0.5]
        self.dtfaces = opt_faces.clone()
        self.freal = th.ones_like(self.dtfaces[:, 0], dtype=th.float32)

        self.logger.info(f"Num. Faces after preal optimization: {opt_faces.shape[0]}")

        '''
        Set preal accordingly
        '''
        with th.no_grad():
            points_on_mesh = th.unique(opt_faces)
            self.preal = th.zeros_like(self.preal)
            self.preal[points_on_mesh] = 1.0

    '''
    Subdivision
    '''
    def _extract_faces_non_differentiable(self, ppos: th.Tensor, preal: th.Tensor):
        with th.no_grad():
            ### extract faces on DT
            dt = CGALDTStruct.forward(ppos)
            tets = dt.dsimp_point_id.to(dtype=th.long)
            t_faces = tets[:, [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]].reshape(-1, 3)
            t_faces = th.sort(t_faces, dim=-1)[0]
            t_faces = th.unique(t_faces, dim=0)
            t_faces_real = preal[t_faces].all(dim=-1)
            t_faces = t_faces[t_faces_real]

            ### select faces that satisfy min-ball condition
            t_faces_minball, _ = MB3_V0.forward(ppos[t_faces[:, 0]], ppos[t_faces[:, 1]], ppos[t_faces[:, 2]])
            t_faces_minball_center = t_faces_minball.center
            t_faces_minball_radius = t_faces_minball.radius

            t_faces_nearest, _ = knn_search(t_faces, t_faces_minball_center, t_faces_minball_radius, ppos)

            t_faces_sdist = projection(t_faces, t_faces_minball_center, t_faces_minball_radius, ppos, t_faces_nearest)

            final_faces = t_faces[t_faces_sdist > 0.0]

            return final_faces

    def _subdivide_by_inserting_face_midpoints(self, epoch: int):
        '''
        Subdivide the mesh.
        '''
        ppos = self.ppos
        preal = self.preal
        pcolor = self.pcolor
        
        num_ppos_before = ppos.shape[0]
        
        ### save prev mesh
        save_dir = os.path.join(self.writer.log_dir, f"save/{epoch}_subdiv")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dtfaces = self.dtfaces
        faces = dtfaces[self.freal == 1.0]         # do not use "every" faces in dt, only use our perceived ones
        
        mesh = trimesh.Trimesh(vertices=ppos.cpu().numpy(), faces=faces.cpu().numpy(), vertex_colors=pcolor.cpu().numpy())
        mesh.export(os.path.join(save_dir, "before.obj"))

        '''
        Insert new points to remove undesirable faces.
        '''
        if True:
            real_dt_faces = self._extract_faces_non_differentiable(ppos, preal)
            faces_to_preserve = faces.clone()
            faces_to_remove = tensor_subtract_1(real_dt_faces, faces_to_preserve)

            faces_to_remove_minball, _ = MB3_V0.forward(ppos[faces_to_remove[:, 0]], ppos[faces_to_remove[:, 1]], ppos[faces_to_remove[:, 2]])
            faces_to_remove_minball_center = faces_to_remove_minball.center
            
            n_points = faces_to_remove_minball_center
            
            n_preals = th.zeros((n_points.shape[0],), dtype=th.bool, device=DEVICE)
            n_pcolors = th.ones((n_points.shape[0], 3), dtype=th.float32, device=DEVICE)

            ppos = th.cat([ppos, n_points], dim=0)
            preal = th.cat([preal, n_preals], dim=0)
            pcolor = th.cat([pcolor, n_pcolors], dim=0)        

        '''
        Insert new points on desirable faces for subdivision.
        '''
        if len(faces) * 4 < TRI_SUBDIV_MAX_NUM:
            num_faces_to_subdiv = len(faces)
        elif len(faces) >= TRI_SUBDIV_MAX_NUM - 10:
            self.logger.warning("Too many faces to subdivide, stop subdivision and exit.")
            exit(1)
        else:
            num_faces_to_subdiv = int((TRI_SUBDIV_MAX_NUM - len(faces))) // 3

        faces_area = triangle_area(ppos, faces)
        faces_idx_by_area = th.argsort(faces_area, descending=True)
        faces_to_subdiv = th.zeros((len(faces),), dtype=th.bool, device=DEVICE)
        faces_to_subdiv[faces_idx_by_area[:num_faces_to_subdiv]] = True
        
        num_faces = faces.shape[0]
        faces = faces[faces_to_subdiv]
        self.logger.info(f"Subdividing {faces_to_subdiv.sum()} ({faces_to_subdiv.sum() / num_faces * 100:.2f} %) faces.")

        edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        edges = th.sort(edges, dim=-1)[0]
        edges = th.unique(edges, dim=0)

        ### insert new points on the faces
        edges_mid = 0.5 * (ppos[edges[:, 0]] + ppos[edges[:, 1]])
        edges_pcolor = 0.5 * (pcolor[edges[:, 0]] + pcolor[edges[:, 1]])

        n_points = edges_mid
        n_preals = th.ones_like(n_points[:, 0])
        n_pcolors = edges_pcolor

        ppos = th.cat([ppos, n_points], dim=0)
        preal = th.cat([preal, n_preals], dim=0)
        pcolor = th.cat([pcolor, n_pcolors], dim=0)

        self.ppos = ppos.detach().clone()
        self.preal = preal.detach().clone()
        self.pcolor = pcolor.detach().clone()

        num_ppos_after = ppos.shape[0]

        self.logger.info(f"Subdivided mesh with {num_ppos_before} points to {num_ppos_after} points.")

        ### save new mesh
        n_faces = self._extract_faces_non_differentiable(ppos, preal)
        self.dtfaces = n_faces

        mesh = trimesh.Trimesh(vertices=ppos.cpu().numpy(), faces=n_faces.cpu().numpy(), vertex_colors=pcolor.cpu().numpy())
        mesh.export(os.path.join(save_dir, "after.obj"))

    def subdivide(self, epoch: int):

        self._subdivide_by_inserting_face_midpoints(epoch)
        
    '''
    Non-manifoldness removal
    '''
    def remove_nonmanifold(self):

        verts = self.ppos.clone()
        faces = self.dtfaces.clone()

        # render layer manager to count number of rendered pixels per face
        mv, proj = self.mv, self.proj
        image_size = 256
        num_layers = 5
        render_layers_manager = RenderLayersManager(verts, faces, mv, proj, image_size, num_layers, DEVICE)
        
        # consider every face in DT
        faces = render_layers_manager.faces
        face_existence = render_layers_manager.faces_existence

        n_face_existence = cpp_remove_non_manifold(verts, faces, face_existence, render_layers_manager)

        ### save mesh
        manifold_faces = faces[n_face_existence.to(device=faces.device)]
        save_dir = os.path.join(self.logdir, "save", "manifold")
        self.save(save_dir, verts, manifold_faces, self.pcolor, time.time() - self.global_optim_start_time, True)

        self.dtfaces = manifold_faces.clone()


def load_input_multi_view_images(input_path):
    try:
        mv_path = os.path.join(input_path, "mv.npy")
        proj_path = os.path.join(input_path, "proj.npy")
        
        mv = th.from_numpy(np.load(mv_path)).to(DEVICE)
        proj = th.from_numpy(np.load(proj_path)).to(DEVICE)

        num_images = mv.shape[0]

        diffuse = []
        depth = []
        for i in range(num_images):
            diffuse_path = os.path.join(input_path, "diffuse_{}.png".format(i))
            depth_path = os.path.join(input_path, "depth_{}.png".format(i))
            
            diffuse_i = np.array(Image.open(diffuse_path), dtype=np.float32) / 255.0
            depth_i = np.array(Image.open(depth_path), dtype=np.float32) / 255.0

            diffuse.append(th.from_numpy(diffuse_i).to(DEVICE))
            depth.append(th.from_numpy(depth_i).to(DEVICE))
        
        diffuse = th.stack(diffuse, dim=0)
        depth = th.stack(depth, dim=0)
        
    except:
        raise ValueError("Input multi-view images not found.")
    
    return mv, proj, diffuse, depth

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp/config/d3/mvrecon.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-path", type=str, default="input/3d/mvrecon/toad")
    parser.add_argument("--no-log-time", action='store_true')
    args = parser.parse_args()

    # load settings from yaml file;
    with open(args.config, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    DEVICE = settings['device']
    settings['args']['seed'] = args.seed

    '''
    Set up logdir and logger
    '''
    # setup logdir;
    logdir = settings['log_dir']
    if not args.no_log_time:
        logdir = logdir + time.strftime("/%Y_%m_%d_%H_%M_%S")
    logdir = setup_logdir(logdir)

    # setup logger;
    logger = get_logger("mvrecon_3d", os.path.join(logdir, "run.log"))
    
    # save settings;
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        yaml.dump(settings, f)
    th.random.manual_seed(args.seed)

    '''
    Input multi-view images and cameras
    '''
    input_path = args.input_path
    try:
        mv, proj, diffuse, depth = load_input_multi_view_images(input_path)
        image_size = diffuse.shape[1]
    except ValueError as e:
        logger.exception(str(e))
        sys.exit(1)
    
    logger.info(f"Num. Input Images: {mv.shape[0]}")
    logger.info(f"Image Size: {image_size}")

    '''
    Arguments: default values;
    '''
    remove_nonmanifold = settings['args']['remove_nonmanifold']
    optimize_color = settings['args']['optimize_color']

    # tetrahedral grid initialization;
    init_args = settings['args']['init_args']

    # init preal;
    init_preal_args = edict(settings['args']['init_preal'])

    # num epochs;
    num_epochs = int(settings['args']['num_epochs'])

    # default epoch args:
    default_epoch_args = settings['args']['default_epoch_args']

    # epoch-specific args:
    epoch_args = settings['args']['epoch_args']

    gt_diffuse_map = diffuse.cpu()
    gt_depth_map = depth.cpu()
    del diffuse, depth      # save memory on GPU
    th.cuda.memory.empty_cache()
    
    optimizer = MVRecon(
        logdir,
        logger,

        remove_nonmanifold,
        optimize_color,

        mv,
        proj,
        image_size,

        gt_diffuse_map,
        gt_depth_map,

        init_args,
        init_preal_args,

        num_epochs,

        default_epoch_args,
        epoch_args,
    )

    try:
        optimizer.optimize()
    except Exception as e:
        logger.exception(str(e))
        sys.exit(1)