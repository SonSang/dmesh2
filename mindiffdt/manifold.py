from dmesh2_renderer import LayeredRenderer
from mindiffdt.cgaldt import CGALDTStruct
from mindiffdt.tetra import TetraSet
from mindiffdt.utils import tensor_intersect_idx
import torch as th

class RenderLayersManager:

    def __init__(self, ppos, faces, mv, proj, image_size, num_layers, device):
        '''
        Keep track of visible faces for each pixel.
        For given points, construct DT and render the faces in DT which are labeled as "exist" for accurate depth testing.

        @ ppos: (N, 3) tensor, points to render. Run DT on these points and render faces in DT
        @ faces: (F, 3) tensor, faces that are labeled as "exist" at initial state, these faces should be subset of faces in DT
        @ mv: (B, 4, 4) tensor, model view matrix
        @ proj: (B, 4, 4) tensor, projection matrix
        @ image_size: int, image size
        @ num_layers: int, number of layers to render
        @ device: torch.device, device to run on
        '''

        self.renderer = LayeredRenderer(mv, proj, image_size, image_size, device)

        '''
        Prepare DT data structure for rendering
        '''
        ### run dt
        dt_result = CGALDTStruct.forward(ppos)
        tets = dt_result.dsimp_point_id.to(dtype=th.long)
        
        ### construct tetra set
        tetra_set = TetraSet(ppos, tets)

        ### get dt data
        dt_points = ppos
        dt_faces = tetra_set.faces
        dt_tets = tetra_set.tets
        dt_face_tet = tetra_set.face_tet
        dt_tet_face = tetra_set.tet_faces

        ### get dtfaces existence
        dt_faces_existence = th.zeros(len(dt_faces), dtype=th.bool, device=ppos.device)

        ### mark faces in [faces] as "exist" in [dt_faces_existence]
        dt_faces, faces = th.sort(dt_faces, dim=-1)[0], th.sort(faces, dim=-1)[0]
        dt_faces, faces = th.unique(dt_faces, dim=0), th.unique(faces, dim=0)
        dt_faces_in_faces, _ = tensor_intersect_idx(dt_faces, faces)
        dt_faces_existence[dt_faces_in_faces] = True

        '''
        Attr
        '''
        self.num_layers = num_layers
        self.pixel_layers = None

        self.verts = dt_points
        self.faces = dt_faces
        self.tets = dt_tets
        self.face_tet = dt_face_tet
        self.tet_faces = dt_tet_face
        self.faces_existence = dt_faces_existence

        ### first generation
        self.generate()
        
    def generate(self):
        pixel_layers = []
        for i in range(len(self.renderer.mv)):
            tmp_render_layers, _ = self.renderer.generate(
                [i],
                self.verts,
                self.faces,
                self.tets,
                self.face_tet,
                self.tet_faces,
                self.faces_existence,
                self.num_layers
            )
            tmp_pixel_layers = tmp_render_layers[0].reshape(-1, self.num_layers)
            tmp_pixel_layers = tmp_pixel_layers[tmp_pixel_layers[..., 0] != -1]     # remove pixels that do not contain any faces
            pixel_layers.append(tmp_pixel_layers)
        pixel_layers = th.cat(pixel_layers, dim=0)

        self.pixel_layers = pixel_layers

    def get_first_pixel_layer_face_cnt(self):
        # get face idx and cnt of the first pixel layer, which is visible right now
        u_face_idx, u_face_cnt = th.unique(self.pixel_layers[..., 0], return_counts=True)
        u_face_idx_is_valid = u_face_idx >= 0

        u_face_idx = u_face_idx[u_face_idx_is_valid]
        u_face_cnt = u_face_cnt[u_face_idx_is_valid]

        # convert to dict for easy access
        u_face_idx = u_face_idx.cpu().tolist()
        u_face_cnt = u_face_cnt.cpu().tolist()

        face_cnt_map = dict(zip(u_face_idx, u_face_cnt))
        return face_cnt_map

    def remove_face_from_pixel_layers(self, face_id: int):
        # remove face_id from pixel_layers and fill in the gap
        pixel_layers = self.pixel_layers
        
        target_pixel_layer_idx = th.where(pixel_layers == face_id)
        pixel_layers[target_pixel_layer_idx] = -2       # mark as removed, -1 is for empty pixel

        # push removed faces to the end
        num_layers = pixel_layers.shape[1]
        target_pixel_layer_0 = target_pixel_layer_idx[0]
        target_pixel_layer_1 = target_pixel_layer_idx[1]

        while True:
            # find pixels to push
            # 1. target pixel's depth should be smaller the total depth
            # 2. target pixel's next layer should exist
            pixels_is_to_push = \
                (target_pixel_layer_1 < num_layers - 1) & \
                (pixel_layers[target_pixel_layer_0, th.clamp(target_pixel_layer_1 + 1, max=num_layers - 1)] >= 0)

            if not th.any(pixels_is_to_push):
                break

            # push pixels
            pixel_layer_to_push_0 = target_pixel_layer_0[pixels_is_to_push]
            pixel_layer_to_push_1 = target_pixel_layer_1[pixels_is_to_push]

            pixel_layers[pixel_layer_to_push_0, pixel_layer_to_push_1] = \
                pixel_layers[pixel_layer_to_push_0, pixel_layer_to_push_1 + 1]
            pixel_layers[pixel_layer_to_push_0, pixel_layer_to_push_1 + 1] = -2

            target_pixel_layer_1[pixels_is_to_push] += 1

        self.pixel_layers = pixel_layers

    def is_first_pixel_layer_complete(self):
        # check if the first pixel layer is complete
        u_face_idx, u_face_cnt = th.unique(self.pixel_layers[..., 0], return_counts=True)
        return th.all(u_face_idx != -2)

    '''
    CPP interfaces
    '''
    def get_face_weight(self):
        return self.get_first_pixel_layer_face_cnt()

    def remove_face(self, face_id: int):
        self.faces_existence[face_id] = False
        self.remove_face_from_pixel_layers(face_id)
        if not self.is_first_pixel_layer_complete():
            self.generate()