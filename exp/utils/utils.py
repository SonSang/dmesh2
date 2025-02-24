import torch as th
import numpy as np
import trimesh
import os
from pytorch3d.ops.knn import knn_points

def setup_logdir(path: str):
    if os.path.exists(path) == False:
        os.makedirs(path)
    else:
        raise ValueError(f"Directory already exists: {path}")
    return path

def import_mesh(fname, device, scale=0.8):
    if fname.endswith(".obj"):
        target_mesh = trimesh.load_mesh(fname, 'obj')
    elif fname.endswith(".stl"):
        target_mesh = trimesh.load_mesh(fname, 'stl')    
    elif fname.endswith(".ply"):
        target_mesh = trimesh.load_mesh(fname, 'ply')
    else:
        raise ValueError(f"unknown mesh file type: {fname}")

    target_vertices, target_faces = target_mesh.vertices,target_mesh.faces
    target_vertices, target_faces = \
        th.tensor(target_vertices, dtype=th.float32, device=device), \
        th.tensor(target_faces, dtype=th.long, device=device)
    
    # normalize to fit mesh into a sphere of radius [scale];
    if scale > 0:
        target_vertices = target_vertices - target_vertices.mean(dim=0, keepdim=True)
        max_norm = th.max(th.norm(target_vertices, dim=-1)) + 1e-6
        target_vertices = (target_vertices / max_norm) * scale

    vertex_normals = th.tensor(target_mesh.vertex_normals, dtype=th.float32, device=device)
    vertex_colors = th.ones_like(target_vertices)

    ### get duplicate verts and define vertex normal from face normals
    face_vert_0 = target_vertices[target_faces[:, 0]]   # [F, 3]
    face_vert_1 = target_vertices[target_faces[:, 1]]   # [F, 3]
    face_vert_2 = target_vertices[target_faces[:, 2]]   # [F, 3]
    face_normal = th.cross(face_vert_1 - face_vert_0, face_vert_2 - face_vert_0, dim=-1)    # [F, 3]
    face_normal = face_normal / (th.norm(face_normal, dim=-1, p=2, keepdim=True) + 1e-6)

    n_verts = th.cat([face_vert_0, face_vert_1, face_vert_2], dim=0)   # [3F, 3]
    n_verts_normal = th.cat([face_normal, face_normal, face_normal], dim=0)   # [3F, 3]
    n_faces = th.arange(0, target_faces.shape[0], device=device)        # [F]
    n_faces = th.stack([n_faces, n_faces + target_faces.shape[0], n_faces + 2 * target_faces.shape[0]], dim=-1)  # [F, 3]
    n_verts_colors = th.ones_like(n_verts)   # [3F, 3]

    return n_verts, n_faces, n_verts_normal, n_verts_colors

def import_textured_mesh(fname, device, scale=0.8):
    assert fname.endswith(".glb"), "Only glb file is supported"

    # Load the mesh from file
    mesh = trimesh.load(fname)

    ### extract vertices, faces, vertex normals, and vertex colors
    verts = []
    faces = []
    normals = []
    textures = []
    uvs = []

    num_verts = 0
    for gi in mesh.geometry.keys():
        t_verts = np.array(mesh.geometry[gi].vertices).astype(np.float32)
        t_faces = np.array(mesh.geometry[gi].faces).astype(np.int32)
        t_normals = np.array(mesh.geometry[gi].vertex_normals).astype(np.float32)

        assert len(mesh.graph.geometry_nodes[gi]) == 1, 'Multi-node geometry is not supported'
        node_name = mesh.graph.geometry_nodes[gi][0]
        node_transform = np.array(mesh.graph.get(node_name, 'world')[0]).astype(np.float32)
        
        t_verts = np.concatenate([t_verts, np.ones((t_verts.shape[0], 1), dtype=np.float32)], axis=-1)
        t_verts = t_verts @ node_transform.T
        t_verts = t_verts[:, :3]

        t_normals = np.concatenate([t_normals, np.zeros((t_normals.shape[0], 1), dtype=np.float32)], axis=-1)
        t_normals = t_normals @ node_transform.T
        t_normals = t_normals[:, :3]

        visual = mesh.geometry[gi].visual
        if isinstance(visual, trimesh.visual.texture.TextureVisuals):
            material = visual.material

            if material.baseColorTexture is None:
                if material.main_color is not None:
                    base_color = np.zeros((1, 1, 3), dtype=np.int32)
                    base_color[0, 0] = np.array(material.main_color).astype(np.int32)[:3]
                    t_uv = np.zeros((t_verts.shape[0], 2), dtype=np.float32)
                else:
                    raise ValueError('Unsupported texture type')
            else:
                base_color_im = material.baseColorTexture.getdata()
                base_color_width, base_color_height = base_color_im.size
                if base_color_im.mode == 'RGB':
                    base_color = np.asarray(base_color_im).reshape(base_color_height, base_color_width, 3)
                elif base_color_im.mode == 'RGBA':
                    base_color = np.asarray(base_color_im).reshape(base_color_height, base_color_width, 4)
                    base_color = base_color[:, :, :3]
                else:
                    try:
                        base_color = base_color_im.convert('RGB')
                        base_color = np.asarray(base_color).reshape(base_color_height, base_color_width, 3)
                    except:
                        raise ValueError('Unsupported texture type')
                t_uv = np.array(visual.uv).astype(np.float32)
                t_uv[:, 1] = 1.0 - t_uv[:, 1]   # flip y
                
        else:

            raise ValueError('Unsupported texture type')
        
        verts.append(th.tensor(t_verts, dtype=th.float32, device=device))
        faces.append(th.tensor(t_faces, dtype=th.long, device=device))
        normals.append(th.tensor(t_normals, dtype=th.float32, device=device))
        # colors.append(t_colors)
        textures.append(th.tensor(base_color, dtype=th.float32, device=device))
        uvs.append(th.tensor(t_uv, dtype=th.float32, device=device))

        num_verts += t_verts.shape[0]


    ### normalize vertices
    if scale > 0:
        total_verts = th.concatenate(verts, axis=0)
        total_verts_mean = total_verts.mean(dim=0, keepdim=True)
        total_verts = total_verts - total_verts_mean
        total_verts_norm = th.norm(total_verts, dim=-1, p=2)

        max_norm = th.max(total_verts_norm) + 1e-6
        scale_factor = (scale / max_norm)

        for i in range(len(verts)):
            verts[i] = (verts[i] - total_verts_mean) * scale_factor

    return verts, faces, normals, textures, uvs

'''
Point sampling
'''
def sample_bary(num_faces: int, device):    
    face_bary = th.zeros((num_faces, 3), dtype=th.float32, device=device)

    face_coords = th.rand((num_faces, 2), dtype=th.float32, device=device)
    face_coords_valid = (face_coords[:, 0] + face_coords[:, 1]) < 1.0
    face_coords[~face_coords_valid] = 1.0 - face_coords[~face_coords_valid]
    assert th.all(face_coords[:, 0] + face_coords[:, 1] <= 1.0), "face coords should be valid."

    face_bary[:, 0] = 1.0 - face_coords[:, 0] - face_coords[:, 1]
    face_bary[:, 1] = face_coords[:, 0]
    face_bary[:, 2] = face_coords[:, 1]

    return face_bary

def sample_points_on_mesh(positions: th.Tensor,
                            faces: th.Tensor,
                            num_points: int,
                            device):

    '''
    Sample points on faces.
    The points are distributed based on the size of triangular faces.

    @ return
    @ sample_positions: [num_points, 3]
    '''

    with th.no_grad():
        # compute face areas;
        face_areas = triangle_area(positions, faces)
        assert th.all(face_areas >= 0.0), "face area should be non-negative."
        assert th.sum(face_areas) > 0, "sum of face area should be positive."

        # sample faces;
        with th.no_grad():
            face_idx = th.multinomial(
                face_areas, 
                num_points, 
                replacement=True
            )

        face_bary = sample_bary(num_points, device)

    # compute sample positions;
    sample_positions = th.sum(
        positions[faces[face_idx]] * face_bary.unsqueeze(-1),
        dim=-2
    )

    # compute sample normals;
    sample_normals = th.nn.functional.normalize(
        th.cross(
            positions[faces[face_idx][:, 1]] - positions[faces[face_idx][:, 0]],
            positions[faces[face_idx][:, 2]] - positions[faces[face_idx][:, 0]],
            dim=-1
        ), dim=-1
    )

    return sample_positions, sample_normals

'''
Differentiable max & min
'''
def dmax(val: th.Tensor, k: float = 100):
    '''
    @ val: [# elem, N]
    '''
    with th.no_grad():
        e_val_denom = val * k
        e_val_denom_max = th.max(e_val_denom, dim=-1, keepdim=True)[0]
        e_val_denom = e_val_denom - e_val_denom_max

        e_val_denom = th.exp(e_val_denom)
        e_val_nom = th.sum(e_val_denom, dim=-1, keepdim=True)
        e_val = e_val_denom / e_val_nom

    return th.sum(e_val * val, dim=-1)

def dmin(val: th.Tensor, k: float = 1000):
    return -dmax(-val, k)

def gdmax(val: th.Tensor, dim: int, k: float = 100):
    '''
    generalization of dmax
    '''
    with th.no_grad():
        e_val_denom = val * k
        e_val_denom_max = th.max(e_val_denom, dim=dim, keepdim=True)[0]
        e_val_denom = e_val_denom - e_val_denom_max

        e_val_denom = th.exp(e_val_denom)
        e_val_nom = th.sum(e_val_denom, dim=dim, keepdim=True)
        e_val = e_val_denom / e_val_nom

    return th.sum(e_val * val, dim=dim)

def gdmin(val: th.Tensor, dim: int, k: float = 100):
    return -gdmax(-val, dim, k)

'''
KNN
'''
def run_knn(src: th.Tensor, tgt: th.Tensor, k: int):

    '''
    @ src: [M, D]
    @ tgt: [N, D]
    '''

    assert src.ndim == 2, "Invalid input dimensions"
    assert tgt.ndim == 2, "Invalid input dimensions"

    assert th.isnan(src).sum() == 0, "NaN in src"
    assert th.isinf(src).sum() == 0, "Inf in src"
    assert th.isnan(tgt).sum() == 0, "NaN in tgt"
    assert th.isinf(tgt).sum() == 0, "Inf in tgt"
    
    t_src = src.unsqueeze(0)
    t_tgt = tgt.unsqueeze(0)
    try:
        knn_result = knn_points(t_src, t_tgt, K=k)
    except:
        print(f"src: {src.shape}, tgt: {tgt.shape}")
        raise ValueError("Error in knn_points")

    knn_idx = knn_result.idx.squeeze(0)
    knn_dist = th.sqrt(knn_result.dists.squeeze(0))

    return knn_idx, knn_dist

def triangle_aspect_ratio(points: th.Tensor, faces: th.Tensor):
    '''
    Compute aspect ratio of triangles, which is in range [1, inf).
    '''

    face_vertex_0 = points[faces[:, 0]]
    face_vertex_1 = points[faces[:, 1]]
    face_vertex_2 = points[faces[:, 2]]

    face_edge_dir_0 = face_vertex_1 - face_vertex_0
    face_edge_dir_1 = face_vertex_2 - face_vertex_0
    face_edge_dir_2 = face_vertex_2 - face_vertex_1

    face_edge_len_0 = th.norm(face_edge_dir_0, dim=-1)
    face_edge_len_1 = th.norm(face_edge_dir_1, dim=-1)
    face_edge_len_2 = th.norm(face_edge_dir_2, dim=-1)

    face_area_0 = th.norm(th.cross(face_edge_dir_0, face_edge_dir_1, dim=-1), dim=-1)
    face_area_1 = th.norm(th.cross(face_edge_dir_1, face_edge_dir_2, dim=-1), dim=-1)
    face_area_2 = th.norm(th.cross(face_edge_dir_2, face_edge_dir_0, dim=-1), dim=-1)

    face_height_0 = face_area_0 / (face_edge_len_0 + 1e-6)
    face_height_1 = face_area_1 / (face_edge_len_1 + 1e-6)
    face_height_2 = face_area_2 / (face_edge_len_2 + 1e-6)

    max_face_edge_len = th.max(th.stack([face_edge_len_0, face_edge_len_1, face_edge_len_2], dim=-1), dim=-1)[0]
    min_face_height = th.min(th.stack([face_height_0, face_height_1, face_height_2], dim=-1), dim=-1)[0]
    min_face_height = min_face_height * (2.0 / np.sqrt(3.0))

    ar = max_face_edge_len / (min_face_height + 1e-6)

    return ar

def triangle_area(points: th.Tensor, faces: th.Tensor):
    '''
    Compute area of triangles.
    '''

    face_vertex_0 = points[faces[:, 0]]
    face_vertex_1 = points[faces[:, 1]]
    face_vertex_2 = points[faces[:, 2]]

    face_edge_dir_0 = face_vertex_1 - face_vertex_0
    face_edge_dir_1 = face_vertex_2 - face_vertex_0

    face_area = th.norm(th.cross(face_edge_dir_0, face_edge_dir_1, dim=-1), dim=-1) * 0.5

    return face_area