import os
import argparse
import numpy as np
import torch as th
import trimesh
import nvdiffrast.torch as dr
from PIL import Image

from input.common import DOMAIN, LIGHT_DIR

DEVICE = 'cuda:0'

'''
=======================================================
Implementations from [Continous Remeshing For Inverse Rendering](https://github.com/Profactor/continuous-remeshing).
=======================================================
'''
def _translation(x, y, z, device):
    return th.tensor([[1., 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]],device=device) #4,4

def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = th.zeros([4,4],device=device)
    p[0,0] = 2*n/(r-l)
    p[0,2] = (r+l)/(r-l)
    p[1,1] = 2*n/(t-b) * (-1 if flip_y else 1)
    p[1,2] = (t+b)/(t-b)
    p[2,2] = -(f+n)/(f-n)
    p[2,3] = -(2*f*n)/(f-n)
    p[3,2] = -1
    return p #4,4

def make_star_cameras(az_count,pol_count,distance:float=10.,r=None, n=None, f=None, image_size=[512,512],device='cuda'):
    if r is None:
        r = 1/distance
    if n is None:
        n = 1
    if f is None:
        f = 50
    A = az_count
    P = pol_count
    C = A * P

    phi = th.arange(0,A) * (2*th.pi/A)
    phi_rot = th.eye(3,device=device)[None,None].expand(A,1,3,3).clone()
    phi_rot[:,0,2,2] = phi.cos()
    phi_rot[:,0,2,0] = -phi.sin()
    phi_rot[:,0,0,2] = phi.sin()
    phi_rot[:,0,0,0] = phi.cos()
    
    theta = th.arange(1,P+1) * (th.pi/(P+1)) - th.pi/2
    theta_rot = th.eye(3,device=device)[None,None].expand(1,P,3,3).clone()
    theta_rot[0,:,1,1] = theta.cos()
    theta_rot[0,:,1,2] = -theta.sin()
    theta_rot[0,:,2,1] = theta.sin()
    theta_rot[0,:,2,2] = theta.cos()

    mv = th.empty((C,4,4), device=device)
    mv[:] = th.eye(4, device=device)
    mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(C,3,3)
    mv = _translation(0, 0, -distance, device) @ mv

    return mv, _projection(r,device, n=n, f=f)

'''
Render functions: adaptation from [Continous Remeshing For Inverse Rendering](https://github.com/Profactor/continuous-remeshing).
'''
def render_textured_mesh(verts, normals, faces, textures, uvs, mv, proj, image_size):
    '''
    @ verts: list of tensors, each tensor is V,3
    @ normals: list of tensors, each tensor is V,3
    @ faces: list of tensors, each tensor is F,3
    @ textures: list of tensors, each tensor is H,W,3
    @ uvs: list of tensors, each tensor is V,2
    @ mv: C,4,4
    @ proj: C,4,4
    @ image_size: tuple of int
    '''
    mvp = proj @ mv
    eps = 1e-4
    glctx = dr.RasterizeCudaContext()

    num_elements = len(verts)
        
    final_diffuse = None
    final_depth = None

    for ei in range(num_elements):
        ### render element by element
        curr_verts = verts[ei]
        curr_normals = normals[ei]
        curr_faces = faces[ei]
        curr_textures = textures[ei]
        curr_uvs = uvs[ei]

        V = curr_verts.shape[0]
        curr_faces = curr_faces.type(th.int32)
        curr_vert_hom = th.cat((curr_verts, th.ones(V,1,device=curr_verts.device)),axis=-1) #V,3 -> V,4
        curr_verts_clip = curr_vert_hom @ mvp.transpose(-2,-1) #C,V,4
        curr_rast_out, _ = dr.rasterize(glctx, 
                                curr_verts_clip, 
                                curr_faces, 
                                resolution=image_size, 
                                grad_db=False) #C,H,W,4
        
        # view space normal;
        curr_vert_normals_hom = th.cat((curr_normals, th.zeros(V,1,device=curr_verts.device)),axis=-1) #V,3 -> V,4
        curr_vert_normals_view = curr_vert_normals_hom @ mv.transpose(-2,-1) #C,V,4
        curr_vert_normals_view = curr_vert_normals_view[..., :3] #C,V,3
        curr_vert_normals_view = curr_vert_normals_view.contiguous()

        # view space lightdir;
        lightdir = th.tensor(LIGHT_DIR, dtype=th.float32, device=curr_verts.device) #3
        lightdir = lightdir.view((1, 1, 1, 3)) #1,1,1,3

        ### interpolation: normal and texture coords
        # normal;
        curr_pixel_normals_view, _ = dr.interpolate(curr_vert_normals_view, curr_rast_out, curr_faces)  #C,H,W,3
        curr_pixel_normals_view[curr_pixel_normals_view[..., 2] > 0.] = \
            -curr_pixel_normals_view[curr_pixel_normals_view[..., 2] > 0.]
        curr_pixel_normals_view = curr_pixel_normals_view / th.clamp(th.norm(curr_pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5)
        # uv;
        curr_uvs_view, _ = dr.interpolate(curr_uvs[None, ...].contiguous(), curr_rast_out, curr_faces)  #C,H,W,2

        ### coloring
        # diffuse;
        curr_pixel_diffuse = th.sum(lightdir * curr_pixel_normals_view, -1, keepdim=True)           #C,H,W,1
        curr_pixel_diffuse = th.clamp(curr_pixel_diffuse, min=0.0, max=1.0)
        # pixel color;
        curr_pixel_colors = dr.texture(curr_textures[None, ...].contiguous(), curr_uvs_view) #C,H,W,3
        curr_pixel_colors = curr_pixel_colors / 255.0
        
        curr_diffuse = curr_pixel_diffuse * curr_pixel_colors       # C,H,W,3
        curr_diffuse[curr_rast_out[..., -1] == 0] = 0.0         # exclude background;

        ### depth
        curr_verts_clip_w = curr_verts_clip[..., [3]]
        curr_verts_clip_w[th.logical_and(curr_verts_clip_w >= 0.0, curr_verts_clip_w < eps)] = eps
        curr_verts_clip_w[th.logical_and(curr_verts_clip_w < 0.0, curr_verts_clip_w > -eps)] = -eps

        curr_verts_depth = (curr_verts_clip[..., [2]] / curr_verts_clip_w)     #C,V,1
        curr_depth, _ = dr.interpolate(curr_verts_depth, curr_rast_out, curr_faces) #C,H,W,1    range: [-1, 1], -1 is near, 1 is far
        curr_depth = (curr_depth + 1.) * 0.5      # since depth in [-1, 1], normalize to [0, 1]

        curr_depth[curr_rast_out[..., -1] == 0] = 1.0         # exclude background;
        curr_depth = 1 - curr_depth                           # C,H,W,1

        if final_diffuse is None:
            final_diffuse = curr_diffuse
            final_depth = curr_depth
        else:
            pixel_idx = curr_depth > final_depth
            final_diffuse[pixel_idx.expand(-1, -1, -1, 3)] = curr_diffuse[pixel_idx.expand(-1, -1, -1, 3)]
            final_depth[pixel_idx] = curr_depth[pixel_idx]

    return final_diffuse, final_depth       # C,H,W,3; C,H,W,1

def render_non_textured_mesh(verts, normals, faces, mv, proj, image_size):
    '''
    @ verts: V,3
    @ normals: V,3
    @ faces: F,3
    @ mv: C,4,4
    @ proj: C,4,4
    @ image_size: tuple of int
    '''

    mvp = proj @ mv
    eps = 1e-4
    glctx = dr.RasterizeCudaContext()

    V = verts.shape[0]
    faces = faces.type(th.int32)
    vert_hom = th.cat((verts, th.ones(V,1,device=verts.device)),axis=-1) #V,3 -> V,4
    verts_clip = vert_hom @ mvp.transpose(-2,-1) #C,V,4
    rast_out,_ = dr.rasterize(glctx, 
                            verts_clip, 
                            faces, 
                            resolution=image_size, 
                            grad_db=False) #C,H,W,4

    # view space normal;
    vert_normals_hom = th.cat((normals, th.zeros(V,1,device=verts.device)),axis=-1) #V,3 -> V,4
    vert_normals_view = vert_normals_hom @ mv.transpose(-2,-1) #C,V,4
    vert_normals_view = vert_normals_view[..., :3] #C,V,3
    # in the view space, normals should be oriented toward viewer, 
    # so the z coordinates should be negative;
    vert_normals_view[vert_normals_view[..., 2] > 0.] = \
        -vert_normals_view[vert_normals_view[..., 2] > 0.]
    vert_normals_view = vert_normals_view.contiguous()

    # view space lightdir;
    lightdir = th.tensor(LIGHT_DIR, dtype=th.float32, device=verts.device) #3
    lightdir = lightdir.view((1, 1, 1, 3)) #1,1,1,3

    # normal;
    pixel_normals_view, _ = dr.interpolate(vert_normals_view, rast_out, faces)  #C,H,W,3
    pixel_normals_view = pixel_normals_view / th.clamp(th.norm(pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5)
    diffuse = th.sum(lightdir * pixel_normals_view, -1, keepdim=True)           #C,H,W,1
    diffuse = th.clamp(diffuse, min=0.0, max=1.0)
    diffuse = diffuse[..., [0, 0, 0]] #C,H,W,3
    
    # depth;
    verts_clip_w = verts_clip[..., [3]]
    verts_clip_w[th.logical_and(verts_clip_w >= 0.0, verts_clip_w < eps)] = eps
    verts_clip_w[th.logical_and(verts_clip_w < 0.0, verts_clip_w > -eps)] = -eps

    verts_depth = (verts_clip[..., [2]] / verts_clip_w)     #C,V,1
    depth, _ = dr.interpolate(verts_depth, rast_out, faces) #C,H,W,1    range: [-1, 1], -1 is near, 1 is far
    depth = (depth + 1.) * 0.5      # since depth in [-1, 1], normalize to [0, 1]
    
    depth[rast_out[..., -1] == 0] = 1.0         # exclude background;
    depth = 1 - depth                           # C,H,W,1

    return diffuse, depth

'''
Mesh Importer
'''
def import_non_textured_mesh(fname, device, scale=0.8):
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

    ### get duplicate verts and define vertex normal from face normals
    ### @ this is because there was abnormal shading when using vertex normals from the mesh directly
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
    assert fname.endswith(".glb"), "Only glb file is supported for textured mesh"

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

def save_image(img, path):
    with th.no_grad():
        img = img.detach().cpu().numpy()
        img = img * 255.0
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="data/3d/98576.stl")
    parser.add_argument("--output-path", type=str, default="input/3d/mvrecon/")
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--save-gt-mesh", action='store_true')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    num_views = int(args.num_views)
    image_size = int(args.image_size)
    save_gt_mesh = args.save_gt_mesh

    ### adjust output path
    file_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_path, file_name)
    os.makedirs(output_path, exist_ok=True)

    '''
    Load mesh.
    '''
    is_textured_mesh = input_path.endswith(".glb")
    if is_textured_mesh:
        print("Loading textured mesh...")

        verts, faces, vnormals, textures, uvs = import_textured_mesh(input_path, DEVICE, scale=DOMAIN)
        num_verts = np.sum([verts[i].shape[0] for i in range(len(verts))])
        num_faces = np.sum([faces[i].shape[0] for i in range(len(faces))])
        optimize_color = True
    else:
        print("Loading non-textured mesh...")
        
        verts, faces, vnormals, vcolors = import_non_textured_mesh(input_path, DEVICE, scale=DOMAIN)
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]
        optimize_color = False
    
    print("Number of vertices: ", num_verts)
    print("Number of faces: ", num_faces)
    
    ### save gt mesh
    if save_gt_mesh:
        gt_verts = th.cat(verts, dim=0)
        acc_num_verts = 0
        gt_faces = []
        for i in range(len(verts)):
            gt_faces.append(faces[i] + acc_num_verts)
            acc_num_verts += verts[i].shape[0]
        gt_faces = th.cat(gt_faces, dim=0)
        mesh = trimesh.base.Trimesh(vertices=gt_verts.cpu().numpy(), faces=gt_faces.cpu().numpy())
        mesh.export(os.path.join(output_path, "gt_mesh.obj"))

    '''
    Render the loaded mesh.
    '''
    mv, proj = make_star_cameras(num_views, num_views, distance=2.0, r=0.6, n=1.0, f=3.0)
    proj = proj.unsqueeze(0).expand(mv.shape[0], -1, -1)

    if input_path.endswith(".glb"):
        gt_diffuse_map, gt_depth_map = render_textured_mesh(verts, vnormals, faces, textures, uvs, mv, proj, [image_size, image_size])
    else:
        gt_diffuse_map, gt_depth_map = render_non_textured_mesh(verts, vnormals, faces, mv, proj, [image_size, image_size])

    gt_depth_map = gt_depth_map[..., [0,0,0]] # C,H,W,1 -> C,H,W,3

    print("Rendering done.")
    
    # save gt images;
    for i in range(len(gt_diffuse_map)):
        save_image(gt_diffuse_map[i], os.path.join(output_path, "diffuse_{}.png".format(i)))
        save_image(gt_depth_map[i], os.path.join(output_path, "depth_{}.png".format(i)))

    print("Images saved.")

    # save camera parameters;
    np.save(os.path.join(output_path, "mv.npy"), mv.cpu().numpy())
    np.save(os.path.join(output_path, "proj.npy"), proj.cpu().numpy())

    print("Camera parameters saved.")