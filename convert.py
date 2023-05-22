import torch
import json
from binvox_rw import read_as_3d_array
from skimage.measure import marching_cubes
from sklearn.preprocessing import normalize
import numpy as np
import shutil
import os
import sys
def sample_triangle_mesh_with_normals(vertices: torch.Tensor, 
                                      faces: torch.Tensor,
                                      num_samples: int, 
                                      eps: float = 1e-5):
    r""" Uniformly samples the surface of a mesh, and extracts the associated surface normals.

    NOTE: we assume this is a batch of different instantiations of the same mesh.
    Thus, F is merely |F| x 3 and V can be B x |V|=N x 3, since
        |V| is always the same (though V changes) and F is always the same (as is |F|).

    Args:
        vertices (torch.Tensor): Vertices of the mesh (shape:
            :math:`B x N \times 3`, where :math:`N` is the number of vertices)
        faces (torch.LongTensor): Faces of the mesh (shape: :math:`F \times 3`,
            where :math:`F` is the number of faces).
        num_samples (int): Number of points to sample
        eps (float): A small number to prevent division by zero
                     for small surface areas.
    Returns:
        V, n_hat (torch.Tensor, torch.tensor): 
            Uniformly sampled points from the triangle mesh along with their normals.
    """
    print(vertices.shape)
    B, nV, _ = vertices.shape
    F, _ = faces.shape
    # B, nV = vertices.shape
    # F, _ = faces.shape

    # Precompute surface normals per face across the batch
    # B x |F| x 3
    #face_surface_normals = compute_surface_normals_per_face_batch_template(V, F, eps=1e-7)

    dist_uni = torch.distributions.Uniform(torch.tensor([0.]).to(
                    vertices.device), torch.tensor([1.]).to(vertices.device))

    # Obtain the coordinates per ith nodal entry per face
    v0 = torch.index_select(vertices, 1, faces[:, 0]) # B x |F| x 3
    v1 = torch.index_select(vertices, 1, faces[:, 1])
    v2 = torch.index_select(vertices, 1, faces[:, 2])
    # Cross product between nodes-per-face
    crosses = torch.cross( (v1 - v0), (v2 - v1), dim = 2 )
    # Normalized face normals (B x |F| x 3)
    n_hat = crosses / crosses.norm(dim=2, p=2, keepdim=True).clamp(min=eps)    

    # Calculate area of each face
    x1, x2, x3 = torch.split(v0 - v1, 
            1, # Chunk the target dimension into individual tensors (i.e. size=1)
            dim=2) # Chunk along xyz coords dimension
    y1, y2, y3 = torch.split(v1 - v2,
            1,     
            dim=2)
    a = (x2 * y3 - x3 * y2)**2 # Each of xi & yi is B x |F|
    b = (x3 * y1 - x1 * y3)**2
    c = (x1 * y2 - x2 * y1)**2
    abc_sum = (a + b + c).clamp(min=eps)
    Areas = ( torch.sqrt(abc_sum) / 2 ).squeeze(-1)
    # percentage of each face w.r.t. full surface area
    # After this, "areas" holds the proportion of each triangle in the mesh (i.e., in [0,1])
    # It seems the total area may be sufficiently large to cause some numerators to go to zero
    Areas = Areas / ( Areas.sum(dim=-1, keepdim=True).clamp(min=eps) ) # B x |F|
    # Add an additional smoothing correction to ensure positive multinomial probs
    Areas = (Areas.clamp(min=eps) + 1e-5)
    Areas = Areas / Areas.sum(dim=-1, keepdim=True)

    # NOTE TO SELF:
    # THE ERROR IS NOT HERE
    # IT WAS A NAN BEFORE-HAND, AND ARRIVED HERE
    # SEEMS TO COME FROM THE BACKWARD STEP, NOT A PARTICULAR LOSS

    #print(Areas[Areas < 0.001])
    # define discrete distribution w.r.t. face area ratios caluclated
    cat_dist = torch.distributions.Categorical(Areas.view(B,F))
    face_choices = cat_dist.sample( (num_samples,) ).T # B x N_S

    # from each chosen face sample a point
    # faces : |F| x 3
    select_faces = faces[face_choices] # B x N_S x 3

    # Gather nodal points of the chosen faces
    v1s = torch.gather(vertices, 
                      dim=1, 
                      index=select_faces[:,:,0].unsqueeze(-1).expand(-1,-1,3)) 
    v2s = torch.gather(vertices, 
                      dim=1, 
                      index=select_faces[:,:,1].unsqueeze(-1).expand(-1,-1,3)) 
    v3s = torch.gather(vertices, 
                      dim=1, 
                      index=select_faces[:,:,2].unsqueeze(-1).expand(-1,-1,3))     
    u = torch.sqrt(dist_uni.sample([B, num_samples]))
    v = dist_uni.sample([B, num_samples])
    points = (1 - u) * v1s + (u * (1 - v)) * v2s + u * v * v3s

    # Gather the face normals of the chosen faces
    normals = torch.gather(n_hat, # B x |F| x 3
                           # index only selects in the face dimension 
                           dim = 1,
                           # face_choices is B x N_S -> must be duplicated along the coords axis
                           index = face_choices.unsqueeze(-1).expand(-1,-1,3) )

    return points, normals


MAX_CAMERA_DISTANCE = 1.75

# for each model in the folder:
data_dir = "./datasets/ShapeNet_Selected/Chair/"

# output folders
model_dir = "./chair_models_train"
image_dir = "./chair_images_train"
model_test_dir = "./chair_models_test"
image_test_dir = "./chair_images_test"

# rm existing files
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

if os.path.exists(image_dir):
    shutil.rmtree(image_dir)

if os.path.exists(model_test_dir):
    shutil.rmtree(model_test_dir)

if os.path.exists(image_test_dir):
    shutil.rmtree(image_test_dir)

# create new folders
os.makedirs(model_dir)
os.makedirs(image_dir)
os.makedirs(model_test_dir)
os.makedirs(image_test_dir)

def compute_camera_params_np(azimuth: float, elevation: float, distance: float):

    theta = np.deg2rad(azimuth)
    phi = np.deg2rad(elevation)

    camY = distance * np.sin(phi)
    temp = distance * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    l2[l2 == 0] = 1
    cam_mat = cam_mat / np.expand_dims(l2, 1)

    return torch.FloatTensor(cam_mat), torch.FloatTensor(cam_pos)

test_folder_names = ["1a6f615e8b1b5ae4dbbc9440457e303e", "1a8bbf2994788e2743e99e0cae970928", "1a74a83fa6d24b3cacd67ce2c72c02e"]

# for each file
rendering_data = dict()
test_rendering_data = dict()
Rs, Ts, voxel_RTs = [], [], []
for subdir, dirs, files in os.walk(data_dir):
    for dir in dirs:
        if dir == "rendering":
            prefix = subdir.split("/")[-1]
            img_output_folder = image_dir
            if prefix in test_folder_names:
                img_output_folder = image_test_dir

            for i in range(24):
                idx = str(i)
                if i < 10:
                    idx = "0" + idx

                shutil.copy(f'{subdir}/{dir}/{idx}.png', f'{img_output_folder}/{prefix}_{idx}.png')

            # Load the metadata file:
            metadata_file = f'{subdir}/{dir}/rendering_metadata.txt'
            m = open(metadata_file, "r")
            metadata_lines = m.readlines()
        
            # Get camera calibration.
            for i in range(len(metadata_lines)):
                idx = str(i)
                if i < 10:
                    idx = "0" + idx

                azim, elev, yaw, dist_ratio, fov = [
                    float(v) for v in metadata_lines[i].strip().split(" ")
                ]
                dist = dist_ratio * MAX_CAMERA_DISTANCE
                # Extrinsic matrix before transformation to PyTorch3D world space.
                # RT = compute_extrinsic_matrix(azim, elev, dist)
                # R, T = _compute_camera_calibration(RT)
                # Rs.append(R)
                # Ts.append(T)
                # voxel_RTs.append(RT)

                R, T = compute_camera_params_np(azim, elev, dist_ratio)
                if prefix in test_folder_names:
                    test_rendering_data[f"{prefix}_{idx}"] = dict()
                    test_rendering_data[f"{prefix}_{idx}"]["rotation"] = R.tolist()
                    test_rendering_data[f"{prefix}_{idx}"]["translation"] = T.tolist()
                else:
                    rendering_data[f"{prefix}_{idx}"] = dict()
                    rendering_data[f"{prefix}_{idx}"]["rotation"] = R.tolist()
                    rendering_data[f"{prefix}_{idx}"]["translation"] = T.tolist()

        else:
            model_output_folder = model_dir
            if dir in test_folder_names:
                model_output_folder = model_test_dir

            # Load the .binvox file:
            vox_file = f'{data_dir}{dir}/model.binvox'
            with open(vox_file, 'rb') as f:
                voxel = read_as_3d_array(f)

            # Convert the voxel representation to a point cloud:    
            vertices, faces, normals, _ = marching_cubes(voxel.data, 0, allow_degenerate=False)
            point_cloud = torch.from_numpy(vertices.astype(float))

            # print(point_cloud.shape)
            # print(faces.shape)
            # faces_tensor = torch.from_numpy(faces.astype(int))
            # points, normals = sample_triangle_mesh_with_normals(point_cloud, faces_tensor, 1000)

            # Compute the normals for the point cloud:
            # normals = torch.from_numpy(normalize(faces[:, :3], norm='l2'))
            # normals = torch.from_numpy(normals.astype(float))

            # Save the point cloud and normals as PyTorch objects:
            output_file = dir
            torch.save(point_cloud, f'{model_output_folder}/{output_file}.PC.pt')
            torch.save(normals, f'{model_output_folder}/{output_file}.normals.pt')
    
# write metadata to file
with open(f'{image_dir}/rendering_metadata.json', 'w') as fp:
    json.dump(rendering_data, fp)

with open(f'{image_test_dir}/rendering_metadata.json', 'w') as fp:
    json.dump(test_rendering_data, fp)