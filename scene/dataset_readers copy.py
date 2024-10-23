#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
import open_clip
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON
from torchvision import transforms 
# from hdbscan import HDBSCAN

# def generate_sam_masks(images, sam_masks_save_path, sam_type='vit_h', sam_ckpt_path='ckpts/sam_vit_h_4b8939.pth'):
#     registry = sam_model_registry['vit_h']
#     model = registry('ckpts/sam_vit_h_4b8939.pth')
#     model = model.to(device='cuda')
#     model = SamAutomaticMaskGenerator(model)
#     save_sam_mask_dict = {'sam_mask': {}, 'pixel_mask_array': {}}
    

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    sam_clip: dict
    # binary_sam_masks: torch.Tensor
    # mask_clip_embeddings: torch.Tensor
    depth: Optional[np.array] = None
    
    


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


@torch.no_grad()
def readNerfiesCameras(path, load_mask=False):
    
    def get_box_by_mask(mask):
        non_zero_indices = torch.nonzero(mask.float())
        min_indices = torch.min(non_zero_indices, dim=0).values
        max_indices = torch.max(non_zero_indices, dim=0).values
        top_left = min_indices
        bottom_right = max_indices
        return [top_left[1].item(), top_left[0].item(), bottom_right[1].item() + 1, bottom_right[0].item() + 1]
    
    if load_mask:
        sam_clip_path = f'{path}/sam_clip'
        # registry = sam_model_registry['vit_h']
        # model = registry('ckpts/sam_vit_h_4b8939.pth')
        # model = model.to(device='cuda')
        # model = SamAutomaticMaskGenerator(model)
        # sam_mask_save_path = f'{path}/sam2_mask.pt'
        # upsampler = torch.hub.load("/root/.cache/torch/hub/mhamilton723_FeatUp_main", 'maskclip', source='local', use_norm=False).cuda().eval()
        # clip_model, _, _ = open_clip.create_model_and_transforms(
        #     'ViT-B-16',  # e.g., ViT-B-16
        #     pretrained="laion2b_s34b_b88k",  # e.g., laion2b_s34b_b88k
        #     precision="fp16",
        # )
        # clip_model = clip_model.cuda().eval()
        # # resize_size = 224
        # transform = transforms.Compose(
        #     [
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.48145466, 0.4578275, 0.40821073],
        #             std=[0.26862954, 0.26130258, 0.27577711],
        #         ),
                
        #     ]
        # )
        # # print(sam_mask_save_path)
        # # total_clip_embeddings = []
        # if os.path.exists(sam_mask_save_path):
        #     save_sam_mask_dict = torch.load(sam_mask_save_path)
        # else:
        #     # save_sam_mask_dict = {'binary_sam_masks': {}, 'mask_clip_embeddings': {}}
        #     save_sam_mask_dict = {'binary_sam_masks': {}, 'mask_clip_embeddings': {}}
    
    
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        segmentations_path = os.path.join(path, 'segmentations')
        if os.path.exists(segmentations_path):
            eval_img = os.listdir(segmentations_path)
            all_img = train_img + eval_img
        else:
            all_img = train_img
            eval_img = []
        # val_img = dataset_json['val_ids']
        all_img = train_img + eval_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        segmentations_path = os.path.join(path, 'segmentations')
        if os.path.exists(segmentations_path):
            eval_img = os.listdir(segmentations_path)
            eval_img = [img for img in eval_img if img in meta_json.keys()]
            # print(eval_img)
            all_img = train_img + eval_img
        else:
            all_img = train_img
            eval_img = []
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)
    all_object_mask = [f'{path}/object_mask/{i}.npy' for i in all_img]
    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in tqdm(range(len(all_img))):
        image_path = all_img[idx]
        image_np = np.array(Image.open(image_path))
        image = Image.fromarray((image_np).astype(np.uint8))
        image_name = Path(image_path).stem
        
        if load_mask and image_name not in eval_img:
            # object_masks = torch.from_numpy((np.load(all_object_mask[idx]))).long()
            # if os.path.exists(sam_mask_save_path):
            mask_path = f'{sam_clip_path}/{image_name}_m.pt'
            # print(mask_path)
            # scale_path = f'{sam_clip_path}/{image_name}_s.pt'
            # feature_path = f'{sam_clip_path}/{image_name}_f.pt'
            clip_feature_path = f'{sam_clip_path}/{image_name}_f.pt'
            if os.path.exists(mask_path):
                masks = torch.load(mask_path)
                # scales = torch.load(scale_path)
                # features = torch.load(feature_path)
                # low_dim_features = torch.from_numpy(np.load(low_dim_feature_path))
                clip_features = torch.load(clip_feature_path)
            sam_clip = masks
            sam_clip.update({'clip_features': clip_features})
                # binary_sam_masks = save_sam_mask_dict['binary_sam_masks'][image_name]
                # mask_clip_embeddings = save_sam_mask_dict['mask_clip_embeddings'][image_name]
            # else:
                # binary_sam_masks_list = []
                # mask_clip_embeddings_list = []
                # for i in [0, 1]:
                #     binary_sam_masks = []
                #     mask_clip_embeddings = []
                #     object_mask = object_masks[i]
                #     mask_indexes = torch.unique(object_mask)
                #     for mask_index in mask_indexes:
                #         if mask_index == 0:
                #             continue
                #         binary_mask = object_mask == mask_index
                #         box = get_box_by_mask(binary_mask)
                #         # print(box)
                #         box_image_np = image_np.copy()
                #         box_image_np[~binary_mask.cpu().numpy(), :] = np.array([0, 0, 0])
                #         box_image_np = box_image_np[box[1]:box[3], box[0]:box[2], :]
                #         box_image_pil = Image.fromarray(box_image_np)
                #         mask_clip_embedding = clip_model.encode_image(transform(box_image_pil)[None].cuda().half(), normalize=True)
                #         # coarse_binary_mask_lr = torch.nn.functional.interpolate(coarse_binary_mask[None, None, ...].float(), size=(resize_size, resize_size), mode='bilinear')[0, 0, ...]
                #         # coarse_binary_mask_lr = coarse_binary_mask_lr > 0.5
                #         # mask_clip_embedding = hr_clip_feat[:, coarse_binary_mask_lr].mean(-1)[None]
                #         # mask_clip_embedding = (mask_clip_embedding / mask_clip_embedding.norm(dim=-1, keepdim=True) + 1e-9)
                #         # clip_embed /= (clip_embed.norm(dim=-1, keepdim=True) + 1e-9)
                #         # mask_clip_embedding = torch.nn.functional.normalize(mask_clip_embedding, dim=-1)[0]
                #         binary_sam_masks.append(binary_mask.cpu())
                #         mask_clip_embeddings.append(mask_clip_embedding.cpu())
                #     binary_sam_masks_list.append(torch.stack(binary_sam_masks))
                #     mask_clip_embeddings_list.append(torch.cat(mask_clip_embeddings))
                # sam_mask = sam_mask.cpu()
                # pixel_mask_array = torch.stack([fine_mask_array, coarse_mask_array], -1).cpu()
                # save_sam_mask_dict['sam_mask'][image_name] = sam_mask
                # binary_sam_masks = torch.stack(binary_sam_masks)
                # mask_clip_embeddings = torch.cat(mask_clip_embeddings)
                # save_sam_mask_dict['binary_sam_masks'][image_name] = binary_sam_masks_list
                # save_sam_mask_dict['mask_clip_embeddings'][image_name] = mask_clip_embeddings_list
                # total_clip_embeddings.append(mask_clip_embeddings)
                        
        else:
            # binary_sam_masks = None
            # mask_clip_embeddings = None
            # object_mask = None
            sam_clip = None

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid, sam_clip=sam_clip)
        cam_infos.append(cam_info)
    # if load_mask and not os.path.exists(sam_mask_save_path):
        # total_clip_embeddings = torch.cat(total_clip_embeddings)
        # total_clip_embeddings_np = total_clip_embeddings.cpu().numpy()
        # clusterer = HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
        # clusterer.fit(total_clip_embeddings_np)
        # labels = clusterer.labels_
        # unique_labels = list(set(labels))
        # print(unique_labels)
        # label2clip_embedding = {}
        # for unique_label in unique_labels:
        #     label_mask = unique_label == labels
        #     label_clip_embedding = torch.nn.functional.normalize(total_clip_embeddings[label_mask, :].mean(0)[None, :].float(), dim=-1)[0].half()
        #     label2clip_embedding[unique_label] = label_clip_embedding.numpy()
        # semantic_embeddings = {}
        # semantic_masks = {}
        # start = 0
        # for i, key in enumerate(save_sam_mask_dict['binary_sam_masks'].keys()):
        #     # key = binary_sam_mask.keys()[0]
        #     mask = save_sam_mask_dict['binary_sam_masks'][key]
        #     mask_num = mask.shape[0]
        #     tmp_labels = labels[start:start+mask_num]
        #     tmp_semantic_masks = []
        #     tmp_semantic_embeddings = []
        #     tmp_unique_labels = list(set(tmp_labels))
        #     for tmp_unique_label in tmp_unique_labels:
        #         tmp_unique_label_mask = tmp_unique_label == tmp_labels
        #         semantic_mask = mask[tmp_unique_label_mask, ...].sum(0) > 0
        #         tmp_semantic_masks.append(semantic_mask)
        #         tmp_semantic_embeddings.append(label2clip_embedding[tmp_unique_label])
        #     semantic_masks[key] = torch.from_numpy(np.stack(tmp_semantic_masks))
        #     semantic_embeddings[key] = torch.from_numpy(np.stack(tmp_semantic_embeddings))
        #     cam_infos[i].binary_sam_masks = semantic_masks
        #     cam_infos[i].mask_clip_embeddings = semantic_embeddings
            
        #     start += mask_num
        # save_sam_mask_dict['binary_sam_masks'] = semantic_masks
        # save_sam_mask_dict['mask_clip_embeddings'] = semantic_embeddings
        # torch.save(save_sam_mask_dict, sam_mask_save_path)
    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readNerfiesInfo(path, eval, load_mask=False):
    
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path, load_mask)

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    # video_paths = sorted(glob(os.path.join(path, 'frames/*')))
    video_names = os.listdir(path)
    video_paths = [os.path.join(path, video_name) for video_name in video_names if 'mp4' in video_name]
    video_paths.sort()
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = video_paths[i]
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def readPlenopticVideoDataset(path, eval, num_images, hold_id=[0]):
    print("Reading Training Camera")
    train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id,
                                         num_images=num_images)

    print("Reading Training Camera")
    test_cam_infos = readCamerasFromNpy(
        path, 'poses_bounds.npy', split="test", hold_id=hold_id, num_images=num_images)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3D.ply')
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                fid = time, binary_sam_masks=None, mask_clip_embeddings=None))

    return cameras

def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            fid = time, binary_sam_masks=None, mask_clip_embeddings=None))
    return cameras

def readdynerfInfo(datadir):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    # ply_path = None
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
        datadir,
        "train",
        1.0,
        time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0,
            )    
    test_dataset = Neural3D_NDC_Dataset(
        datadir,
        "test",
        1.0,
        time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0,
        )
    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # xyz = np.load
    pcd = fetchPly(ply_path)
    print("origin points,",pcd.points.shape[0])
    
    print("after points,",pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                        #    video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                        #    maxtime=300
                           )
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "neu3d": readdynerfInfo,
}
