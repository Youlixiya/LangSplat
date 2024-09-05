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
import cv2
import numpy as np
from PIL import Image
import torch
from scene import Scene
from pathlib import Path
import os
from tqdm import tqdm
from copy import deepcopy
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from eval import colormaps
from autoencoder.model import Autoencoder
from eval.openclip_encoder import OpenCLIPNetwork
from eval.utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    dt = (dt>128).astype('uint8')
    gt = (gt>128).astype('uint8')
    

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou

def calculate_iou(mask1, mask2, input_bool=False):
    """Calculate IoU between two boolean masks."""
    if not input_bool:
        mask1_bool = mask1 > 128
        mask2_bool = mask2 > 128
    else:
        mask1_bool = mask1
        mask2_bool = mask2
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def load_mask(mask_path):
    """Load the mask from the given path."""
    return np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale

def resize_mask(mask, target_shape):
    """Resize the mask to the target shape."""
    return np.array(Image.fromarray(mask).resize((target_shape[1], target_shape[0]), resample=Image.NEAREST))

def activate_stream(sem_map, 
                    gt_masks,
                    image, 
                    clip_model, 
                    save_path: Path = None,
                    img_ann = None, 
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    # chosen_iou_list, chosen_lvl_list = [], []
    iou_scores = {}
    biou_scores = {}
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        biou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w), dtype=bool)
        text_prompt = clip_model.positives[k]
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            output_path_relev = save_path / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)
            
            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = save_path / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh)
            mask_lvl[i] = mask_pred
            mask_pred = (smooth(mask_pred) * 255).astype(np.uint8)
            gt_mask = gt_masks[k]
            
            # mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
            
            # calculate iou
            # intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            # union = np.sum(np.logical_or(mask_gt, mask_pred))
            # iou = np.sum(intersection) / np.sum(union)
            # iou_lvl[i] = iou
            if mask_pred.shape != gt_mask.shape:
                gt_mask = resize_mask(gt_mask, mask_pred.shape)
            iou = calculate_iou(gt_mask, mask_pred)
            biou = boundary_iou(gt_mask, mask_pred)
            iou_lvl[i] = iou
            biou_lvl[i] = biou
            # text_prompt = clip_model.positives[k]
            # if text_prompt not in iou_scores:
            #     iou_scores[text_prompt] = []
            #     biou_scores[text_prompt] = []
            # iou_scores[text_prompt].append(iou)
            # biou_scores[text_prompt].append(biou)

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)
        mask_pred = torch.from_numpy(mask_lvl[chosen_lvl])
        iou_scores[text_prompt] = iou_lvl[chosen_lvl]
        biou_scores[text_prompt] = biou_lvl[chosen_lvl]

        output_path_mask_map = save_path / 'mask_map' / f'{text_prompt}.jpg'
        output_path_mask_map.parent.mkdir(exist_ok=True, parents=True)
        mask_map = image.clone().permute(2, 0, 1)
        # print(mask_map.shape)
        # print(mask_pred.shape)
        mask_map[:, mask_pred] = mask_map[:, mask_pred] * 0.5 + torch.tensor([1, 0, 0], device=image.device).reshape(3, 1) * 0.5
        mask_map[:, ~mask_pred] /= 2
        # mask_3d = (mask_3d.cpu().numpy() * 255).astype(np.uint8)
        torchvision.utils.save_image(mask_map, str(output_path_mask_map))
        # chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        # # save for visulsization
        # save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        # vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return iou_scores, biou_scores

def render_set(model_path, source_path, name, iteration, views, gaussians_1, gaussians_2, gaussians_3, pipeline, background, args, clip_model, ae_model):
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # heatmap_path = os.path.join(model_path, name, "ours_{}".format(iteration), "heatmaps")
    # feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "features")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    # gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")
    mask_path = os.path.join(source_path, 'segmentations')

    
    test_views = os.listdir(mask_path)
    test_views = [test_view for test_view in test_views if test_view != 'classes.txt']
    # print(source_path)
    # makedirs(render_npy_path, exist_ok=True)
    # makedirs(heatmap_path, exist_ok=True)
    # makedirs(gts_npy_path, exist_ok=True)
    # makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    iou_scores = {}
    biou_scores = {}
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    # if args.dataset_name == 'figurines':
    #     prompt_dict = {"green apple":"what is green fruit","green toy chair":"what is suitable for people to sit down and is green","old camera":"what can be used to take pictures and is black","porcelain hand":"what is like a part of a person","red apple":"what is red fruit","red toy chair":"what is suitable for people to sit down and is red","rubber duck with red hat":"which is the small yellow rubber duck"}
    # elif args.dataset_name == 'ramen':
    #     prompt_dict = {"chopsticks":"which one is the chopstic on the side of yellow bowl","egg":"what is the round, golden, protein-rich object in the bowl","glass of water":"which one is a transparent cup with water in it", "pork belly":"which is the big piece of meat in the bowl", "wavy noodles in bowl":"which are long and thin noodles","yellow bowl":"which is the yellow bowl used to hold noodles"}
    # elif args.dataset_name == 'teatime':
    #     prompt_dict = {"apple":"which is red fruit","bag of cookies":"which is the brown bag on the side of the plate","coffee mug":"which cup is used for coffee","cookies on a plate":"which are the cookies","paper napkin":"what can be used to wipe hands","plate":"what can be used to hold cookies","sheep":"which is a cute white doll","spoon handle":"which is spoon handle","stuffed bear":"which is the brown bear doll","tea in a glass":"which is the drink in the transparent glass"}
    # elif args.dataset_name == 'bed':
    #     prompt_dict = {"banana":"which is the yellow fruit","black leather shoe":"which can be worn on the foot","camera":"which can be used to take photos","hand":"which is the part of person, excluding other objects","red bag":"which is red and leather","white sheet":"where is a good place to lie down"}       
    # elif args.dataset_name == 'bench':
    #     prompt_dict = {"dressing doll":"which is a cute humanoid doll that girls like","green grape":"which is green fruit","mini offroad car":"which one is the model of the vehicle","orange cat":"which is an animal","pebbled concrete wall":"which is made of many stones", "Portuguese egg tart":"which is like baked food","wood":"which is made of wood"}
    # elif args.dataset_name == 'lawn':
    #     prompt_dict = {"red apple":"which is the red fruit","New York Yankees cap":"which is worn on the head and is white","stapler":"which is small device used for stapling paper","black headphone":"which can convert electric signals into sounds","hand soap":"which is bottled", "green lawn":"which is an area of ground covered in short grass"}
# 
#   
    # print(views)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output_1 = render(view, gaussians_1, pipeline, background, args)
        output_2 = render(view, gaussians_2, pipeline, background, args)
        output_3 = render(view, gaussians_3, pipeline, background, args)
        rendering = output_1['render']
        language_feature_image_1 = output_1['language_feature_image']
        language_feature_image_2 = output_2['language_feature_image']
        language_feature_image_3 = output_3['language_feature_image']
        
        image_name = view.image_name
        image_index = image_name.split('_')[-1]
        if args.reasoning:
            reasoning = '_reasoning'
        else:
            reasoning = ''
        save_path = os.path.join(model_path, name, f"ours_{iteration}{reasoning}", '{0:05d}'.format(idx))
        print(image_index)
        print(test_views)
        
        if image_index not in test_views:
            continue
        view_mask_path = os.path.join(mask_path, image_index)
        masks_name = os.listdir(view_mask_path)
        text_prompts = [mask_name.split('.')[0] for mask_name in masks_name]
        # if args.reasoning:
        #     text_prompts = [prompt_dict[text_prompt] for text_prompt in text_prompts]
        gt_masks = [load_mask(os.path.join(view_mask_path, mask_name)) for mask_name in masks_name]

        sem_feat = torch.stack([language_feature_image_1, language_feature_image_2, language_feature_image_3]).permute(0, 2, 3, 1)
        lvl, h, w, _ = sem_feat.shape
        restored_feat = ae_model.decode(sem_feat.flatten(0, 2))
        restored_feat = restored_feat.view(lvl, h, w, -1) 
        clip_model.set_positives(text_prompts)
        # valid_map = clip_model.get_max_across(sem_map)

        iou_score, biou_score = activate_stream(restored_feat,
                                                gt_masks,
                                                rendering.permute(1, 2, 0),
                                                clip_model,
                                                Path(save_path),
                                                None,
                                                0.4,
                                                colormap_options,
                                                )
        # print(iou_score)
        # print(biou_score)
        for key in iou_score.keys():
            if key not in iou_scores:
                iou_scores[key] = [iou_score[key]]
            else:
                iou_scores[key].append(iou_score[key])
            if key not in biou_scores:
                biou_scores[key] = [biou_score[key]]
            else:
                biou_scores[key].append(biou_score[key])
    mean_ious = []
    mean_bious = []
    for key in iou_scores.keys():
        mean_iou = np.mean(iou_scores[key])
        mean_biou = np.mean(biou_scores[key])
        mean_ious.append(mean_iou)
        mean_bious.append(mean_biou)
        print(f'{key} iou: {mean_iou} biou: {mean_biou}')
    print(f'mean iou: {np.mean(mean_ious)} biou: {np.mean(mean_bious)}')
    
        # if not args.include_feature:
            
        #     rendering = output["render"]
        # else:
        #     output = render(view, gaussians, pipeline, background, args)
        #     rendering = output["language_feature_image"]
            
        # if not args.include_feature:
        #     gt = view.original_image[0:3, :, :]
            
        # else:
        #     gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)

        # np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        # np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        args.model_path = args.model_path.split('_')[0]
        gaussians_1 = GaussianModel(dataset.sh_degree)
        gaussians_2 = deepcopy(gaussians_1)
        gaussians_3 = deepcopy(gaussians_1)
        dataset.eval = True
        scene = Scene(dataset, gaussians_1, shuffle=False)
        checkpoint_1 = os.path.join(f'{args.model_path}_1', 'chkpnt30000.pth')
        checkpoint_2 = os.path.join(f'{args.model_path}_2', 'chkpnt30000.pth')
        checkpoint_3 = os.path.join(f'{args.model_path}_3', 'chkpnt30000.pth')
        (model_params_1, first_iter_1) = torch.load(checkpoint_1)
        gaussians_1.restore(model_params_1, args, mode='test')

        (model_params_2, first_iter_2) = torch.load(checkpoint_2)
        gaussians_2.restore(model_params_2, args, mode='test')

        (model_params_3, first_iter_3) = torch.load(checkpoint_3)
        gaussians_3.restore(model_params_3, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        ae_ckpt_path = os.path.join(args.ae_ckpt_dir, args.dataset_name, "best_ckpt.pth")
        clip_model = OpenCLIPNetwork('cuda')
        checkpoint = torch.load(ae_ckpt_path, map_location='cuda')
        ae_model = Autoencoder(args.encoder_dims, args.decoder_dims).to('cuda')
        ae_model.load_state_dict(checkpoint)
        ae_model.eval()

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians_1, gaussians_2, gaussians_3, pipeline, background, args, clip_model, ae_model)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians_1, gaussians_2, gaussians_3, pipeline, background, args, clip_model, ae_model)

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--ae_ckpt_dir", type=str, default='autoencoder/ckpt')
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)