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

def get_box_by_mask(mask):
    non_zero_indices = torch.nonzero(mask.float())
    min_indices = torch.min(non_zero_indices, dim=0).values
    max_indices = torch.max(non_zero_indices, dim=0).values
    top_left = min_indices
    bottom_right = max_indices
    return [top_left[1].item(), top_left[0].item(), bottom_right[1].item(), bottom_right[0].item()]

def draw_circle_box(img, center, radius, pt1, pt2, color, thickness, dash_length, gap_length):  
                
    cv2.circle(img, center, radius, color, thickness)  
                
    # 转换为整数坐标
    pt1 = tuple(map(int, pt1))  
    pt2 = tuple(map(int, pt2))   
    # 绘制顶部边  
    for x in range(pt1[0], pt2[0] + 1, dash_length + gap_length):  
        end_x = min(x + dash_length, pt2[0])  
        cv2.line(img, (x, pt1[1]), (end_x, pt1[1]), color, thickness)  
    # 绘制底部边  
    for x in range(pt1[0], pt2[0] + 1, dash_length + gap_length):  
        end_x = min(x + dash_length, pt2[0])  
        cv2.line(img, (x, pt2[1]), (end_x, pt2[1]), color, thickness)  
    # 绘制左侧边  
    for y in range(pt1[1], pt2[1] + 1, dash_length + gap_length):  
        end_y = min(y + dash_length, pt2[1])  
        cv2.line(img, (pt1[0], y), (pt1[0], end_y), color, thickness)  
    # 绘制右侧边  
    for y in range(pt1[1], pt2[1] + 1, dash_length + gap_length):  
        end_y = min(y + dash_length, pt2[1])  
        cv2.line(img, (pt2[0], y), (pt2[0], end_y), color, thickness)  

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
    acc_num = 0
    total_box = 0
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        biou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w), dtype=bool)
        gt_mask_lvl = np.zeros((n_head, h, w), dtype=bool)
        relevancy_lvl = torch.zeros((n_head, h, w))
        composited_processed_map_lvl = np.zeros((n_head, h, w, 3), np.uint8)
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
            # p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            # valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            # mask = (valid_map[i][k] < 0.5).squeeze()
            # valid_composited[mask, :] = image[mask, :] * 0.3
            # output_path_compo = save_path / 'composited' / f'{clip_model.positives[k]}_{i}'
            # output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            # colormap_saving(valid_composited, colormap_options, output_path_compo)
            # print(valid_composited.shape)
            # composited_processed_map = (valid_composited.cpu().numpy() * 255).astype(np.uint8)    
            # composited_processed_map_lvl[i] = composited_processed_map       
            

            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            relevancy_lvl[i] = output.cpu()

            mask_pred_bool = (output.cpu().numpy() > thresh)
            mask_lvl[i] = mask_pred_bool
            mask_pred = (smooth(mask_pred_bool) * 255).astype(np.uint8)

            # output_path_mask = save_path / 'mask' / f'{text_prompt}.png'
            # output_path_mask.parent.mkdir(exist_ok=True, parents=True)
            # Image.fromarray(mask_pred).save(str(output_path_mask))

            # p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(valid_map[i][k].unsqueeze(-1), colormaps.ColormapOptions("turbo",normalize=True, colormap_min=-1))
            # mask = (valid_map[i][k] < 0.5).squeeze()
            # print(valid_composited.shape)
            # print(mask_lvl[i].shape)
            # print(image.shape)
            valid_composited[~mask_pred_bool, :] = image[~mask_pred_bool, :] * 0.3
            output_path_compo = save_path / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)

            composited_processed_map = (valid_composited.cpu().numpy() * 255).astype(np.uint8)    
            composited_processed_map_lvl[i] = composited_processed_map   

            gt_mask = gt_masks[k]
            # print(gt_mask.shape)

            # mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
            
            # calculate iou
            # intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            # union = np.sum(np.logical_or(mask_gt, mask_pred))
            # iou = np.sum(intersection) / np.sum(union)
            # iou_lvl[i] = iou
            if mask_pred.shape != gt_mask.shape:
                gt_mask = resize_mask(gt_mask, mask_pred.shape)
            gt_mask_lvl[i] = gt_mask

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
        relevancy = relevancy_lvl[chosen_lvl]
        composited_processed_map = composited_processed_map_lvl[chosen_lvl]
        max_relevancy_coord = torch.nonzero(relevancy == relevancy.max())[0]
        gt_box = get_box_by_mask(torch.from_numpy(gt_mask_lvl[chosen_lvl]))
        x1, y1, x2, y2 = gt_box
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        if (max_relevancy_coord[1] >= x_min and max_relevancy_coord[1] <= x_max and 
            max_relevancy_coord[0] >= y_min and max_relevancy_coord[0] <= y_max):
            acc_num += 1

        total_box +=1

        # for i in range(n_head):
            # image = composited_processed_map
            # 定义圆心和半径
        x0 = int(max_relevancy_coord[1])
        y0 = int(max_relevancy_coord[0])
        center = (x0, y0)
        radius = 9 
        # 定义颜色（BGR格式）和线条粗细  
        color = (255, 255, 255)
        thickness = 3 
        dash_length = 10  # 虚线段长度
        gap_length = 5  # 虚线段间隔   

        # gt_box是[x1, y1, x2, y2]格式，表示左上角(x1, y1)和右下角(x2, y2)。  
        x1, y1, x2, y2 = gt_box
        # 使用自定义函数绘制虚线矩形框
        draw_circle_box(composited_processed_map, center, radius, (x1, y1), (x2, y2), color, thickness, dash_length, gap_length)  

        # 保存图像到指定路径
        output_path_composited_processed_map = save_path / 'composited_processed' / f'{clip_model.positives[k]}.png'
        output_path_composited_processed_map.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path_composited_processed_map), composited_processed_map[:,:,::-1])



        iou_scores[text_prompt] = iou_lvl[chosen_lvl]
        biou_scores[text_prompt] = biou_lvl[chosen_lvl]

        mask_pred_bool = mask_pred.cpu().numpy()
        # mask_pred = 

        output_path_mask = save_path / 'mask' / f'{text_prompt}.png'
        output_path_mask.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray((smooth(mask_pred_bool) * 255).astype(np.uint8)).save(str(output_path_mask))

        output_path_mask_map = save_path / 'mask_map' / f'{text_prompt}.jpg'
        output_path_mask_map.parent.mkdir(exist_ok=True, parents=True)
        mask_map = image.clone().permute(2, 0, 1)
        # print(mask_map.shape)
        # print(mask_pred.shape)
        mask_map[:, mask_pred] = mask_map[:, mask_pred] * 0.5 + torch.tensor([1, 0, 0], device=image.device).reshape(3, 1) * 0.5
        mask_map[:, ~mask_pred] /= 2
        # mask_3d = (mask_3d.cpu().numpy() * 255).astype(np.uint8)
        torchvision.utils.save_image(mask_map, str(output_path_mask_map))


        output_path_rendering = save_path / 'renders' / f'rendering.jpg'
        output_path_rendering.parent.mkdir(exist_ok=True, parents=True)
        # print(image.shape)
        torchvision.utils.save_image(image.permute(2, 0, 1), str(output_path_rendering))
        # chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        # # save for visulsization
        # save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        # vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return iou_scores, biou_scores, acc_num, total_box

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
    acc_nums = 0
    total_boxes = 0
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    if args.dataset_name == 'figurines':
        prompt_dict = {"green apple":"what is green fruit","green toy chair":"what is suitable for people to sit down and is green","old camera":"what can be used to take pictures and is black","porcelain hand":"what is like a part of a person","red apple":"what is red fruit","red toy chair":"what is suitable for people to sit down and is red","rubber duck with red hat":"which is the small yellow rubber duck"}
    elif args.dataset_name == 'ramen':
        prompt_dict = {"chopsticks":"which one is the chopstic on the side of yellow bowl","egg":"what is the round, golden, protein-rich object in the bowl","glass of water":"which one is a transparent cup with water in it", "pork belly":"which is the big piece of meat in the bowl", "wavy noodles in bowl":"which are long and thin noodles","yellow bowl":"which is the yellow bowl used to hold noodles"}
    elif args.dataset_name == 'teatime':
        prompt_dict = {"apple":"which is red fruit","bag of cookies":"which is the brown bag on the side of the plate","coffee mug":"which cup is used for coffee","cookies on a plate":"which are the cookies","paper napkin":"what can be used to wipe hands","plate":"what can be used to hold cookies","sheep":"which is a cute white doll","spoon handle":"which is spoon handle","stuffed bear":"which is the brown bear doll","tea in a glass":"which is the drink in the transparent glass"}
    elif args.dataset_name == 'bed':
        prompt_dict ={'banana':'which is a fruit with a yellow peel',
  'black leather shoe': 'which is an object that can be worn on the feet',
  'camera': 'which is a device used for taking pictures',
  'hand': 'which is a part of the human body',
  'red bag': 'which is red,leathern object used to put items in',
  'white sheet': 'which is a piece of fabric used for covering a bed'}
        # prompt_dict ={"banana":"which is a yellow fruit often eaten as a snack","black leather shoe":"which is a black shoe with a gold buckle","camera":"which is a device used for taking pictures","hand":"which is a part of the human body used for holding objects","red bag":"which is a red bag with a quilted pattern","white sheet":"which is a white sheet with black lines"}
        # prompt_dict = {"banana":"which is the yellow fruit","black leather shoe":"which can be worn on the foot","camera":"which can be used to take photos","hand":"which is the part of person, excluding other objects","red bag":"which is red and leather","white sheet":"where is a good place to lie down"}       
    elif args.dataset_name == 'bench':
        prompt_dict ={"dressing doll": "which is an object used for dressing up",
  "green grape": "which is a fruit that is green",
  "mini offroad car": "which is a small vehicle used for off-road driving",
  "orange cat": "which is an animal that is orange",
  "pebbled concrete wall": "which is a wall made of pebbled concrete",
  "Portuguese egg tart": "which is a dessert that is a Portuguese egg tart",
  "wood": "which is the object made of wood"}
        # prompt_dict = {"dressing doll":"which is a toy used for dressing up","green grape":"which is a green fruit that grows in clusters","mini offroad car":"which is a small toy car designed for off-road use","orange cat":"which is a feline with orange fur","pebbled concrete wall":"which is a wall made of concrete with embedded pebbles","Portuguese egg tart":"which is a pastry with a custard filling","wood":"which is a material used for building and furniture"}
        # prompt_dict = {"dressing doll":"which is a cute humanoid doll that girls like","green grape":"which is green fruit","mini offroad car":"which one is the model of the vehicle","orange cat":"which is an animal","pebbled concrete wall":"which is made of many stones", "Portuguese egg tart":"which is like baked food","wood":"which is made of wood"}
    elif args.dataset_name == 'lawn':
        prompt_dict = {"red apple":"which is a red fruit rich in vitamins","New York Yankees cap":"which is a cap with a sports team logo","stapler":"which is a device used for fastening paper","black headphone":"which is a black device used for listening to audio","hand soap":"which is a liquid used for cleaning hands","green lawn":"which is a green grassy area"}
        # prompt_dict = {"red apple":"which is the red fruit","New York Yankees cap":"which is worn on the head and is white","stapler":"which is small device used for stapling paper","black headphone":"which can convert electric signals into sounds","hand soap":"which is bottled", "green lawn":"which is an area of ground covered in short grass"}
    elif args.dataset_name == 'room':
        prompt_dict = {"wood":"which is a type of material used for furniture and construction","shrilling chicken":"which is a toy that makes a loud noise when squeezed","weaving basket":"which is a container made from woven materials","rabbit":"which is a small, furry animal with long ears","dinosaur":"which is a prehistoric creature that lived millions of years ago","baseball":"which is a round, white ball used in a sport"}
        # prompt_dict = {"wood":"which is background wood board","shrilling chicken":"which is a yellow animal doll","weaving basket":"which can be uesd to hold a water bottle","rabbit":"which is a cute mammal doll","dinosaur":"which has a long tail", "baseball":"which is spherical and white"}
    elif args.dataset_name == 'sofa':
        prompt_dict = {'Pikachu': 'which is a yellow electric-type creature',
  'a stack of UNO cards': 'which is a deck of playing cards',
  'grey sofa': 'which is a piece of furniture',
  'a red Nintendo Switch joy-con controller': 'which is a handheld gaming device',
  'Gundam': 'which is a model of a robot',
  'Xbox wireless controller': 'which is a device used to play video games'}
        prompt_dict = {"Pikachu":"which is a yellow plush toy with a hat","a stack of UNO cards":"which is a deck of cards with a colorful design","grey sofa":"which is a piece of furniture with a soft, grey surface","a red Nintendo Switch joy-con controller":"which is a red handheld gaming device","Gundam":"which is a blue and white action figure","Xbox wireless controller":"which is a white gaming controller with buttons and joysticks"}
        # prompt_dict = {"Pikachu":"which is the yellow doll","a stack of UNO cards":"what is made of cards stacked together", "grey sofa":"where can I sit down","a red Nintendo Switch joy-con controller":"which is red and looks like a controller","Gundam":"which is the body of a robot model","Xbox wireless controller":"which can be used to play games and is large and white"}
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
        # image_index = image_name.split('_')[-1]
        # print(image_name)
        # print(image_index)
        if args.reasoning:
            reasoning = '_reasoning'
        else:
            reasoning = ''
        save_path = os.path.join(model_path, name, f"ours_{iteration}{reasoning}", f'{image_name}')
        
        if image_name not in test_views:
            continue
        # print(image_index)
        # print(test_views)
        view_mask_path = os.path.join(mask_path, image_name)
        masks_name = os.listdir(view_mask_path)
        text_prompts = [mask_name.split('.')[0] for mask_name in masks_name]
        if args.reasoning:
            text_prompts = [prompt_dict[text_prompt] for text_prompt in text_prompts]
        gt_masks = [load_mask(os.path.join(view_mask_path, mask_name)) for mask_name in masks_name]

        sem_feat = torch.stack([language_feature_image_1, language_feature_image_2, language_feature_image_3]).permute(0, 2, 3, 1)
        lvl, h, w, _ = sem_feat.shape
        restored_feat = ae_model.decode(sem_feat.flatten(0, 2))
        restored_feat = restored_feat.view(lvl, h, w, -1) 
        clip_model.set_positives(text_prompts)
        # valid_map = clip_model.get_max_across(sem_map)

        iou_score, biou_score, acc_num, total_box = activate_stream(restored_feat,
                                                gt_masks,
                                                rendering.permute(1, 2, 0),
                                                clip_model,
                                                Path(save_path),
                                                None,
                                                0.4,
                                                colormap_options,
                                                )
        acc_nums += acc_num
        total_boxes += total_box
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
    acc = round(acc_nums / total_boxes, 5)
    print(f'mean iou: {np.mean(mean_ious)} biou: {np.mean(mean_bious)} acc: {acc}')
    
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