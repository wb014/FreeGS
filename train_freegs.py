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
import json
import os
import random
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import math
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render_msplat as render
from scene import Scene, GaussianModel
from utils.crf import DenseCRF
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ins_l1_loss
from utils.layers import CNN_decoder

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from cuml.cluster import HDBSCAN
from cuml.neighbors import NearestNeighbors
import cupy as cp
import open3d as o3d
from cuml.decomposition import PCA
import matplotlib.pyplot as plt
import shutil
from featup.util import norm
import torchvision.transforms as T
from utils.downsamplers import SimpleDownsampler

from info_nce import InfoNCE

from utils.post_group_combine import group_combine
from utils.scaler import StandardScaler, MinMaxScaler, RobustScaler, minmax_scaler

import gc

torch.backends.cudnn.benchmark = False
mempool = cp.get_default_memory_pool()

MAX_GROUP = 64
PIXEL_FEAT = {}
INS_FEAT = {}
ID_MAP = {}
MASK_ORI = {}
MASK_CRF = {}
FEATURE_DIM = 128

resize = T.Resize((224, 224))

def generate_unique_colors(num_colors):
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    random.shuffle(colors)
    return np.array(colors)[:, :3]

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def cluster_gauss_sh(gaussians, groups=None, fea_enable=True, iteration=None, params=None, decoder=None):
    points = gaussians.get_xyz
    opacity = gaussians.get_opacity
    colors = gaussians.get_features[:, 0]
    if fea_enable:
        language_feature = gaussians.get_language_feature
    else:
        language_feature = None

    if params is not None and 'SCALER' in params:
        assert params['SCALER'] in ['standard', 'minmax', 'robust']
        if params['SCALER'] == 'standard':
            scaler = StandardScaler()
        elif params['SCALER'] == 'minmax':
            scaler = MinMaxScaler()
        elif params['SCALER'] == 'robust':
            scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()
    if not params:
        in_mask = (opacity > 0).squeeze(-1)
    else:
        in_mask = params['salient_idx'] if 'salient_idx' in params else torch.ones(points.shape[0], dtype=torch.bool, device="cuda")
        in_mask = in_mask & (opacity > 0).squeeze(-1)
    scaled_colors_all = torch.clip(colors * 0.282 + 0.5, 0, 1)
    if params:
        scaled_filtered_points = scaler.fit_transform(points[in_mask])
        scaled_filtered_colors = scaled_colors_all[in_mask]
    else:
        scaled_filtered_points = scaler.fit_transform(points)
        scaled_filtered_colors = scaled_colors_all
    if language_feature is not None and fea_enable:
        language_feature = language_feature[in_mask]
        if decoder is not None:
            decoder.eval()
            language_feature = decoder(language_feature.unsqueeze(0).unsqueeze(0).permute(0, 3, 1, 2))
            language_feature = language_feature.squeeze().transpose(0, 1)
            decoder.train()

        feature_for_cluster = torch.zeros(language_feature.shape[0], params["PCA_DIM"], device=language_feature.device)
        pca_mask = torch.all(language_feature == 0, dim=1)
        pca = PCA(n_components=params["PCA_DIM"])
        pca.fit(language_feature[~pca_mask])
        feature_pca = pca.transform(language_feature[~pca_mask])
        scaled_feature = scaler.fit_transform(torch.as_tensor(feature_pca, device=language_feature.device))
        feature_for_cluster[~pca_mask] = scaled_feature
        scaled_feature_for_cluster = feature_for_cluster
        scaled_points = torch.cat([scaled_filtered_points, scaled_filtered_colors, scaled_feature_for_cluster], dim=1)
    else:
        scaled_points = torch.cat([scaled_filtered_points, scaled_filtered_colors], dim=1)

    if not params:
        clusterer = HDBSCAN(cluster_selection_epsilon=0.01, min_cluster_size=200, min_samples=30, max_cluster_size=30000).fit(scaled_points)
        labels = torch.as_tensor(clusterer.labels_, dtype=torch.int32, device="cuda")
    else:
        clusterer = HDBSCAN(cluster_selection_epsilon=0.01, min_cluster_size=params['MIN_SAMPLES'], min_samples=params['MIN_SAMPLES'], max_cluster_size=params['MAX_CLUSTER_SIZE']).fit(scaled_points)
        labels_in_mask = clusterer.labels_.copy()
        labels = -torch.ones(points.shape[0], dtype=torch.int32, device="cuda")
        labels[in_mask] = torch.as_tensor(labels_in_mask, dtype=torch.int32, device="cuda")
        
    label_ids = torch.unique(labels)
    new_group = {}
    for id in label_ids:
        if id == -1:
            continue
        obj_id = torch.where(labels == id)[0]
        new_group[len(new_group) + 1] = obj_id
    return new_group

def group_to_id(gaussians, groups, max_groups=50):
    n = gaussians.get_xyz.shape[0]
    # keep the topk largest groups
    sorted_groups = sorted(groups.values(), key=lambda x: len(x), reverse=True)
    top_sets = sorted_groups[:max_groups - 1]
    keep_groups = {key: value for key, value in zip(torch.arange(1, max_groups, dtype=torch.int), top_sets)}
    keep_groups[0] = torch.zeros(0, dtype=torch.int, device="cuda")

    lang_ids = torch.zeros(n, len(keep_groups), dtype=torch.float32, device="cuda")
    id_matrix = torch.eye(len(keep_groups), dtype=torch.int, device="cuda")
    for group_id, obj_id in keep_groups.items():
        if len(obj_id) > 0:
            lang_ids[obj_id] += id_matrix[group_id]

    if lang_ids.shape[1] < max_groups:
        lang_ids = F.pad(lang_ids, (0, max_groups - lang_ids.shape[1]), value=0)
    return lang_ids

def id_to_group(ids):
    labels = torch.argmax(ids, dim=1)
    groups = {}
    unique_ids = torch.unique(labels)
    for id in unique_ids:
        if id == 0:
            continue
        obj_id = torch.where(labels == id)[0]
        groups[id] = obj_id
    return groups

def push_noise_points_to_nbr(gaussians, salient_idx=None, thresh=0.01):
    # noise point to neighbor label
    if salient_idx is None:
        positions = minmax_scaler(gaussians.get_xyz)
        language_ids = gaussians.get_language_id
    else:
        positions = minmax_scaler(gaussians.get_xyz[salient_idx])
        language_ids = gaussians.get_language_id[salient_idx]
    labels = language_ids.argmax(dim=1)

    noise_index = labels == 0
    noise_points = positions[noise_index]
    fg_points = positions[~noise_index]
    labels_non_noise = labels[~noise_index]

    pos_nn_model = NearestNeighbors(
        n_neighbors=1, algorithm="auto", metric="euclidean"
    ).fit(fg_points)
    distances, pos_nn_indices = pos_nn_model.kneighbors(noise_points)
    distances = torch.as_tensor(distances[..., 0], device='cuda')

    pos_nn_indices = torch.as_tensor(pos_nn_indices[..., 0], device='cuda')
    nbr_indices = distances < thresh
    pos_indices = pos_nn_indices[nbr_indices]
    pos_feat_idx = labels_non_noise[pos_indices]

    one_hot_tensor = torch.zeros(pos_feat_idx.shape[0], MAX_GROUP, device='cuda')
    one_hot_tensor.scatter_(1, pos_feat_idx.unsqueeze(1), 1)

    indices = torch.arange(language_ids.size(0), device='cuda')[noise_index][nbr_indices]
    language_ids[indices] = one_hot_tensor
    if salient_idx is not None:
        full_language_ids = torch.zeros(gaussians.get_language_id.shape[0], MAX_GROUP, device='cuda')
        full_language_ids[salient_idx] = language_ids
        gaussians.set_lang_id(full_language_ids)
    else:
        gaussians.set_lang_id(language_ids)

def mask_norm(mask):
    mask_min = mask.view(-1).min()
    mask_max = mask.view(-1).max()
    mask = (mask - mask_min) / (mask_max - mask_min)
    return mask

def get_ins_feature(gt_image, img_id, upsampler, transform, mask_prob, postprocessor=None):
    def get_bounding_box(mask, padding=5):
        y, x = torch.nonzero(mask, as_tuple=True)
        box = [max(0, x.min() - padding), max(0, y.min() - padding), min(gt_image.shape[2], x.max() + padding),
               min(gt_image.shape[1], y.max() + padding)]
        return box

    global INS_FEAT, ID_MAP, MASK_ORI, MASK_CRF

    if img_id in INS_FEAT:
        ins_feat = INS_FEAT[img_id]
        id_map = ID_MAP[img_id]
        ins_feat_map = ins_feat.index_select(0, id_map.view(-1)).view(1, 512, id_map.shape[0], id_map.shape[1])
        return ins_feat_map, MASK_ORI[img_id], MASK_CRF[img_id]

    h, w = gt_image.shape[1], gt_image.shape[2]
    norm_img = norm(gt_image)
    masks = mask_prob.argmax(dim=0)
    mask_ids = torch.unique(masks)
    mask_to_merge = torch.zeros((0, h, w), dtype=torch.bool, device="cuda")

    for i in range(len(mask_ids)):
        mask_id = mask_ids[i]
        if mask_id == 0 or torch.sum(masks == mask_id) < 500:
            continue
        else:
            mask = masks == mask_id
            mask_to_merge = torch.cat((mask_to_merge, mask.unsqueeze(0)), dim=0)
    id_map = torch.zeros(h, w, dtype=torch.int64, device="cuda")
    if len(mask_to_merge) <= 1:
        return torch.zeros(1, 512, h, w, dtype=torch.float32, device="cuda"), None, None
    mask_final = torch.cat(
        (torch.zeros((1, h, w), dtype=torch.float, device="cuda") + 0.1, mask_to_merge.to(dtype=torch.float)), dim=0)

    i = 1
    ds_size = (h//2, w//2)
    ds_gt_image = F.interpolate(gt_image.unsqueeze(0), size=ds_size, mode='bilinear', align_corners=False).squeeze(0)
    ds_mask_final = F.interpolate(mask_final.unsqueeze(0), size=ds_size, mode='bilinear', align_corners=False).squeeze(0)
    
    mask_crf = postprocessor(np.array(ds_gt_image.permute(1, 2, 0).detach().cpu() * 255, dtype='uint8'),
                             F.softmax(ds_mask_final, dim=0).detach().cpu().numpy())
    
    mask_crf = torch.tensor(mask_crf, dtype=torch.float32, device="cuda")
    mask_crf = F.interpolate(mask_crf.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
    
    patch_imgs = torch.zeros((0, 3, 224, 224), dtype=torch.float32, device="cuda")
    for mask in mask_crf[1:]:
        mask = mask_norm(mask) > 0.5
        if torch.sum(mask) > 300 and torch.sum(mask) < 30000:
            id_map[torch.logical_and(mask, ~id_map.bool())] = i
            i += 1

            mask = dilate(mask.to(dtype=torch.float32).unsqueeze(0), ksize=15)
            mask = mask.squeeze(0).to(dtype=torch.bool)
            mask_bbox = get_bounding_box(mask, padding=10)

            patch_img = norm_img.clone()
            patch_img = patch_img[:, mask_bbox[1]:mask_bbox[3], mask_bbox[0]:mask_bbox[2]]
            patch_img = resize(patch_img)
            patch_imgs = torch.cat((patch_imgs, patch_img.unsqueeze(0)), dim=0)
    if patch_imgs.shape[0] == 0:
        return torch.zeros(1, 512, h, w, dtype=torch.float32, device="cuda"), None, None
    with torch.no_grad():
        ins_feat = upsampler.model.model.encode_image(patch_imgs).to(dtype=torch.float32)

    ins_feat = torch.cat((torch.zeros(1, 512, dtype=torch.float32, device="cuda"), ins_feat), dim=0)

    INS_FEAT[img_id] = ins_feat
    ID_MAP[img_id] = id_map
    MASK_ORI[img_id] = mask_to_merge
    MASK_CRF[img_id] = mask_crf[1:]

    ins_feat_map = ins_feat.index_select(0, id_map.view(-1)).view(1, 512, id_map.shape[0], id_map.shape[1])

    return ins_feat_map, mask_to_merge, mask_crf[1:]

def save_pcd(gaussians, iteration, args, postfix=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gaussians.get_xyz.detach().cpu().numpy())
    max_indices = torch.argmax(gaussians.get_language_id, dim=1)
    color_map = generate_unique_colors(MAX_GROUP)
    colors = color_map[max_indices.detach().cpu().numpy()]
    colors[torch.all(gaussians.get_language_id == 0, dim=1).detach().cpu().numpy()] = [0, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    save_name = os.path.join(args.model_path, args.exp_name, 'clustered_pc', f'clustered_point_cloud{iteration}.ply')
    if postfix is not None:
        save_name = save_name.replace('.ply', f'_{postfix}.ply')
    o3d.io.write_point_cloud(save_name, pcd)

def calculate_regularization_loss(gaussians, salient_idx, scaled_points_all, pos_nn_model, sample_rate=0.01):
    salient_features = gaussians.get_language_feature_decode[salient_idx]

    salient_features = F.normalize(salient_features, p=2, dim=1)

    points_num = scaled_points_all.shape[0]
    sample_num = int(sample_rate * points_num)
    sample_idx = torch.randperm(points_num)[:sample_num]

    sampled_points = scaled_points_all[sample_idx]
    sampled_features = salient_features[sample_idx]

    _, pos_nn_indices = pos_nn_model.kneighbors(sampled_points)
    pos_nn_indices = torch.as_tensor(pos_nn_indices, device='cuda')

    nearest_idx = pos_nn_indices[:, 1:3]

    weights = torch.zeros(pos_nn_indices.shape[1], device='cuda')
    weights[points_num // 2:] = 1
    random_indices = torch.multinomial(weights, 5, replacement=False).expand(sample_num, -1)

    farthest_idx = torch.gather(pos_nn_indices, 1, random_indices)

    nearest_idx = nearest_idx.reshape(-1)
    farthest_idx = farthest_idx.reshape(-1)

    positive_sampled_features = sampled_features.repeat(2, 1).reshape(-1, 512)
    negative_sampled_features = sampled_features.repeat(5, 1).reshape(-1, 512)

    # dot product as loss
    positive_loss = torch.sigmoid(1 - torch.sum(positive_sampled_features * salient_features[nearest_idx].detach(), dim=1)).mean()
    negative_loss = torch.sigmoid(torch.sum(negative_sampled_features * salient_features[farthest_idx].detach(), dim=1)).mean()

    loss_regularization = positive_loss + negative_loss

    del pos_nn_indices
    torch.cuda.empty_cache()
    return loss_regularization

def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    postprocessor = DenseCRF()
    upsampler = torch.hub.load("./ckpts/FeatUp", 'maskclip', source='local', use_norm=False).to("cuda")
    upsampler.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        norm
    ])

    global PIXEL_FEAT, INS_FEAT, ID_MAP, MASK_ORI, MASK_CRF
    dataset_length = len(scene.train_cameras[1.0])
    INS_FEAT = {}
    ID_MAP = {}
    MASK_ORI = {}
    MASK_CRF = {}
    with torch.no_grad():
        for camera in scene.train_cameras[1.0]:
            gt_img = camera.original_image
            norm_img = transform(gt_img.unsqueeze(0))
            pix_feat = upsampler(norm_img)
            PIXEL_FEAT[camera.image_name] = pix_feat

    H, W = args.extra_params["HEIGHT"], args.extra_params["WIDTH"]
    kernel_size = (H//224, W//224)
    final_size = 224
    downsampler = SimpleDownsampler(kernel_size, final_size).to(dtype=torch.float32, device="cuda")
    feature_out_dim = args.clip_feature_dim
    feature_in_dim = opt.language_features_dim
    cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim).to(device="cuda", dtype=torch.float32)
    cnn_decoder_optimizer = torch.optim.Adam([*cnn_decoder.parameters(), *downsampler.parameters()], lr=0.0001)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        else:
            decoder_checkpoint = torch.load(checkpoint.replace("lang", "decoder"))
            cnn_decoder.load_state_dict(decoder_checkpoint)
            downsampler_checkpoint = torch.load(checkpoint.replace("lang", "downsampler"))
            downsampler.load_state_dict(downsampler_checkpoint)
            optimizer_state_dict = torch.load(checkpoint.replace("lang_chkpnt", "optim"))
            cnn_decoder_optimizer.load_state_dict(optimizer_state_dict)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not checkpoint:
        raise ValueError("checkpoint missing!!!!!")

    viewpoint_stack = scene.getTrainCameras().copy()
    extra_params = args.extra_params

    positions = gaussians.get_xyz
    groups = cluster_gauss_sh(gaussians, {}, iteration=0, fea_enable=False)
    groups_xyz = []
    for _, idx in groups.items():
        group_xyz = torch.mean(positions[idx], dim=0)
        groups_xyz.append(group_xyz)
    groups_xyz = torch.stack(groups_xyz)

    cluster = HDBSCAN(min_cluster_size=3)
    cluster.fit_predict(groups_xyz)

    labels = torch.tensor(cluster.labels_, dtype=torch.int32, device="cuda")
    labels_ins = labels[labels != -1]
    labels_unique = torch.unique(labels_ins)
    idx = torch.topk(torch.bincount(labels_ins), 1).indices
    select_label = labels_unique[idx]

    select_groups = torch.where(labels == select_label)[0]
    salient_region_pos = []
    for select_group_id in select_groups:
        group_pos = positions[groups[int(select_group_id + 1)]]
        salient_region_pos.append(group_pos)
    salient_region_pos = torch.cat(salient_region_pos, dim=0)
    
    salient_region_center = torch.mean(salient_region_pos, dim=0)
    region_radius = torch.max(torch.norm(salient_region_pos - salient_region_center, dim=1)) / args.extra_params["RADIUS_RATIO"]

    distances = torch.norm(positions - salient_region_center, dim=1)
    extra_params['salient_idx'] = distances <= region_radius
    gaussians.set_salient_idx(extra_params['salient_idx'])

    with torch.no_grad():
        groups = cluster_gauss_sh(gaussians, None, fea_enable=(first_iter != 0), params=extra_params, decoder=cnn_decoder)

    lang_ids = group_to_id(gaussians, groups, max_groups=MAX_GROUP)
    gaussians.set_lang_id(lang_ids)

    clustered_pc_path = os.path.join(args.model_path, args.exp_name, 'clustered_pc')
    if os.path.exists(clustered_pc_path):
        shutil.rmtree(clustered_pc_path)
    os.makedirs(clustered_pc_path, exist_ok=True)
    save_pcd(gaussians, first_iter, args)

    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), initial=first_iter, total=opt.iterations, desc="Training progress")
    first_iter += 1

    salient_gaussians = gaussians.get_xyz[extra_params['salient_idx']]
    scaled_points_all = minmax_scaler(salient_gaussians)

    pos_nn_model = NearestNeighbors(
        n_neighbors=salient_gaussians.shape[0], algorithm="auto", metric="euclidean"
    ).fit(scaled_points_all)

    loss_infoNce = InfoNCE(negative_mode='unpaired')
    
    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        loss_regularization = torch.tensor(0, dtype=torch.float32, device="cuda")
        loss_contrastive = torch.tensor(0, dtype=torch.float32, device="cuda")
        loss_ins = torch.tensor(0, dtype=torch.float32, device="cuda")

        gaussians._language_feature_decode = cnn_decoder(
            gaussians.get_language_feature.view(1, 1, -1, FEATURE_DIM).permute(0, 3, 1, 2)).squeeze().transpose(0, 1)

        if iteration >= 1000 and iteration % 10 == 0 and args.loss_regularization_weight > 0:
            loss_regularization = calculate_regularization_loss(gaussians, extra_params['salient_idx'], scaled_points_all, pos_nn_model, sample_rate=0.005)
        if iteration >= args.contrastive_iteration:
            if iteration % args.cluster_interval == 0 or 'indices' not in locals():
                with torch.no_grad():
                    groups = cluster_gauss_sh(gaussians, {}, fea_enable=True, iteration=iteration, params=extra_params, decoder=cnn_decoder)
                lang_ids = group_to_id(gaussians, groups, max_groups=MAX_GROUP)
                gaussians.set_lang_id(lang_ids)

                push_noise_points_to_nbr(gaussians, extra_params['salient_idx'], thresh=0.003)
                save_pcd(gaussians, iteration, args, postfix="noise_pushed")
                torch.save((gaussians.capture(opt.include_feature), iteration),
                           os.path.join(scene.model_path, args.exp_name, "lang_chkpnt" + str(iteration) + "_noise_pushed.pth"))
                group_combine(gaussians, cnn_decoder, extra_params['salient_idx'])
                save_pcd(gaussians, iteration, args)

                groups = id_to_group(gaussians.get_language_id)

                for key in list(INS_FEAT.keys()):
                    if INS_FEAT[key] is not None:
                        del INS_FEAT[key]
                        del ID_MAP[key]
                        del MASK_ORI[key]   
                        del MASK_CRF[key]
                INS_FEAT = {}
                ID_MAP = {}
                MASK_ORI = {}
                MASK_CRF = {}
                torch.cuda.empty_cache()
                gc.collect()
                mempool.free_all_blocks()

                group_xyzs = torch.zeros((0, 3), device="cuda")
                for _, idx in groups.items():
                    group_xyzs = torch.cat(
                        (group_xyzs, torch.mean(gaussians.get_xyz[idx], dim=0).unsqueeze(0)), dim=0)
                distances = torch.cdist(group_xyzs, group_xyzs)
                _, indices = torch.topk(distances, k=len(group_xyzs) - 1, dim=1, largest=True)

            if iteration % 10 == 0 and args.loss_contrastive_weight > 0:
                group_feats = []

                gaussian_features = F.normalize(cnn_decoder(
                    gaussians.get_language_feature.view(1, 1, -1, FEATURE_DIM).permute(0, 3, 1, 2)).squeeze().transpose(0, 1), p=2, dim=1)
                group_mean_feats = torch.zeros((0, 512), device="cuda")
                for _, idx in groups.items():
                    group_mean_feat = torch.mean(gaussian_features[idx], dim=0).unsqueeze(0)
                    group_mean_feats = torch.cat((group_mean_feats, group_mean_feat), dim=0)
                    group_feats.append(gaussian_features[idx])

                group_feat_similarity = torch.matmul(group_mean_feats, group_mean_feats.T)

                for i, group_feat in enumerate(group_feats):
                    positive_feat = group_mean_feats[i: i + 1].repeat(group_feat.shape[0], 1)
                    farthest_idx = indices[i]
                    not_similar_nbr = group_feat_similarity[i][farthest_idx] < 0.96
                    farthest_idx = farthest_idx[not_similar_nbr]

                    negative_feat = group_mean_feats[farthest_idx]
                    loss_contrastive += loss_infoNce(group_feat, positive_feat.detach(), negative_feat.detach())
                loss_contrastive = loss_contrastive / len(groups.keys())

                for group_feat in group_feats:
                    del group_feat
                del group_feats

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        mask_prob = None
        if iteration < args.contrastive_iteration:
            language_feature = render(viewpoint_cam, gaussians, pipe, background, opt, override_color=gaussians.get_language_feature)
        else:
            render_map = render(viewpoint_cam, gaussians, pipe, background, opt, override_color=torch.cat((gaussians.get_language_feature, gaussians.get_language_id), dim=1))
            language_feature, mask_prob = render_map[:FEATURE_DIM], render_map[FEATURE_DIM:]

        pix_feat = PIXEL_FEAT[viewpoint_cam.image_name]

        language_feature = language_feature.unsqueeze(0)
        language_feature = cnn_decoder(language_feature)
        ds_language_feature = F.interpolate(language_feature, size=(224*kernel_size[0], 224*kernel_size[1]), mode='bicubic', align_corners=True)
        ds_language_feature = downsampler(ds_language_feature)

        loss_pix = l1_loss(ds_language_feature, pix_feat)

        ins_flag = len(INS_FEAT) < 200 or viewpoint_cam.image_name in INS_FEAT

        if iteration >= args.contrastive_iteration and args.loss_ins_weight > 0 and ins_flag:
            gt_image = viewpoint_cam.original_image.cuda()

            with torch.no_grad():
                ins_feat, mask_ori, _ = get_ins_feature(gt_image, viewpoint_cam.image_name, upsampler, transform, mask_prob, postprocessor=postprocessor)
            
            if mask_ori is not None:
                loss_ins = ins_l1_loss(language_feature, ins_feat)
        
        loss = args.loss_pix_weight * loss_pix + args.loss_ins_weight * loss_ins + args.loss_contrastive_weight * loss_contrastive + args.loss_regularization_weight * loss_regularization

        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            assert not torch.isnan(loss).any(), f"Loss is NaN at iteration {iteration}"
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                cnn_decoder_optimizer.step()
                cnn_decoder_optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(opt.include_feature), iteration),
                           os.path.join(scene.model_path, args.exp_name, "lang_chkpnt" + str(iteration) + ".pth"))
                torch.save(cnn_decoder.state_dict(), os.path.join(scene.model_path, args.exp_name, "decoder_chkpnt" + str(iteration) + ".pth"))
                torch.save(downsampler.state_dict(), os.path.join(scene.model_path, args.exp_name, "downsampler_chkpnt" + str(iteration) + ".pth"))
                torch.save(cnn_decoder_optimizer.state_dict(), os.path.join(scene.model_path, args.exp_name, "optim" + str(iteration) + ".pth"))

        if iteration % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            mempool.free_all_blocks()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 3_000, 7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 3_000, 5_000, 7_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_000, 3_000, 5_000, 7_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--clip_feature_dim", type=int, default=512)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--loss_ins_weight", type=float, default=0.3)
    parser.add_argument("--loss_pix_weight", type=float, default=1.0)
    parser.add_argument("--loss_contrastive_weight", type=float, default=0.05)
    parser.add_argument("--loss_regularization_weight", type=float, default=0.1)
    parser.add_argument("--contrastive_iteration", type=int, default=3000)
    parser.add_argument("--cluster_interval", type=int, default=2000)
    parser.add_argument("--config_file", type=str)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)

    extra_config_json = json.load(open(args.config_file))
    args.extra_params = extra_config_json

    safe_state(args.quiet)

    if os.path.exists(os.path.join(args.model_path, args.exp_name)) and len(os.listdir(os.path.join(args.model_path, args.exp_name))) != 0:
        print("exp already exists")
    os.makedirs(os.path.join(args.model_path, args.exp_name), exist_ok=True)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    torch.manual_seed(int(time.time()))
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
