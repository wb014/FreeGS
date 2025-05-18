import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.layers import CNN_decoder

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt
from utils.scaler import MinMaxScaler


def generate_unique_colors(num_colors):
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    random.shuffle(colors)
    return np.array(colors)[:, :3]

def group_to_id(gaussians, groups, max_groups=50):
    n = gaussians.get_xyz.shape[0]
    sorted_groups = sorted(groups.values(), key=lambda x: len(x), reverse=True)
    top_sets = sorted_groups[:max_groups - 1]
    keep_groups = {key: value for key, value in zip(torch.arange(1, max_groups, dtype=torch.int), top_sets)}
    keep_groups[0] = set()

    lang_ids = torch.zeros(n, len(keep_groups), dtype=torch.float16, device="cuda")
    id_matrix = torch.eye(len(keep_groups), dtype=torch.int, device="cuda")
    for group_id, obj_id in keep_groups.items():
        if len(obj_id) > 0:
            lang_ids[torch.tensor(list(obj_id), device="cuda")] += id_matrix[group_id]
    zero_rows = torch.all(lang_ids == 0, dim=1)
    non_zero_rows = lang_ids[~zero_rows]
    normalized_non_zero_rows = non_zero_rows / non_zero_rows.norm(dim=1, p=1, keepdim=True)
    lang_ids[~zero_rows] = normalized_non_zero_rows
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

def group_combine(gaussians, decoder, salient_idx=None):
    ids = gaussians.get_language_id
    positions = gaussians.get_xyz

    pos_scaler = MinMaxScaler()
    pos_scaler.fit(positions) if salient_idx is None else pos_scaler.fit(positions[salient_idx])
    norm_positions = pos_scaler.transform(positions)
    dc_feat = gaussians.get_features[:, 0]
    colors = torch.clip(dc_feat * 0.282 + 0.5, 0, 1)
    features = gaussians.get_language_feature
    features = decoder(features.view(1, 1, -1, 128).permute(0, 3, 1, 2)).squeeze().transpose(0, 1)
    groups = id_to_group(ids)
    sorted_groups = sorted(groups.values(), key=lambda x: len(x), reverse=False)
    group_xyzs = []
    group_colors = torch.zeros((0, 3), device="cuda")
    group_feats = torch.zeros((0, 512), device="cuda")
    for group in sorted_groups:
        group_xyzs.append(norm_positions[group])
        group_feats = torch.cat((group_feats, torch.mean(features[group], dim=0, keepdim=True)), dim=0)
        group_colors = torch.cat((group_colors, torch.mean(colors[group], dim=0, keepdim=True)), dim=0)
    group_feats = F.normalize(group_feats, p=2, dim=1)
    group_pos_distances = torch.zeros((len(group_xyzs), len(group_xyzs)), device="cuda")
    for i in range(len(group_xyzs)):
        group_i = group_xyzs[i]
        for j in range(i + 1, len(group_xyzs)):
            group_j = group_xyzs[j]
            all_dist = torch.cdist(group_i, group_j)
            group_pos_distances[i, j] = group_pos_distances[j, i] = torch.kthvalue(all_dist.flatten(), int(all_dist.numel() * 0.005))[0]

    group_color_distances = torch.cdist(group_colors, group_colors)
    group_feat_similarity = torch.matmul(group_feats, group_feats.T)

    select_group_id = torch.ones(group_feats.shape[0], dtype=torch.int32, device="cuda")
    for i in range(len(sorted_groups)):
        neighbor_group = {}
        for j in range(i + 1, len(sorted_groups)):
            if group_pos_distances[i, j] < 0.015 and group_feat_similarity[i, j] > 0.98:
                neighbor_group[j] = group_feat_similarity[i, j]
        if len(neighbor_group) > 0:
            max_group = max(neighbor_group, key=neighbor_group.get)
            select_group_id[i] = 0
            sorted_groups[max_group] = torch.cat((sorted_groups[i], sorted_groups[max_group]))
            group_xyzs[max_group] = norm_positions[sorted_groups[max_group]]
            group_colors[max_group] = torch.mean(colors[sorted_groups[max_group]], dim=0, keepdim=True)
            group_feats[max_group] = F.normalize(torch.mean(features[sorted_groups[max_group]], dim=0, keepdim=True), p=2, dim=1)
            for k in range(i, len(sorted_groups)):
                if k == max_group:
                    continue
                all_dist = torch.cdist(group_xyzs[max_group], group_xyzs[k])
                group_pos_distances[max_group, k] = group_pos_distances[k, max_group] = all_dist.min()
                group_feat_similarity[max_group, k] = group_feat_similarity[k, max_group] = torch.matmul(group_feats[max_group], group_feats[k].T)
                group_color_distances[max_group, k] = group_color_distances[k, max_group] = torch.cdist(group_colors[max_group].unsqueeze(0), group_colors[k].unsqueeze(0))

    new_group = {}
    for i in range(len(sorted_groups)):
        if select_group_id[i] == 1 and len(sorted_groups[i]) > 200:
            new_group[len(new_group) + 1] = sorted_groups[i]
    lang_ids = group_to_id(gaussians, new_group)
    gaussians.set_lang_id(lang_ids)

