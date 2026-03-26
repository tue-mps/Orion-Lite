# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# Modified from OmniDrive(https://github.com/NVlabs/OmniDrive)
# Copyright (c) Xiaomi, Inc. All rights reserved.
# ------------------------------------------------------------------------
# Modified from orion_distilled.py - replaces LLM with integrated transformer decoder

import torch
import torch.nn.functional as F
from mmcv.utils import auto_fp16
from mmcv.models import DETECTORS
import copy
import os
from mmcv.models.builder import build_head

from mmcv.core import bbox3d2result

from mmcv.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.models.utils.grid_mask import GridMask

from mmcv.utils.misc import locations

from mmcv.models import builder

from mmcv.utils.misc import MLN
from mmcv.models.utils.transformer import inverse_sigmoid
import torch.nn as nn
import os
import json
import mmcv
import time
import numpy as np
from mmcv.models.dense_heads.planning_head_plugin.metric_stp3 import PlanningMetric
from scipy.optimize import linear_sum_assignment
from mmcv.utils import force_fp32, auto_fp16
from mmcv.models.utils import  DistributionModule, PredictModel,  \
                                CustomTransformerDecoder, CustomTransformerDecoderLayer, SinusoidalPosEmb, gen_sineembed_for_position, \
                                    linear_relu_ln, py_sigmoid_focal_loss
from mmcv.models.bricks import Linear
from mmcv.models.builder import HEADS 
import pickle

import sys
sys.path.append("/mnt/adas7tb/jgu/Orion") # Ensure root is in path to find distill
from distill.student_model_with_orion_loss import OrionMimicModel

from diffusers.schedulers import DDIMScheduler
from mmcv.utils.misc import memory_refresh
from mmcv.models.utils import build_transformer
from mmcv.models.builder import HEADS, build_loss 


class CodeTimer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.time()
    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[{self.name}] took {time.time() - self.start:.4f}s")



@DETECTORS.register_module()
class OrionWithoutLLM(MVXTwoStageDetector):
    def __init__(self,
                 save_path='./results_planning_only/',
                 use_grid_mask=False,
                 embed_dims=256,
                 LID=True,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 depth_num=64,
                 depth_start = 1,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 map_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 pretrained=None,
                 fp16_infer=False, # for faster close-loop infer, infer without evaluation
                 fp16_eval=False,
                 fp32_infer=False,  # for infer without evaluation
                 fut_ts=6,
                 freeze_backbone=False,
                 use_col_loss = False,
                 use_diff_decoder=False,
                 use_mlp_decoder=False,
                 plan_anchor_path=None,
                 diff_loss_weight=2.0,
                 ego_fut_mode=20,
                 with_bound_loss=True,
                 noise_x_offset=12,
                 noise_x_scale=24,
                 noise_y_offset=10,
                 noise_y_scale=40,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
                 loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=1.0),
                 plan_cls_loss_smooth = False,
                 # OrionMimicModel config (replaces LLM and VAE)
                 mimic_model_path=None,  # Path to trained checkpoint from distill/runs
                 input_dim=4096,
                 hidden_dim=1024,
                 output_dim=4096,
                 num_layers=6,
                 num_heads=16,
                 dropout=0.1,
                 ):
        super(OrionWithoutLLM, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.save_path = save_path
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.stride = stride
        self.use_col_loss = use_col_loss
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.query_pos = nn.Sequential(
            nn.Linear(396, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.fut_ts = fut_ts
        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims)
        )

        self.ego_pose_pe = MLN(156)

        self.pts_bbox_head.query_pos = self.query_pos
        self.pts_bbox_head.time_embedding = self.time_embedding
        self.pts_bbox_head.ego_pose_pe = self.ego_pose_pe

        if map_head is not None:
            self.map_head = builder.build_head(map_head)
            self.map_head.query_pos = self.query_pos
            self.map_head.time_embedding = self.time_embedding
            self.map_head.ego_pose_pe = self.ego_pose_pe
        
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        if LID:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - depth_start) / (depth_num * (1 + depth_num))
            coords_d = depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            bin_size = (self.position_range[3] - depth_start) / depth_num
            coords_d = depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self.position_encoder = nn.Sequential(
                nn.Linear(depth_num*3, embed_dims*4),
                nn.ReLU(),
                nn.Linear(embed_dims*4, embed_dims),
            )
        
        # ========== OrionMimicModel (replaces LLM and VAE) ==========
        self.mimic_model = OrionMimicModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            with_bound_loss=with_bound_loss,
            use_col_loss=use_col_loss,
            loss_plan_reg=loss_plan_reg,
            loss_plan_bound=loss_plan_bound,
            loss_plan_col=loss_plan_col,
            loss_vae_gen=loss_vae_gen,
        )
        
        # Load trained weights if provided
        if mimic_model_path is not None:
            print(f"Loading OrionMimicModel from {mimic_model_path}...")
            checkpoint = torch.load(mimic_model_path, map_location='cpu')
            # Handle both direct state_dict and checkpoint with 'model_state_dict' key
            if 'model_state_dict' in checkpoint:
                self.mimic_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.mimic_model.load_state_dict(checkpoint, strict=False)
            print("OrionMimicModel loaded successfully!")
        
        self.use_diff_decoder = use_diff_decoder
        self.use_mlp_decoder = use_mlp_decoder
        self.with_bound_loss = with_bound_loss
        self.use_col_loss = use_col_loss
        
        # Loss functions for training (mimic_model has its own, but we keep these for compatibility)
        if not self.use_diff_decoder and not self.use_mlp_decoder:
            self.loss_plan_reg = build_loss(loss_plan_reg)
            self.loss_plan_bound = build_loss(loss_plan_bound)
            if self.use_col_loss:
                self.loss_plan_col = build_loss(loss_plan_col)
            self.loss_vae_gen = build_loss(loss_vae_gen)
        
        elif self.use_diff_decoder:
            self.plan_cls_loss_smooth = plan_cls_loss_smooth
            self.diff_loss_weight = diff_loss_weight
            self.diff_traj_cls_loss_weight = 10.0
            self.diff_traj_reg_loss_weight = 8.0
            self.noise_x_offset = noise_x_offset
            self.noise_x_scale = noise_x_scale
            self.noise_y_offset = noise_y_offset
            self.noise_y_scale = noise_y_scale
            self.ego_fut_mode = ego_fut_mode
            with open(plan_anchor_path, 'rb') as f:
                anchors = pickle.load(f)
            plan_anchor = np.array(anchors) # (20,6,2)
            assert self.ego_fut_mode == plan_anchor.shape[0]
            self.plan_anchor = nn.Parameter(
                torch.tensor(plan_anchor, dtype=torch.float32),
                requires_grad=False,
            ) # 20,6,2

            self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(4096, 1, 1,512*6),
                nn.Linear(4096, 4096),
            )
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(4096),
                nn.Linear(4096, 4096),
                nn.Mish(),
                nn.Linear(4096, 4096),
            )
            diff_decoder_layer = CustomTransformerDecoderLayer(
                num_poses=6,
                d_model=4096,
                d_ffn=4096,
                num_head=32,
            )
            self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
            self.diffusion_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="scaled_linear",
                prediction_type="sample",
            )
        elif self.use_mlp_decoder: 
            self.waypoint_decoder = nn.Sequential(
                nn.Linear(4096, 4096 // 2),
                nn.GELU(),
                nn.Linear(4096//2, 6*2),
            )
            self.waypoints_loss = nn.MSELoss(reduction='none')
            
        self.test_flag = False
        self.planning_metric = None
        if fp16_infer:
            self.img_backbone.half()
        self.fp16_infer = fp16_infer
        self.fp16_eval = fp16_eval
        assert fp16_infer if fp16_eval else True
        self.fp32_infer = fp32_infer
        assert not fp16_infer if fp32_infer else True

        self.freeze_backbone = freeze_backbone

    @property
    def with_map_head(self):
        """bool: Whether the detector has a map head."""
        return hasattr(self,
                       'map_head') and self.map_head is not None
        
    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()

        img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B), C, H, W)

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        return img_feats

    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi

    def position_embeding(self, data, memory_centers, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['cam_intrinsic'].size(0)

        intrinsic = torch.stack([data['cam_intrinsic'][..., 0, 0], data['cam_intrinsic'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = memory_centers.repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)

        return coords_position_embeding

    # @force_fp32(apply_to=('img'))
    def forward(self, data, return_loss=True):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        """
        if return_loss:
            losses = self.forward_train(**data)
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
            return outputs
        else:
            return self.forward_test(**data)
        
    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_attr_labels= None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      ego_fut_trajs = None,
                      **data):
        """Forward training function."""
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        
        img_metas = [img_meta[0] for img_meta in img_metas]

        data['img_feats'] = self.extract_feat(data['img'])
        losses = self.forward_pts_train(gt_bboxes_3d, gt_labels_3d, gt_attr_labels,map_gt_bboxes_3d, map_gt_labels_3d, img_metas, ego_fut_trajs,**data)

        return losses

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_attr_labels,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,   
                          img_metas,
                          ego_fut_trajs,
                          **data):
        """Forward function for point cloud branch."""
        B = data['img'].shape[0]
        location = self.prepare_location(img_metas, **data) # (6, 40, 40, 2)
        pos_embed = self.position_embeding(data, location, img_metas) # (1, 9600, 256)
        losses = dict()

        if self.with_pts_bbox:
            outs_bbox, det_query = self.pts_bbox_head(img_metas, pos_embed, **data) # (1, 257, 4096)
            vision_embeded_obj = det_query.clone()
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs_bbox, gt_attr_labels]
            if self.pts_bbox_head.pred_traffic_light_state:
                loss_inputs.append(data['traffic_state'])
                loss_inputs.append(data['traffic_state_mask'])
            if self.use_col_loss:
                loss, agent_outs = self.pts_bbox_head.loss(*loss_inputs)
            else:
                loss = self.pts_bbox_head.loss(*loss_inputs)
                agent_outs = None
            losses.update(loss)
            
        if self.with_map_head:
            outs_lane, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            # reference vad trans
            device = gt_labels_3d[0].device
            map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_3d)
            lane_pts = [F.pad(map_gt_bboxes.fixed_num_sampled_points.to(device),(0,1)) for map_gt_bboxes in map_gt_vecs_list]
            loss_inputs = [lane_pts, map_gt_labels_3d, outs_lane, img_metas]

            losses.update(self.map_head.loss(*loss_inputs))

        # ========== Use OrionMimicModel (replaces LLM and VAE) ==========
        vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (B, N, 4096)
        
        # Prepare lane predictions for loss computation
        lane_scores = outs_lane['all_lane_cls_one2one'][-1] if self.with_map_head else None
        lane_preds = outs_lane['all_lane_preds_one2one'][-1] if self.with_map_head else None
        if lane_preds is not None:
            for p in range(self.map_head.n_control):
                lane_preds[..., 3 * p].clamp_(min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
                lane_preds[..., 3 * p + 1].clamp_(min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
            lane_preds = lane_preds.reshape(lane_preds.shape[0],lane_preds.shape[1],-1,3)[...,:2]
        
        # Use OrionMimicModel which handles both transformer decoder and VAE planner
        mimic_outputs = self.mimic_model(
            vision_embeded=vision_embeded,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_masks=data.get('ego_fut_masks'),
            ego_fut_cmd=data.get('ego_fut_cmd'),
            lane_preds=lane_preds,
            lane_scores=lane_scores,
            agent_outs=agent_outs if self.use_col_loss else None,
            target_ego_feature=None,  # Not needed for inference
            return_loss=self.training,
        )
        
        # Extract predictions
        ego_fut_preds = mimic_outputs['ego_fut_preds']  # (B, ego_fut_mode, fut_ts, 2)
        
        # Update losses if in training mode
        if self.training and 'losses' in mimic_outputs:
            losses.update(mimic_outputs['losses'])
        elif self.use_diff_decoder:
            bs = B
            device = ego_feature.device
            # 1. add truncated noise to the plan anchor
            plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
            odo_info_fut = self.norm_odo(plan_anchor)
            timesteps = torch.randint(
                0, 50,
                (bs,), device=device
            )
            noise = torch.randn(odo_info_fut.shape, device=device)
            noisy_traj_points = self.diffusion_scheduler.add_noise(
                original_samples=odo_info_fut,
                noise=noise,
                timesteps=timesteps,
            ).float()
            noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(noisy_traj_points)

            ego_fut_mode = noisy_traj_points.shape[1]
            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=512)
           
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
            # 3. embed the timesteps
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, current_states, time_embed)
            targets = torch.cumsum(ego_fut_trajs,dim=-2).squeeze(1)
            trajectory_loss_dict = {}

            lane_scores = outs_lane['all_lane_cls_one2one'][-1]
            lane_preds = outs_lane['all_lane_preds_one2one'][-1]
            for p in range(self.map_head.n_control):
                lane_preds[..., 3 * p].clamp_(min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
                lane_preds[..., 3 * p + 1].clamp_(min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
            lane_preds = lane_preds.reshape(lane_preds.shape[0],lane_preds.shape[1],-1,3)[...,:2]
            for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
                trajectory_cls_loss, trajectory_reg_loss, trajectory_bound_loss = self.loss_planning_diffusion(poses_reg, poses_cls, targets, plan_anchor, data['ego_fut_masks'][:,0,0],lane_preds, lane_scores)
                trajectory_loss_dict[f"traj_diff_loss_cls_{idx}"] = trajectory_cls_loss
                trajectory_loss_dict[f"traj_diff_loss_reg_{idx}"] = trajectory_reg_loss
                trajectory_loss_dict[f"traj_diff_loss_bound_{idx}"] = trajectory_bound_loss
                
            losses.update(trajectory_loss_dict)
        elif self.use_mlp_decoder:
            waypoint = self.waypoint_decoder(current_states)
            waypoint = waypoint.reshape(-1,2)
            wp_loss = self.waypoints_loss(waypoint.to(torch.float32), ego_fut_trajs.view(-1, 2).to(torch.float32))
            if 'ego_fut_masks' in data: # ignore invalid fut trajs supervision
                wp_loss = (wp_loss * data['ego_fut_masks'].view(-1, 1)).mean()
            else:
                wp_loss = wp_loss.mean()
            wp_loss = torch.nan_to_num(wp_loss)
            losses.update(wp_loss=wp_loss)
            
        return losses
    
    def forward_test(self, img_metas, **data):
        if not self.test_flag: #for interval evaluation
            if self.with_pts_bbox:
                self.pts_bbox_head.reset_memory()
            if self.with_map_head:
                self.map_head.reset_memory()
            self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key not in ['img', 'gt_bboxes_3d']:
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        B = 1
        mapped_class_names = [
        'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
        ]

        # 2. Profile Position Embedding / View Transform
        with CodeTimer("2. Position Embed & Prep"):
            location = self.prepare_location(img_metas, **data)
            outs_roi = self.forward_roi_head(location, **data)
            pos_embed = self.position_embeding(data, location, img_metas)
        bbox_results = []
        if self.with_pts_bbox:
            with CodeTimer("3. Detection Head"):
                outs, det_query = self.pts_bbox_head(img_metas, pos_embed, **data)
                vision_embeded_obj = det_query.clone()
                if self.use_col_loss:
                    bbox_list = self.pts_bbox_head.get_motion_bboxes(
                    outs, img_metas)
                    for bboxes, scores, labels, trajs in bbox_list:
                        bbox_result = bbox3d2result(bboxes, scores, labels)
                        bbox_result['trajs_3d'] = trajs.cpu()
                        bbox_results.append(bbox_result)
                else:
                    bbox_list = self.pts_bbox_head.get_bboxes(
                        outs, img_metas)
                    for bboxes, scores, labels in bbox_list:
                        bbox_results.append(bbox3d2result(bboxes, scores, labels))
        
        lane_results = None 
        outs_lane = None
        if self.with_map_head:
            with CodeTimer("4. Map Head"):
                outs_lane, map_query = self.map_head(img_metas, pos_embed, **data)
                vision_embeded_map = map_query.clone()
                lane_results = self.map_head.get_bboxes(outs_lane, img_metas)
        generated_text = []
        metric_dict = {}
        if not (self.fp16_infer or self.fp32_infer) or self.fp16_eval :
            gt_attr_label = data['gt_attr_labels'][0].to('cpu')
            gt_bbox = data['gt_bboxes_3d'][0]
            fut_valid_flag = bool(data['fut_valid_flag'][0])
            gt_label = data['gt_labels_3d'][0].to('cpu')
            if self.use_col_loss:
                score_threshold = 0.6
                with torch.no_grad():
                    c_bbox_results = copy.deepcopy(bbox_results)
                    bbox_result = c_bbox_results[0]
                    # filter pred bbox by score_threshold
                    mask = bbox_result['scores_3d'] > score_threshold
                    bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
                    bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
                    bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
                    bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]
        
        # ========== Use OrionMimicModel (replaces LLM and VAE) ==========
        with CodeTimer("5. OrionMimicModel"):
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (1, N, 4096)
            
            # Prepare lane predictions
            lane_scores = outs_lane['all_lane_cls_one2one'][-1] if (self.with_map_head and outs_lane is not None) else None
            lane_preds = outs_lane['all_lane_preds_one2one'][-1] if (self.with_map_head and outs_lane is not None) else None
            if lane_preds is not None:
                for p in range(self.map_head.n_control):
                    lane_preds[..., 3 * p].clamp_(min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
                    lane_preds[..., 3 * p + 1].clamp_(min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
                lane_preds = lane_preds.reshape(lane_preds.shape[0],lane_preds.shape[1],-1,3)[...,:2]
            
            # Use OrionMimicModel which handles both transformer decoder and VAE planner
            if next(self.mimic_model.parameters()).device != vision_embeded.device:
                self.mimic_model = self.mimic_model.to(vision_embeded.device)
            with torch.no_grad():
                mimic_outputs = self.mimic_model(
                    vision_embeded=vision_embeded,
                    ego_fut_trajs=None,  # Not needed for inference
                    ego_fut_masks=None,
                    ego_fut_cmd=None,
                    lane_preds=lane_preds,
                    lane_scores=lane_scores,
                    agent_outs=None,
                    target_ego_feature=None,
                    return_loss=False,
                )
            
            # Extract predictions
            ego_fut_preds = mimic_outputs['ego_fut_preds']  # (B, ego_fut_mode, fut_ts, 2)
            # We don't generate text with the mimic model
            generated_text = [{"Q": "OrionMimicModel", "A": "Trajectory generated by OrionMimicModel (Transformer + VAE)"}]

        # Post-process predictions - EXACTLY matching original orion_distilled.py logic
        # Since we're using OrionMimicModel (which uses VAE), this matches the VAE path in original
        full_match = False
        # Note: qa_pretrain is not used in this model, so we skip that check
        # Since we're using mimic model (equivalent to use_gen_token=True), we use the VAE path
        if not self.use_diff_decoder and not self.use_mlp_decoder:  # VAE path (what OrionMimicModel uses)
            mask_active_cmd = data['ego_fut_cmd'][:,0,0] == 1
            ego_fut_preds_inactive = ego_fut_preds[~mask_active_cmd].to('cpu')
            ego_fut_preds = ego_fut_preds[mask_active_cmd].flatten(0,1).to('cpu') # (6, 2)

        # This matches the original: if self.use_gen_token or full_match:
        # Since we're using mimic model (equivalent to use_gen_token=True), we always enter this block
        ego_fut_preds = ego_fut_preds.to(torch.float32) # for fp16 infer
        if not self.use_diff_decoder:
            ego_fut_pred = ego_fut_preds.cumsum(dim=-2) 
        else:
            ego_fut_pred = ego_fut_preds
        if not (self.fp16_infer or self.fp32_infer) or self.fp16_eval:
            ego_fut_trajs = data['ego_fut_trajs'][0, 0]
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)
            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                    pred_ego_fut_trajs = ego_fut_pred[None].to('cpu'),
                    gt_ego_fut_trajs = ego_fut_trajs[None].to('cpu'),
                    gt_agent_boxes = gt_bbox,
                    gt_agent_feats = gt_attr_label.unsqueeze(0),
                    fut_valid_flag = fut_valid_flag # 当前帧是否涵盖6个轨迹
                )
            metric_dict.update(metric_dict_planner_stp3)
            lane_results[0]['fut_valid_flag'] = fut_valid_flag
        else:
            metric_dict.update({'fut_valid_flag': False})
            lane_results[0]['fut_valid_flag'] = False
        lane_results[0]['ego_fut_preds'] = torch.nan_to_num(ego_fut_pred)
        lane_results[0]['ego_fut_cmd'] = data['ego_fut_cmd']

        return bbox_results, generated_text, lane_results, metric_dict
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        with CodeTimer("1. Image Backbone"):
            data['img_feats'] = self.extract_feat(data['img'])
        bbox_list = [dict() for i in range(len(img_metas))]
        if data['img'].dim() == 4: # (6,3,640,640)
            data['img'] = data['img'].unsqueeze(0)
        bbox_pts, generated_text, lane_results, metric_dict = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
        bbox_list[0]['text_out'] = generated_text
        bbox_list[0]['pts_bbox'].update(lane_results[0])
       
        return bbox_list

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]

        odo_info_fut_x = 2*(odo_info_fut_x + self.noise_x_offset)/self.noise_x_scale -1 
        odo_info_fut_y = 2*(odo_info_fut_y + self.noise_y_offset)/self.noise_y_scale -1 
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * self.noise_x_scale - self.noise_x_offset
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * self.noise_y_scale - self.noise_y_offset
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag or pred_ego_fut_trajs.size(1)==6 :
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                metric_dict['plan_L2_{}s'.format(i+1)] = np.nan_to_num(traj_L2)
                metric_dict['plan_obj_col_{}s'.format(i+1)] = np.nan_to_num(obj_coll.mean().item())
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = np.nan_to_num(obj_box_coll.mean().item())
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict


    def loss_planning_diffusion(self,
                      ego_fut_preds,
                      ego_fut_cls,
                      ego_fut_gt,
                      plan_anchor,
                      ego_fut_masks,
                      lane_preds = None,
                      lane_score_preds = None,
                      ):
        bs, num_mode, ts, d = ego_fut_preds.shape
        target_traj = ego_fut_gt
        dist = torch.linalg.norm(target_traj.unsqueeze(1)[...,:2] - plan_anchor, dim=-1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        mode_masks = torch.zeros(*ego_fut_cls.shape[:2],device=ego_fut_cls.device)
        for mask, idx in zip(mode_masks, mode_idx):
            mask[idx] = 1
        mode_masks = mode_masks.to(torch.bool)
        cls_target = mode_idx
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,ts,d)
        best_reg = torch.gather(ego_fut_preds, 1, mode_idx).squeeze(1)
        # Calculate cls loss using focal loss
        target_classes_onehot = torch.zeros([bs, num_mode],
                                            dtype=ego_fut_cls.dtype,
                                            layout=ego_fut_cls.layout,
                                            device=ego_fut_cls.device)
        target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)

        loss_plan_l1_weight = ego_fut_masks[:, :, None]
        
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1,  1, 2)

        loss_plan_l1 = self.diff_traj_reg_loss_weight * self.loss_plan_reg(
            best_reg,
            target_traj,
            loss_plan_l1_weight
        )

        if lane_preds is not None and lane_score_preds is not None:
            loss_plan_bound = self.loss_plan_bound(
                best_reg,
                lane_preds,
                lane_score_preds,
                weight=ego_fut_masks,
                denormalize=False,
            )

        
        if self.plan_cls_loss_smooth:
            loss_plan_cls_weight = torch.clip(dist, min=0, max=10.)*10 # scale factor
            loss_plan_cls_weight[mode_masks] = 10.
            loss_plan_cls_weight *= ego_fut_masks.all(dim=-1).to(torch.float).unsqueeze(-1)
            avg_factor = bs*self.ego_fut_mode
            loss_cls = self.diff_traj_cls_loss_weight * py_sigmoid_focal_loss(
                ego_fut_cls,
                target_classes_onehot,
                weight=loss_plan_cls_weight,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                avg_factor=avg_factor
            )
        else:
            loss_plan_cls_weight = ego_fut_masks.all(dim=-1).to(torch.float)
            loss_cls = self.diff_traj_cls_loss_weight * py_sigmoid_focal_loss(
                ego_fut_cls,
                target_classes_onehot,
                weight=loss_plan_cls_weight,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                avg_factor=None
            )

        return loss_cls, loss_plan_l1, loss_plan_bound

