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

from sqlite3 import Timestamp
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

from ...datasets.data_utils.constants import IGNORE_INDEX, EGO_WAYPOINT_TOKEN
from mmcv.models import builder

# Removed LlavaLlama imports as we are replacing LLM
# from ...utils.llava_llama import LlavaLlamaForCausalLM, add_special_token
# from transformers import AutoTokenizer, GenerationConfig

from mmcv.utils.misc import load_model

from ..utils.positional_encoding import pos2posemb2d
import torch.nn as nn
import os
import json
import mmcv
from mmcv.utils.misc import MLN
from mmcv.models.utils.transformer import inverse_sigmoid
from pathlib import Path
import time
import re
import numpy as np
from mmcv.models.dense_heads.planning_head_plugin.metric_stp3 import PlanningMetric
from scipy.optimize import linear_sum_assignment
import cv2
from mmcv.utils import force_fp32, auto_fp16
from ..utils.freeze_module import freeze_module
from mmcv.models.utils import  DistributionModule, PredictModel,  \
                                CustomTransformerDecoder, CustomTransformerDecoderLayer, SinusoidalPosEmb, gen_sineembed_for_position, \
                                    linear_relu_ln, py_sigmoid_focal_loss
from mmcv.models.bricks import Linear
from mmcv.models.builder import HEADS 
import pickle

import sys
# sys.path.append("/mnt/adas7tb/jgu/Orion") # Ensure root is in path to find distill
# from distill.student_model import OrionStudent 
from distill.student_model_with_orion_loss import OrionMimicModel

from diffusers.schedulers import DDIMScheduler
import matplotlib.pyplot as plt
from mmcv.utils.misc import memory_refresh
from mmcv.models.utils import build_transformer
from mmcv.models.builder import HEADS, build_loss 

@DETECTORS.register_module()
class OrionDistilledNew(MVXTwoStageDetector):
    @staticmethod
    def _default_student_cfg():
        return dict(
            input_dim=4096,
            hidden_dim=1024,
            output_dim=4096,
            num_layers=6,
            num_heads=16,
            dropout=0.1,
        )

    def _resolve_student_cfg(self, student_model_conf):
        student_cfg = self._default_student_cfg()
        if student_model_conf is not None:
            student_cfg.update(student_model_conf)
            print(f"Using custom student config: {student_model_conf}")
        return student_cfg

    def _build_student_model(
        self,
        student_cfg,
        with_bound_loss,
        use_col_loss,
        loss_plan_reg,
        loss_plan_bound,
        loss_plan_col,
        loss_vae_gen,
    ):
        return OrionMimicModel(
            input_dim=student_cfg['input_dim'],
            hidden_dim=student_cfg['hidden_dim'],
            output_dim=student_cfg['output_dim'],
            num_layers=student_cfg['num_layers'],
            num_heads=student_cfg['num_heads'],
            dropout=student_cfg['dropout'],
            with_bound_loss=with_bound_loss,
            use_col_loss=use_col_loss,
            loss_plan_reg=loss_plan_reg,
            loss_plan_bound=loss_plan_bound,
            loss_plan_col=loss_plan_col,
            loss_vae_gen=loss_vae_gen,
        )

    def __init__(self,
                 save_path='./results_vlm/',
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
                 # lm_head=None, # Removed
                 # tokenizer=None, # Removed
                 train_cfg=None,
                 test_cfg=None,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 frozen=True,
                 use_lora=False,
                 pretrained=None,
                 fp16_infer=False, # for faster close-loop infer, infer without evaluation
                 fp16_eval=False,
                 fp32_infer=False,  # for infer without evaluation
                 fut_ts=6,
                 freeze_backbone=False,
                 use_col_loss = False,
                 use_gen_token=False,
                 use_critical_qa=False,
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
                 qa_pretrain=False,
                 temporal_prompt_input=False,
                 mix_qa_training=False,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
                 loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=1.0),
                 plan_cls_loss_smooth = False,
                 student_model_path=None,
                 student_model_conf=None,  # Config dict for student model (e.g. dict(num_layers=16))
                 student_load_strict=True,  # Load student weights with strict=True so keys must match exactly
                 load_checkpoint_verbose=False,  # If True, mmcv load_checkpoint will print "Model loaded. Missing: 0, Unexpected: 0"
                 ):
        super(OrionDistilledNew, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self._print_load_success = load_checkpoint_verbose
        self.save_path = save_path
        self.mix_qa_training = mix_qa_training
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

        self.tokenizer = None # No tokenizer needed
        
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
        
        use_critical_qa = use_critical_qa or qa_pretrain
        self.qa_pretrain = qa_pretrain
        
        # Removed self.lm_head loading
        self.lm_head = None 
        
        self.use_gen_token = use_gen_token
        self.use_diff_decoder = use_diff_decoder
        self.use_mlp_decoder = use_mlp_decoder
        
        # VAE components are now inside OrionMimicModel, so we do NOT initialize them here.
        # This prevents loading "Orion" weights into these attributes at the top level.
        
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
        self.temporal_prompt_input = temporal_prompt_input

        student_cfg = self._resolve_student_cfg(student_model_conf)
        self.student_model = self._build_student_model(
            student_cfg=student_cfg,
            with_bound_loss=with_bound_loss,
            use_col_loss=use_col_loss,
            loss_plan_reg=loss_plan_reg,
            loss_plan_bound=loss_plan_bound,
            loss_plan_col=loss_plan_col,
            loss_vae_gen=loss_vae_gen,
        )

        if student_model_path is not None:
            print(f"Loading Student Model (OrionMimicModel) from {student_model_path}...")

            # Load weights with strict=student_load_strict so student checkpoint keys must match exactly
            checkpoint = torch.load(student_model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # strict=student_load_strict: when True, raises if any keys missing/unexpected
            load_result = self.student_model.load_state_dict(state_dict, strict=student_load_strict)
            missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
            print(f"Student Model loaded from {student_model_path} (strict={student_load_strict}). Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print(
                "Student model initialized from config only. "
                "Expect the outer checkpoint to provide `student_model.*` weights."
            )
        self.student_model.eval()

    @property
    def with_map_head(self):
        """bool: Whether the detector has a map head."""
        return hasattr(self,
                       'map_head') and self.map_head is not None
        
    @property
    def with_lm_head(self):
        """bool: Whether the detector has a lm head."""
        return hasattr(self,
                       'lm_head') and self.lm_head is not None

    @staticmethod
    def _cuda_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _timer_start(self):
        self._cuda_sync()
        return time.perf_counter()

    def _timer_end_ms(self, start_t):
        self._cuda_sync()
        return (time.perf_counter() - start_t) * 1000.0
        
    @auto_fp16(apply_to=('img'), out_fp32=True)
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
                      input_ids=None,
                      vlm_labels=None,
                      ego_fut_trajs = None,
                      **data):
        
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        
        # We don't use input_ids/vlm_labels for Student Model training/inference (usually)
        # But if we want to support mix_qa_training or similar, we might need them.
        # For now, we assume Student replaces LLM and uses visual features + dummy tokens if needed.
        
        # Pad input_ids/vlm_labels if they exist (legacy support or if used elsewhere)
        if input_ids is not None and self.tokenizer is not None:
             # ... (Keep existing padding logic if needed, but we removed tokenizer)
             pass
        
        img_metas = [img_meta[0] for img_meta in img_metas]

        data['img_feats'] = self.extract_feat(data['img'])
        losses = self.forward_pts_train(gt_bboxes_3d, gt_labels_3d, gt_attr_labels,map_gt_bboxes_3d, map_gt_labels_3d, img_metas,input_ids, vlm_labels, None, ego_fut_trajs,**data)

        return losses


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_attr_labels,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,   
                          img_metas,
                          input_ids, 
                          vlm_labels, 
                          vlm_attn_mask,
                          ego_fut_trajs,
                          **data):
        
        B = data['img'].shape[0]
        location = self.prepare_location(img_metas, **data) # (6, 40, 40, 2)
        pos_embed = self.position_embeding(data, location, img_metas) # (1, 9600, 256)
        losses = dict()

        agent_outs = {}
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
            losses.update(loss)
            
        lane_preds = None
        lane_scores = None
        if self.with_map_head:
            outs_lane, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            # reference vad trans
            device = gt_labels_3d[0].device
            map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_3d)
            lane_pts = [F.pad(map_gt_bboxes.fixed_num_sampled_points.to(device),(0,1)) for map_gt_bboxes in map_gt_vecs_list]
            loss_inputs = [lane_pts, map_gt_labels_3d, outs_lane, img_metas]

            losses.update(self.map_head.loss(*loss_inputs))
            
            # Extract lane preds for planning loss
            lane_scores = outs_lane['all_lane_cls_one2one'][-1]
            lane_preds = outs_lane['all_lane_preds_one2one'][-1]
            for p in range(self.map_head.n_control):
                lane_preds[..., 3 * p].clamp_(min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
                lane_preds[..., 3 * p + 1].clamp_(min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
            lane_preds = lane_preds.reshape(lane_preds.shape[0],lane_preds.shape[1],-1,3)[...,:2]

        # Use Student Model (OrionMimicModel) for planning
        if self.student_model is not None:
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (B, 513, 4096)
            
            # Prepare inputs for OrionMimicModel
            # OrionMimicModel.forward expects inputs for loss calculation
            
            # We might need target_ego_feature if we want feature mimic loss, but we don't have LLM here.
            # If we are just training the VAE/Planning part or finetuning, target_ego_feature might be None or omitted.
            # The prompt implies we use student_model_with_orion_loss.py which supports losses.
            
            student_outputs = self.student_model(
                vision_embeded=vision_embeded,
                ego_fut_trajs=ego_fut_trajs,
                ego_fut_masks=data.get('ego_fut_masks'),
                ego_fut_cmd=data.get('ego_fut_cmd'),
                lane_preds=lane_preds,
                lane_scores=lane_scores,
                agent_outs=agent_outs,
                target_ego_feature=None, # No Teacher LLM available in this detector
                return_loss=True
            )
            
            student_losses = student_outputs['losses']
            losses.update(student_losses)

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
            if key not in ['img', 'input_ids','gt_bboxes_3d','vlm_labels']:
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, timing_ms=None, **data):
        """Test function of point cloud branch."""
        B = 1
        timing_ms = {} if timing_ms is None else dict(timing_ms)
        mapped_class_names = [
        'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
        ]

        qformer_start = self._timer_start()
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        pos_embed = self.position_embeding(data, location, img_metas)
        bbox_results = []
        if self.with_pts_bbox:
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
        if self.with_map_head:
            outs, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            lane_results = self.map_head.get_bboxes(outs, img_metas)
        timing_ms['q_former_ms'] = self._timer_end_ms(qformer_start)
        
        generated_text = []
        metric_dict = {}
        
        # Calculate Perception Metrics (Optional, copied from original)
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
                    mask = bbox_result['scores_3d'] > score_threshold
                    bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
                    bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
                    bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
                    bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]

        if self.student_model is not None:
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (1, 513, 4096)
            
            # Optimization: Ensure device
            if next(self.student_model.parameters()).device != vision_embeded.device:
                self.student_model = self.student_model.to(vision_embeded.device)
            
            with torch.no_grad():
                # Call OrionMimicModel
                # It returns outputs dict with 'ego_fut_preds'
                outputs = self.student_model(
                    vision_embeded=vision_embeded,
                    return_loss=False,
                    return_timing=True
                )
                ego_fut_preds = outputs['ego_fut_preds'] # (B, ego_fut_mode, fut_ts, 2)
                student_timing = outputs.get('latency_breakdown_ms', {})
                if isinstance(student_timing, dict):
                    for key, value in student_timing.items():
                        if isinstance(value, (int, float)):
                            timing_ms[key] = float(value)
            
            # We don't generate text
            generated_text = [{"Q": "Student", "A": "Trajectory generated by Student Model"}]
            
            # In inference, we usually pick the mode corresponding to the active command.
            # The Mimic model returns ALL modes.
            # Original Orion logic: 
            # if ego_fut_cmd is available, pick that mode.
            
            if 'ego_fut_cmd' in data:
                # data['ego_fut_cmd'] shape: (B, 1, ego_fut_mode) or (B, ego_fut_mode)
                cmd = data['ego_fut_cmd']
                while cmd.dim() > 2: cmd = cmd.squeeze(1)
                if cmd.dim() == 1: cmd = cmd.unsqueeze(0)
                
                # Pick active cmd
                mask_active_cmd = (cmd == 1)
                # If using multiple samples, iterate. Here B=1 for simple_test
                
                # ego_fut_preds: (B, ego_fut_mode, fut_ts, 2)
                # We want (fut_ts, 2) for the active mode
                
                # If mask_active_cmd has exactly one 1 per batch:
                if mask_active_cmd.sum() > 0:
                    # Select the trajectory for the active command
                    # Logic adapted from original:
                    # ego_fut_preds = ego_fut_preds[mask_active_cmd].flatten(0,1).to('cpu') 
                    # Note: ego_fut_preds[mask_active_cmd] selects the matching rows.
                    # shape becomes (N_active, fut_ts, 2) -> (1, fut_ts, 2) -> flatten -> (fut_ts, 2)
                    ego_fut_preds_selected = ego_fut_preds[mask_active_cmd]
                    if ego_fut_preds_selected.shape[0] > 0:
                        ego_fut_preds = ego_fut_preds_selected[0].to('cpu')
                    else:
                        ego_fut_preds = ego_fut_preds[0,0].to('cpu') # Fallback
                else:
                     ego_fut_preds = ego_fut_preds[0,0].to('cpu') # Fallback
            else:
                 ego_fut_preds = ego_fut_preds[0,0].to('cpu') # Fallback to mode 0

            # ego_fut_preds is now (fut_ts, 2)
            ego_fut_preds = ego_fut_preds.to(torch.float32)
            
            # Original Orion VAE output is usually relative displacements (offset from previous),
            # OR it produces positions directly?
            # In OrionDistilled, lines 943:
            # if not self.use_diff_decoder: ego_fut_pred = ego_fut_preds.cumsum(dim=-2)
            # The VAE decoder in Orion (and Mimic) usually outputs offsets.
            # Let's check OrionMimicModel structure.
            # It uses the same VAE decoder as Orion.
            # "ego_fut_decoder" (Linear layers).
            # Usually these output delta-xy. So cumsum is needed.
            
            ego_fut_pred = ego_fut_preds.cumsum(dim=-2)
            
            if not (self.fp16_infer or self.fp32_infer) or self.fp16_eval:
                ego_fut_trajs = data['ego_fut_trajs'][0, 0]
                ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)
                metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                        pred_ego_fut_trajs = ego_fut_pred[None].to('cpu'),
                        gt_ego_fut_trajs = ego_fut_trajs[None].to('cpu'),
                        gt_agent_boxes = gt_bbox,
                        gt_agent_feats = gt_attr_label.unsqueeze(0),
                        fut_valid_flag = fut_valid_flag 
                    )
                metric_dict.update(metric_dict_planner_stp3)
                lane_results[0]['fut_valid_flag'] = fut_valid_flag
            else:
                metric_dict.update({'fut_valid_flag': False})
                lane_results[0]['fut_valid_flag'] = False
            
            lane_results[0]['ego_fut_preds'] = torch.nan_to_num(ego_fut_pred)
            lane_results[0]['ego_fut_cmd'] = data['ego_fut_cmd']

        else:
            metric_dict.update({'fut_valid_flag': False})
            lane_results[0]['ego_fut_preds'] = torch.zeros((6, 2), dtype=torch.float32).to(location.device)
            lane_results[0]['ego_fut_cmd'] = data['ego_fut_cmd']
            lane_results[0]['fut_valid_flag'] = False

        if lane_results is not None and len(lane_results) > 0:
            lane_results[0]['latency_breakdown_ms'] = timing_ms

        return bbox_results, generated_text, lane_results, metric_dict, timing_ms
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        timing_ms = {}
        vision_start = self._timer_start()
        data['img_feats'] = self.extract_feat(data['img'])
        timing_ms['vision_encoder_ms'] = self._timer_end_ms(vision_start)
        bbox_list = [dict() for i in range(len(img_metas))]
        if data['img'].dim() == 4: # (6,3,640,640)
            data['img'] = data['img'].unsqueeze(0)
        bbox_pts, generated_text, lane_results, metric_dict, timing_ms = self.simple_test_pts(
            img_metas, timing_ms=timing_ms, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
        bbox_list[0]['text_out'] = generated_text
        bbox_list[0]['pts_bbox'].update(lane_results[0])
        bbox_list[0]['pts_bbox']['latency_breakdown_ms'] = timing_ms
       
        return bbox_list

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
