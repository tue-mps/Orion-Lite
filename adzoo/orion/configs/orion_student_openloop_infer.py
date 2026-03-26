# Open-loop evaluation config for Orion student models (single canonical config).
_base_ = ["./orion_student_base_no_vqa.py"]

# Inference loads full model from the main checkpoint provided to test.py.
model = dict(img_backbone_ckpt=None)

# Open-loop eval dataset settings (aligned with official stage3 infer without LLM).
ORION_ROOT = __import__("os").environ.get(
    "ORION_ROOT", __import__("os").path.abspath("{{fileDirname}}/../../..")
)
batch_size = 4
dataset_type = "B2DOrionDataset"
data_root = ORION_ROOT + "/data/bench2drive"
info_root = ORION_ROOT + "/data/infos"
map_root = data_root + "/maps"
map_file = info_root + "/b2d_map_infos.pkl"
file_client_args = dict(backend="disk")
ann_file_test = info_root + "/b2d_infos_val.pkl"

# Keep expression-sensitive objects local to avoid mmcv base substitution edge cases.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    "car",
    "van",
    "truck",
    "bicycle",
    "traffic_sign",
    "traffic_cone",
    "traffic_light",
    "pedestrian",
    "others",
]
collect_keys = [
    "lidar2img",
    "cam_intrinsic",
    "timestamp",
    "ego_pose",
    "ego_pose_inv",
    "command",
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
ida_aug_conf = {
    "resize_lim": (0.37, 0.45),
    "final_dim": (320, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

# Reuse stable shared objects from the base config.
NameMapping = {{_base_.NameMapping}}
input_modality = {{_base_.input_modality}}
past_frames = {{_base_.past_frames}}
future_frames = {{_base_.future_frames}}
map_fixed_ptsnum_per_gt_line = {{_base_.map_fixed_ptsnum_per_gt_line}}
eval_cfg = {{_base_.eval_cfg}}

test_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True,
    ),
    dict(type="VADObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="VADObjectNameFilter", classes=class_names),
    dict(type="ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=False),
    dict(
        type="ResizeMultiview3D",
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode="value",
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="PETRFormatBundle3D",
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False,
            ),
            dict(
                type="CustomCollect3D",
                keys=[
                    "gt_bboxes_3d",
                    "gt_labels_3d",
                    "img",
                    "ego_his_trajs",
                    "gt_attr_labels",
                    "ego_fut_trajs",
                    "ego_fut_masks",
                    "ego_fut_cmd",
                    "ego_lcf_feat",
                    "can_bus",
                    "fut_valid_flag",
                ]
                + collect_keys,
            ),
        ],
    ),
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        classes=class_names,
        name_mapping=NameMapping,
        map_root=map_root,
        map_file=map_file,
        modality=input_modality,
        past_frames=past_frames,
        future_frames=future_frames,
        point_cloud_range=point_cloud_range,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        eval_cfg=eval_cfg,
    ),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

log_config = dict(
    interval=10,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)
