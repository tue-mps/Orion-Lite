_base_ = ["./orion_stage2_train.py"]

custom_imports = dict(
    imports=["experiments.student_orion.orion_student_planner"],
    allow_failed_imports=False,
)

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

model = dict(
    type="OrionStudentPlanner",
    tokenizer=None,
    lm_head=None,
    use_lora=False,
    use_gen_token=False,
    student_cfg=dict(
        input_dim=4096,
        hidden_dim=1024,
        output_dim=4096,
        num_layers=6,
        num_heads=16,
        dropout=0.1,
    ),
    img_backbone_ckpt=None,
    img_backbone_prefix="img_backbone.",
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True,
        with_light_state=True,
    ),
    dict(type="VADObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="VADObjectNameFilter", classes=class_names),
    dict(type="ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=True),
    dict(
        type="ResizeMultiview3D",
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode="value",
    ),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys),
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
            "traffic_state_mask",
            "traffic_state",
        ]
        + collect_keys,
    ),
]

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
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)


evaluation = dict(pipeline=test_pipeline)

log_config = dict(
    interval=10,
    hooks=[dict(type="TextLoggerHook")],
)

custom_hooks = [
    dict(type="StudentBackboneLoadDebugHook", fail_on_violation=True, run_once=True),
]
