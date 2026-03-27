_base_ = ["/mnt/adas7tb/jgu/project_backup/Orion/adzoo/orion/configs/orion_distill_new_agent_with_mimic.py"]

ORION_RUNTIME_ROOT = __import__("os").environ.get("ORION_RUNTIME_ROOT", "/mnt/adas7tb/jgu/Orion")
llm_path = __import__("os").environ.get(
    "ORION_QFORMER_PATH",
    ORION_RUNTIME_ROOT + "/ckpts/pretrain_qformer",
)

# Re-declare the small set of backup constants needed by our local pipeline
# overrides. MMCV config inheritance merges dicts/lists, but names from the
# base module are not directly visible in this local shim.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
ida_aug_conf = {
    "resize_lim": (0.37, 0.45),
    "final_dim": (320, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
    'traffic_cone', 'traffic_light', 'pedestrian', 'others',
]
use_gen_token = True
collect_keys = ['lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command']

model = dict(
    student_model_path=__import__("os").environ.get(
        "ORION_STUDENT_CKPT",
        ORION_RUNTIME_ROOT + "/distill/runs/orion_student_with_mimic_loss/checkpoints/last.pt",
    )
)

# The backup config bakes llm_path into these pipeline entries, so we redefine
# the two pipelines locally with only the tokenizer path updated.
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='LoadAnnoatationCriticalVQATest',
        load_type=['critical_qa'],
        tokenizer=llm_path,
        use_gen_token=use_gen_token,
        max_length=2048,
        desc_qa=False,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False,
            ),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'input_ids', 'gt_attr_labels', 'ego_fut_trajs',
                    'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                    'vlm_labels', 'can_bus', 'fut_valid_flag',
                ] + collect_keys,
            ),
        ],
    ),
]

inference_only_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='LoadAnnoatationCriticalVQATest',
        load_type=['critical_qa'],
        tokenizer=llm_path,
        use_gen_token=use_gen_token,
        max_length=2048,
        desc_qa=False,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False,
            ),
            dict(
                type='CustomCollect3D',
                keys=['img', 'input_ids', 'ego_fut_cmd', 'vlm_labels', 'can_bus'] + collect_keys,
            ),
        ],
    ),
]
