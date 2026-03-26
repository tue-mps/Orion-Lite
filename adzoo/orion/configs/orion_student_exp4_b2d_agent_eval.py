_base_ = ["./orion_student_exp4_orion_init_freeze_vision.py"]

# Align with orion_distill_new_agent_with_mimic: same pipeline and precision
ORION_ROOT = __import__("os").environ.get("ORION_ROOT", "/mnt/adas7tb/jgu/Orion")
llm_path = __import__("os").environ.get("ORION_QFORMER_PATH", ORION_ROOT + "/ckpts/pretrain_qformer")
use_gen_token = True

collect_keys = [
    "lidar2img",
    "cam_intrinsic",
    "timestamp",
    "ego_pose",
    "ego_pose_inv",
    "command",
]

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

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
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
    fp16_infer=True,
    img_backbone_ckpt=None,
)

inference_only_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True),
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
        type="LoadAnnoatationCriticalVQATest",
        load_type=["critical_qa"],
        tokenizer=llm_path,
        use_gen_token=use_gen_token,
        max_length=2048,
        desc_qa=False,
    ),
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
                keys=["img", "input_ids", "ego_fut_cmd", "vlm_labels", "can_bus"] + collect_keys,
            ),
        ],
    ),
]
