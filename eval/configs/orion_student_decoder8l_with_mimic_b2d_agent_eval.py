import os

# Decoder layers: 8  |  Mimic: with_mimic
# Base config is included in this repo — no ORION_ROOT path dependency.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_base_ = [os.path.join(_REPO_ROOT, "eval/configs/base/orion_student_planner_base.py")]

model = dict(
    fp16_infer=True,
    img_backbone_ckpt=None,
    orion_ckpt=None,
    student_cfg=dict(input_dim=4096, hidden_dim=1024, output_dim=4096, num_layers=8, num_heads=16, dropout=0.1),
)

class_names = ["car","van","truck","bicycle","traffic_sign","traffic_cone","traffic_light","pedestrian","others"]
collect_keys = ["lidar2img","cam_intrinsic","timestamp","ego_pose","ego_pose_inv","command"]
ida_aug_conf = {"resize_lim":(0.37,0.45),"final_dim":(320,640),"bot_pct_lim":(0.0,0.0),"rot_lim":(0.0,0.0),"H":900,"W":1600,"rand_flip":False}
inference_only_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph",to_float32=True,file_client_args=dict(backend="disk"),img_root="data/bench2drive"),
    dict(type="ResizeCropFlipRotImage",data_aug_conf=ida_aug_conf,training=False),
    dict(type="ResizeMultiview3D",img_scale=(640,640),keep_ratio=False,multiscale_mode="value"),
    dict(type="NormalizeMultiviewImage",mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],to_rgb=True),
    dict(type="PadMultiViewImage",size_divisor=32),
    dict(type="MultiScaleFlipAug3D",img_scale=(1600,900),pts_scale_ratio=1,flip=False,
         transforms=[dict(type="PETRFormatBundle3D",class_names=class_names,collect_keys=collect_keys,with_label=False),
                     dict(type="CustomCollect3D",keys=["img","ego_fut_cmd","can_bus"]+collect_keys)]),
]
