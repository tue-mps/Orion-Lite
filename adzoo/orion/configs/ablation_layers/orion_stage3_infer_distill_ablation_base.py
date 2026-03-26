_base_ = ["../orion_stage3_infer_distill_new.py"]

custom_imports = dict(
    imports=["mmcv.models.detectors.orion_distilled_ablation_layers"],
    allow_failed_imports=False,
)

model = dict(
    type="OrionDistilledAblationLayers",
    student_num_layers=6,
)

