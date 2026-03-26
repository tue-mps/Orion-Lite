_base_ = ["./orion_student_exp2_orion_init.py"]

custom_imports = dict(
    imports=[
        "experiments.student_orion.orion_student_planner",
        "experiments.student_orion.vision_freeze_hook",
    ],
    allow_failed_imports=False,
)

model = dict(
    img_backbone=dict(frozen=True),
)

custom_hooks = [
    dict(type="StudentBackboneLoadDebugHook", fail_on_violation=True, run_once=True),
    dict(type="FreezeVisionEncoderHook", log_once=True),
]
