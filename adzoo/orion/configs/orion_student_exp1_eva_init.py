_base_ = ["./orion_student_base_no_vqa.py"]

model = dict(
    img_backbone_ckpt="ckpts/eva02_petr_proj.pth",
)

num_gpus = 8
batch_size = 32
num_epochs = 6
num_iters_per_epoch = 234769 // (num_gpus * batch_size)

optimizer = dict(
    lr=4e-5,
    paramwise_cfg=dict(
        head_decay_rate=1.0,
    ),
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    shuffler_sampler=dict(
        num_iters_to_seq=num_iters_per_epoch,
    ),
)

runner = dict(
    type="IterBasedRunner",
    max_iters=num_epochs * num_iters_per_epoch,
)

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
evaluation = dict(interval=num_iters_per_epoch * (num_epochs + 1))
