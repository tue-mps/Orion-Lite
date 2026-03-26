_base_ = ["./orion_stage3_infer_distill_ablation_base.py"]

student_model_path = "/mnt/adas7tb/jgu/Orion/distill/runs_ablation_layers/first_try_decoder2l_gpu0/checkpoints/last.pt"
model = dict(student_model_path=student_model_path, student_num_layers=2)

