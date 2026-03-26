_base_ = ["./orion_distill_new_agent_with_mimic.py"]

_student_input_dim = int(__import__("os").environ.get("ORION_STUDENT_INPUT_DIM", 4096))
_student_hidden_dim = int(__import__("os").environ.get("ORION_STUDENT_HIDDEN_DIM", 1024))
_student_output_dim = int(__import__("os").environ.get("ORION_STUDENT_OUTPUT_DIM", 4096))
_student_num_layers = int(__import__("os").environ.get("ORION_STUDENT_NUM_LAYERS", 6))
_student_num_heads = int(__import__("os").environ.get("ORION_STUDENT_NUM_HEADS", 16))
_student_dropout = float(__import__("os").environ.get("ORION_STUDENT_DROPOUT", 0.1))
_student_with_bound_loss = __import__("os").environ.get("ORION_STUDENT_WITH_BOUND_LOSS", "True").strip().lower() in (
    "1", "true", "yes", "y", "on"
)
_student_use_col_loss = __import__("os").environ.get("ORION_STUDENT_USE_COL_LOSS", "True").strip().lower() in (
    "1", "true", "yes", "y", "on"
)


model = dict(
    student_model_path=None,
    student_model_conf=dict(
        input_dim=_student_input_dim,
        hidden_dim=_student_hidden_dim,
        output_dim=_student_output_dim,
        num_layers=_student_num_layers,
        num_heads=_student_num_heads,
        dropout=_student_dropout,
    ),
    with_bound_loss=_student_with_bound_loss,
    use_col_loss=_student_use_col_loss,
)
