from mmcv.models import DETECTORS

from .orion_distilled_new import OrionDistilledNew


@DETECTORS.register_module()
class OrionDistilledAblationLayers(OrionDistilledNew):
    def __init__(self, student_model_path=None, student_num_layers=6, **kwargs):
        self.student_num_layers = int(student_num_layers)
        self.student_model_path = student_model_path

        if self.student_num_layers <= 0:
            raise ValueError("`student_num_layers` must be a positive int.")

        student_model_conf = dict(kwargs.pop("student_model_conf", {}) or {})
        student_model_conf["num_layers"] = self.student_num_layers

        super().__init__(
            student_model_path=student_model_path,
            student_model_conf=student_model_conf,
            **kwargs,
        )
