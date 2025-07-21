from trainer import Trainer, Config,shift_logits_labels
import torch
import torch.nn.functional as F


class KDL(Trainer):
    def __init__(self, run_name: str, cfg: Config, teacher_model:torch.nn.Module, temperature:float=2.0, alpha:float=0.5):
        super().__init__(run_name, cfg)
        self.teacher_model = teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    def compute_loss(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels    = batch["labels"]

        student_outputs = self.model(input_ids)
        
        # Extract logits from HuggingFace model output for student
        if hasattr(student_outputs, 'logits'):
            student_logits = student_outputs.logits
        elif isinstance(student_outputs, dict) and 'logits' in student_outputs:
            student_logits = student_outputs['logits']
        else:
            # Fallback: assume outputs is the logits tensor directly
            student_logits = student_outputs

        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids)
            # Extract logits from HuggingFace model output for teacher
            if hasattr(teacher_outputs, 'logits'):
                teacher_logits = teacher_outputs.logits
            elif isinstance(teacher_outputs, dict) and 'logits' in teacher_outputs:
                teacher_logits = teacher_outputs['logits']
            else:
                # Fallback: assume outputs is the logits tensor directly
                teacher_logits = teacher_outputs
            
        flat_s,flat_t,flat_l = shift_logits_labels([student_logits,teacher_logits], labels)

        ce = self.ce_loss_fn(flat_s, flat_l)

        log_p_s = F.log_softmax(flat_s / self.temperature, dim=-1)
        p_t = F.softmax(flat_t     / self.temperature, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.temperature ** 2)

        loss = self.alpha * kl + (1.0 - self.alpha) * ce
        return loss