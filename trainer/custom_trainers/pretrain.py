from trainer import Trainer, Config,shift_logits_labels
import torch
import torch.nn.functional as F

class PreTrain(Trainer):
    def compute_loss(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = self.model(input_ids)
        
        shift_logits,shift_labels = shift_logits_labels(logits,labels)
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        
        return loss
    
    
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

        student_logits = self.model(input_ids) 

        with torch.no_grad():
            teacher_logits = self.teacher_model(input_ids)["logits"]
            
        flat_s,flat_t,flat_l = shift_logits_labels([student_logits,teacher_logits], labels)

        ce = self.ce_loss_fn(flat_s, flat_l)

        log_p_s = F.log_softmax(flat_s / self.temperature, dim=-1)
        p_t = F.softmax(flat_t     / self.temperature, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.temperature ** 2)

        loss = self.alpha * kl + (1.0 - self.alpha) * ce
        return loss