from trainer import Trainer, Config,shift_logits_labels
import torch
import torch.nn.functional as F

class PreTrain(Trainer):
    def compute_loss(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        
        outputs = self.model(input_ids)
        
        # Extract logits from HuggingFace model output
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs
        
        
        shift_logits, shift_labels = shift_logits_labels(logits, labels)
        
        
        mask = shift_labels != -100
        if mask.sum() == 0:
            return torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
        
        
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        
        return loss
    
    
