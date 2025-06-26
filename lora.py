# here i will build different loras 
# mostly lora like : simple lora , latent lora , quantized lora etc.

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class SimpleLora(nn.Module):
    def __init__(self, dim_in, dim_out, rank=1, device=None):
        super().__init__()
        
        self.A_weight = nn.Parameter(torch.zeros(rank, dim_out, device=device))
        self.B_weight = nn.Parameter(torch.zeros(dim_in, rank, device=device))
        
        nn.init.normal_(self.A_weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.B_weight)
        
        self.scale = 1 / rank
        self.enable = True
    
    def forward(self, base_weight):
        if self.enable:
            delta = torch.matmul(self.B_weight, self.A_weight).view_as(base_weight) * self.scale
            return base_weight + delta
        
        return base_weight

class LatentLora(SimpleLora):
    pass

class QuantizedLora(SimpleLora):
    pass


class LoraApplier(nn.Module):
    def __init__(self, model:nn.Module, lora_type="simple", rank=1, device=None,freeze=True):
        super().__init__()
        self.model = model
        self.device = device or next(model.parameters()).device
        self.rank = rank
        self.lora_type = lora_type
        
        self._apply_lora(freeze)
        self.is_lora = True
    
    def _make_lora(self, layer):
        dim_out, dim_in = layer.weight.shape
        if self.lora_type == "simple":
            return SimpleLora(dim_in, dim_out, self.rank, self.device)
        if self.lora_type == "latent":
            return LatentLora(dim_in, dim_out, self.rank, self.device)
        if self.lora_type == "quantized":
            return QuantizedLora(dim_in, dim_out, self.rank, self.device)
        raise ValueError(f"unsupported lora type : {self.lora_type}")
    
    def _apply_lora(self,freeze=True):
        self._freeze_model(freeze)
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                parametrize.register_parametrization(layer, "weight", self._make_lora(layer))
        
    def _freeze_model(self,freeze=True):
        for param in self.model.parameters():
            param.requires_grad = not freeze
            
    def set_enabled(self, flag: bool = True):
        for module in self.model.modules():
            if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
                for p in module.parametrizations.weight:
                    if isinstance(p, SimpleLora):
                        p.enable = flag
                        
                        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
