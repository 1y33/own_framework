import torch
from dataclasses import dataclass

### config zone
# optimizer, batchsize, lr ,extra things, optimizer schedular
@dataclass
class Config:
    batch_size = 1 
    model_config = None

    optimizer = "adam"
    lr = 1e-6
    schedular = "LrSchedular" | None 
    
    saving_steps = 3
    epochs = 1000
    
    save_dir = "path"
    max_saves = 4
    
    verbose_logger = None


### trainer utilities
# logger, save, generation

### loop
# loss function

