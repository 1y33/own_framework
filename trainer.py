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

class Trainer :
    def __init__(self, config):
        
        self.config = config
        
    def train_step(self,intput,labels):

        loss = torch.nn.functional.cross_entropy(input,labels)

        return loss
    
    def training_loop(self):
        optimizer = None
        dataloader = None
        self.total_loss = []
        
        for epoch in self.config.epochs:
            self.epoch_loss = []
            for input,output in dataloader:
                ## assume input and output on the same device ( we will map it later ) 
                optimizer.zero_grad()
                
                loss = self.train_step(input,output)
                
                loss.backward()
                optimizer.step()