from __future__ import annotations
import os
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional, List

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import logger

@dataclass
class Config:
    model: nn.Module
    train_dataset: torch.utils.data.Dataset
    valid_dataset: Optional[torch.utils.data.Dataset] = None

    batch_size: int = 32
    lr: float = 3e-4
    epochs: int = 10
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True

    optimizer: str = "adamw"  # "adam" | "adamw" | "sgd"
    weight_decay: float = 0.01
    scheduler: Optional[str] = "cosine"  # "cosine" | "step" | None
    warmup_steps: int = 0

    amp: bool = True  # Automatic mixed precision
    compile: bool = False  # torch.compile
    max_grad_norm: Optional[float] = 1.0
    mixed_precision_dtype: torch.dtype = torch.float16

    save_dir: str = "./checkpoints"
    save_every_n_epochs: int = 1
    max_saves: int = 3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    verbose: bool = True
    callbacks: List[Callable[["Trainer"], None]] = field(default_factory=list)


class Trainer:
    """Fast yet readable training loop.

    Override :py:meth:`compute_loss` for custom objectives or plug in custom
    behaviour via ``Config.callbacks`` – each callback receives the Trainer
    instance at the end of every epoch, giving full read/write access.
    """

    def __init__(self,run_name:str,cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = True

        self.device = torch.device(cfg.device)
        self.model = cfg.model.to(self.device)
       
        if cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        
        self.scaler = torch.amp.GradScaler(device="cuda" if cfg.amp is True else None)

        self.train_loader = self._init_dataloader(cfg.train_dataset, train=True)
        self.valid_loader = self._init_dataloader(cfg.valid_dataset, train=False) if cfg.valid_dataset else None

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        
        self.logger = logger.create_logger(run_name)
        self.logger.log_model_info(self.model,batch_size = cfg.batch_size,optimizer = cfg.optimizer, epochs =cfg.epochs)

    def _init_dataloader(self, dataset, train: bool) -> DataLoader:
        if dataset is None:
            return None
        
        return DataLoader(
            dataset,
            batch_size=         self.cfg.batch_size,
            shuffle=            train and self.cfg.shuffle,
            num_workers=        self.cfg.num_workers,
            pin_memory=         self.cfg.pin_memory,
            persistent_workers= self.cfg.num_workers > 0,
        )

    def _init_optimizer(self) -> optim.Optimizer:
        name = self.cfg.optimizer.lower()
        if name == "adam":  return optim.Adam(self.model.parameters(),  lr=self.cfg.lr)
        if name == "adamw": return optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if name == "sgd":   return optim.SGD(self.model.parameters(),   lr=self.cfg.lr, momentum=0.9)
        raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}")

    def _init_scheduler(self) -> Optional[_LRScheduler]:
        if self.cfg.scheduler is None:
            return None
        
        name = self.cfg.scheduler.lower()
        if name == "cosine": return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs)
        if name == "step":   return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        raise ValueError(f"Unknown scheduler: {self.cfg.scheduler}")

    def compute_loss(self, batch) -> torch.Tensor:
        inputs, labels = batch
        logits = self.model(inputs)
        return nn.functional.cross_entropy(logits, labels)

    def train_step(self, batch) -> torch.Tensor:
        with torch.amp.autocast("cuda",enabled=self.cfg.amp and self.device.type == "cuda", dtype=self.cfg.mixed_precision_dtype):
            loss = self.compute_loss(batch)

        self.scaler.scale(loss).backward()

        if self.cfg.max_grad_norm:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.global_step += 1
        return loss.detach()

    @torch.no_grad()
    def valid_step(self, batch):
        self.model.eval()
        with torch.amp.autocast("cuda",enabled=self.cfg.amp and self.device.type == "cuda",
                                     dtype=self.cfg.mixed_precision_dtype):
            return self.compute_loss(batch)

    def fit(self):
        for epoch in range(1, self.cfg.epochs + 1):
            epoch_loss = 0.0
            
            self.model.train()
            for batch in self.train_loader:
                batch = self._move_to_device(batch)
                loss = self.train_step(batch)
                epoch_loss += loss.item()
                self.logger.log_live_epoch(epoch,loss)

            self.model.eval()
            epoch_loss /= len(self.train_loader)
            val_loss = self.validate() if self.valid_loader else None

            if val_loss == None:
                self.logger.log_epoch(epoch,epoch_loss,perplexity=math.exp(epoch_loss/len(self.train_loader)))
            else:
                self.logger.log_epoch(epoch,epoch_loss,perplexity=math.exp(epoch_loss/len(self.train_loader)),Val_Loss = val_loss)

            if self.scheduler:
                self.scheduler.step()

            if epoch % self.cfg.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")
                self._prune_checkpoints(self.cfg.max_saves)

            for cb in self.cfg.callbacks:
                cb(self)

    def validate(self) -> float:
        if self.valid_loader is None:
            return math.nan
        total_loss = 0.0
        for batch in self.valid_loader:
            batch = self._move_to_device(batch)
            total_loss += self.valid_step(batch).item()
        return total_loss / len(self.valid_loader)

    def _move_to_device(self, batch):
        """Recursively move tensors to the configured device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=self.cfg.pin_memory)
        
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(t) for t in batch)
        
        if isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        
        return batch

    def save_checkpoint(self, name: str = "latest.pt"):
        path = Path(self.cfg.save_dir) / name
        torch.save(
            {
                "model_state":      self.model.state_dict(),
                "optimizer_state":  self.optimizer.state_dict(),
                "scaler_state":     self.scaler.state_dict(),
                "cfg":              asdict(self.cfg),
                "global_step":      self.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step = ckpt.get("global_step", 0)

    @staticmethod
    def load_model(model: nn.Module, checkpoint_path: str, device: str = "cpu"):
        """Load only the model weights from a checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return model
    
    def _prune_checkpoints(self, max_keep: int):
        files = sorted(Path(self.cfg.save_dir).glob("epoch_*.pt"), key=os.path.getmtime)
        while len(files) > max_keep:
            oldest = files.pop(0)
            oldest.unlink(missing_ok=True)


# if __name__ == "__main__":
#     from torchvision.models import resnet18
#     from torchvision.datasets import CIFAR10
#     from torchvision import transforms

#     tfm = transforms.Compose([transforms.ToTensor()])
#     train_set = CIFAR10(root="./data", train=True, download=True, transform=tfm)
#     val_set = CIFAR10(root="./data", train=False, download=True, transform=tfm)

#     model = resnet18(num_classes=10)
#     cfg = Config(model=model, train_dataset=train_set, valid_dataset=val_set,epochs=5, batch_size=128, compile=True)

#     trainer = Trainer(cfg)
#     trainer.fit()
