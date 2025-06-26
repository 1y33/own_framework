import logging
import torch
import datetime
import os
import shutil
from typing import Dict, Any, Optional

class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'

class TrainingLogger:
    def __init__(self, run_name: str = "training_run", log_level: int = logging.INFO):
        self.run_name = run_name
        self.start_time = datetime.datetime.now()
        
        self.terminal_width = self._get_terminal_width()
        
        self.logger = logging.getLogger(f"training_{run_name}")
        self.logger.setLevel(log_level)
        
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        
        self._print_header()
    
    def _get_terminal_width(self) -> int:
        """Get terminal width, default to 80 if unable to determine"""
        try:
            return shutil.get_terminal_size().columns
        except:
            return 80
    
    def _make_separator(self, char: str = "=", color: str = Colors.CYAN) -> str:
        """Create a full-width separator"""
        return f"{color}{char * self.terminal_width}{Colors.RESET}"
    
    def _make_centered_text(self, text: str, color: str = Colors.WHITE, bg_color: str = "") -> str:
        """Center text in terminal width"""
        padding = (self.terminal_width - len(text)) // 2
        return f"{bg_color}{color}{' ' * padding}{text}{' ' * (self.terminal_width - len(text) - padding)}{Colors.RESET}"
    
    def _print_header(self):
        """Print training run header information"""
        if torch.cuda.is_available():
            device_info = f"GPU ({torch.cuda.get_device_name()})"
            device_count = torch.cuda.device_count()
            if device_count > 1:
                device_info += f" x{device_count}"
            device_color = Colors.GREEN
        else:
            device_info = "CPU"
            device_color = Colors.YELLOW
        
        date_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        
        self.logger.info(self._make_separator("=", Colors.CYAN))
        self.logger.info(self._make_centered_text("🚀 TRAINING RUN STARTED 🚀", Colors.BOLD + Colors.WHITE, Colors.BG_BLUE))
        self.logger.info(self._make_separator("=", Colors.CYAN))
        
        info_line = f"{Colors.BOLD}Run:{Colors.RESET} {Colors.MAGENTA}{self.run_name}{Colors.RESET} | {Colors.BOLD}Device:{Colors.RESET} {device_color}{device_info}{Colors.RESET} | {Colors.BOLD}Date:{Colors.RESET} {Colors.CYAN}{date_str}{Colors.RESET}"
        
        info_padding = (self.terminal_width - len(self.run_name) - len(device_info) - len(date_str) - 20) // 2  # Approximate
        self.logger.info(f"{' ' * max(0, info_padding)}{info_line}")
        self.logger.info(self._make_separator("=", Colors.CYAN))
    
    def log_model_info(self, model: torch.nn.Module, **kwargs):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        dtype = next(model.parameters()).dtype
        
        model_info = f"{Colors.BOLD}Model Info:{Colors.RESET} {Colors.GREEN}{total_params:,}{Colors.RESET} params ({Colors.YELLOW}{trainable_params:,}{Colors.RESET} trainable) | {Colors.BOLD}dtype:{Colors.RESET} {Colors.CYAN}{dtype}{Colors.RESET}"
        
        if kwargs:
            extra_info = " | ".join([f"{Colors.BOLD}{k}:{Colors.RESET} {Colors.WHITE}{v}{Colors.RESET}"  for k, v in kwargs.items()])
            model_info += f" | {extra_info}"
        
        self.logger.info(model_info)
        self.logger.info(self._make_separator("-", Colors.GRAY))
    
    def log_epoch(self, epoch: int, loss: float, **metrics):
        """Log epoch information with colors"""
        loss_color = Colors.RED if loss > 5.0 else Colors.YELLOW if loss > 2.0 else Colors.GREEN
        loss_str = f"{Colors.BOLD}Loss:{Colors.RESET} {loss_color}{loss:.6f}{Colors.RESET}"
        
        epoch_str = f"{Colors.BOLD + Colors.BLUE}Epoch {epoch:3d}{Colors.RESET}"
        
        if metrics:
            metrics_parts = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    metrics_parts.append(f"{Colors.BOLD}{k}:{Colors.RESET} {Colors.CYAN}{v:.6f}{Colors.RESET}")
                else:
                    metrics_parts.append(f"{Colors.BOLD}{k}:{Colors.RESET} {Colors.WHITE}{v}{Colors.RESET}")
            
            metrics_str = " | ".join(metrics_parts)
            log_message = f"\n{epoch_str} | {loss_str} | {metrics_str}"
        else:
            log_message = f"\n{epoch_str} | {loss_str}"
        
        self.logger.info(log_message)
    
    def format_line(self,epoch:int,loss:float,**metrics):
        """Return the coloured one-line string used by log_epoch()."""
        loss_color = Colors.RED if loss > 5 else Colors.YELLOW if loss > 2 else Colors.GREEN
        
        epoch_part = f"{Colors.BOLD + Colors.BLUE}Epoch {epoch:3d}{Colors.RESET}"
       
        loss_part  = f"{Colors.BOLD}Loss:{Colors.RESET} {loss_color}{loss:.6f}{Colors.RESET}"

        extras = []
        for k, v in metrics.items():
            val = f"{v:.6f}" if isinstance(v, float) else str(v)
            extras.append(f"{Colors.BOLD}{k}:{Colors.RESET} {Colors.CYAN}{val}{Colors.RESET}")
        extra_part = " | ".join(extras)

        return f"{epoch_part} | {loss_part}" + (f" | {extra_part}" if extras else "")

    def log_live_epoch(self,epoch:int, loss:float, **metrics):
        line = self.format_line(epoch,loss,**metrics)
        padded = line.ljust(self.terminal_width)
        self.logger.handlers[0].stream.write(f"\r{padded}")
        self.logger.handlers[0].stream.flush()
    
    def log_info(self, message: str):
        """Log general information"""
        self.logger.info(f"{Colors.WHITE}{message}{Colors.RESET}")
    
    def log_warning(self, message: str):
        """Log warning with color"""
        self.logger.warning(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")
    
    def log_error(self, message: str):
        """Log error with color"""
        self.logger.error(f"{Colors.RED}❌ {message}{Colors.RESET}")
    
    def log_success(self, message: str):
        """Log success message with color"""
        self.logger.info(f"{Colors.GREEN}✅ {message}{Colors.RESET}")
    
    def log_training_complete(self, total_epochs: int, final_loss: Optional[float] = None):
        """Log training completion with colors"""
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info(self._make_separator("=", Colors.GREEN))
        self.logger.info(self._make_centered_text("🎉 TRAINING COMPLETED 🎉", Colors.BOLD + Colors.WHITE, Colors.BG_GREEN))
        
        completion_info = f"{Colors.BOLD}Total Epochs:{Colors.RESET} {Colors.CYAN}{total_epochs}{Colors.RESET} | {Colors.BOLD}Duration:{Colors.RESET} {Colors.MAGENTA}{duration}{Colors.RESET}"
        if final_loss:
            loss_color = Colors.GREEN if final_loss < 2.0 else Colors.YELLOW if final_loss < 5.0 else Colors.RED
            completion_info += f" | {Colors.BOLD}Final Loss:{Colors.RESET} {loss_color}{final_loss:.6f}{Colors.RESET}"
        
        padding = (self.terminal_width - len(str(total_epochs)) - len(str(duration)) - 30) // 2  # Approximate
        self.logger.info(f"{' ' * max(0, padding)}{completion_info}")
        self.logger.info(self._make_separator("=", Colors.GREEN))

def create_logger(run_name: str = "training_run") -> TrainingLogger:
    """Create a training logger"""
    return TrainingLogger(run_name)

# Example usage
if __name__ == "__main__":
    logger = create_logger("gpt2_experiment")
    
    import torch.nn as nn
    model = nn.Linear(100, 50)
    logger.log_model_info(model, batch_size=32, lr=1e-4)
    
    import time
    import random
    for epoch in range(1, 4):
        for batch in range(1,100):
            time.sleep(0.1)
            loss = 2.5 - epoch * 0.3 * 0.2 * random.randint(-1,1) 
            logger.log_live_epoch(epoch,loss)
        logger.log_epoch(epoch, loss, accuracy=0.85 + epoch * 0.02, lr=1e-4)
    
    logger.log_info("This is an info message")
    logger.log_success("Training step completed successfully")
    logger.log_warning("Learning rate might be too high")
    
    logger.log_training_complete(3, final_loss=1.6)