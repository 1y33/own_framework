import parse_files
import create_dataset
import tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

directory =  "director_random"

texts = parse_files.parse_directory(directory)
word_count = parse_files.count_words(texts)
word_stats = parse_files.print_word_statistics(word_count)

toker = tokenizer.get_tokenizer("Qwen/Qwen2.5-0.5B")
ds = create_dataset.TextDataset(texts=texts,tokenizer=toker)

import model
import trainer
import torch

class LanguageModelTrainer(trainer.Trainer):
    def compute_loss(self, batch) -> torch.Tensor:
        """Compute next token prediction loss for language modeling."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        logits = self.model(input_ids)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        return loss
    
import torch.nn.functional as F
class KDL(trainer.Trainer):
    def __init__(self, run_name: str, cfg: trainer.Config, teacher_model, temperature=2.0, alpha=0.5):
        super().__init__(run_name, cfg)
        self.teacher_model = teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    def compute_loss(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels    = batch["labels"]

        # Student forward
        student_logits = self.model(input_ids)  # (B, L, V)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher_model(input_ids)["logits"]
            
        # print("===" * 10)
        # print(teacher_logits)
        # print(teacher_logits.shape)
        # print(student_logits)
        # print(student_logits.shape)

        shift_s = student_logits[..., :-1, :].contiguous()
        shift_t = teacher_logits[..., :-1, :].contiguous()
        shift_l = labels[..., 1:].contiguous()

        B, L, V = shift_s.shape
        flat_s = shift_s.view(-1, V)       # (B*(L), V)
        flat_t = shift_t.view(-1, V)       # (B*(L), V)
        flat_l = shift_l.view(-1)          # (B*(L),)

        ce = self.ce_loss_fn(flat_s, flat_l)

        log_p_s = F.log_softmax(flat_s / self.temperature, dim=-1)
        p_t     = F.softmax(flat_t     / self.temperature, dim=-1)
        kl = F.kl_div(log_p_s, p_t,
                      reduction="batchmean") * (self.temperature ** 2)

        loss = self.alpha * kl + (1.0 - self.alpha) * ce
        return loss


# import lora

# llm = lora.LoraApplier(llm,"simple",rank=12,device=configuration.device)
# language_trainer = LanguageModelTrainer(run_name="gpt_training_1",cfg=configuration)
# language_trainer.fit()

teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
teacher_model.to("cuda")

llm = model.GPT2(vocab_size=teacher_model.config.vocab_size,n_layers=2,d_model=256)
configuration = trainer.Config(model=llm, train_dataset=ds, epochs=10, batch_size=1, lr=1e-5)

kdl_trainer = KDL(run_name="kdl_training", cfg=configuration, teacher_model=teacher_model)
kdl_trainer.fit()
