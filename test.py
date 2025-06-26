import parse_files
import create_dataset
import tokenizer

directory =  "director_random"

texts = parse_files.parse_directory(directory)
word_count = parse_files.count_words(texts)
word_stats = parse_files.print_word_statistics(word_count)

toker = tokenizer.get_tokenizer()
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

llm = model.GPT2(vocab_size=toker.vocab_size,n_layers=3)
ckpt = torch.load("checkpoints/epoch_24.pt", map_location="cuda",weights_only=False)
llm.load_state_dict(ckpt["model_state"])
llm.to("cuda")

# import lora

configuration = trainer.Config(model=llm, train_dataset=ds, epochs=100, batch_size=1, lr=1e-5)
# llm = lora.LoraApplier(llm,"simple",rank=12,device=configuration.device)

language_trainer = LanguageModelTrainer(run_name="gpt_training_1",cfg=configuration)
language_trainer.fit()