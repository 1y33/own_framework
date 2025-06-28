import parse_files
import create_dataset
import tokenizer

toker = tokenizer.get_tokenizer("RWKV/RWKV7-Goose-World2.8-0.1B-HF")

import model
import trainer.trainer as trainer
import torch

# class LanguageModelTrainer(trainer.Trainer):
#     def compute_loss(self, batch) -> torch.Tensor:
#         """Compute next token prediction loss for language modeling."""
#         input_ids = batch["input_ids"]
#         labels = batch["labels"]
        
#         logits = self.model(input_ids)
        
#         shift_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
#         shift_labels = labels[..., 1:].contiguous()      # [batch_size, seq_len-1]
        
#         shift_logits = shift_logits.view(-1, shift_logits.size(-1))
#         shift_labels = shift_labels.view(-1)
        
#         loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
#         return loss

# Load model using the static method approach
from transformers import AutoModelForCausalLM
toker = tokenizer.get_tokenizer("RWKV/RWKV7-Goose-World2.8-0.1B-HF")

llm = model.GPT2(vocab_size=toker.vocab_size,n_layers=6,d_model=512)
ckpt = torch.load("gpt_1024stride/epoch_3.pt", map_location="cpu",weights_only=False)
llm.load_state_dict(ckpt["model_state"])
llm.eval()


def generate_text(model,tokenizer,prompt,max_length):
    input_ids  = tokenizer.encode(prompt)
    with torch.inference_mode():
        for _ in range(max_length):
            logits = model(input_ids)
            next_token = logits[0,-1,:]
            
            probs = torch.sofmtax(next_token,dim=-1)
            next_token = torch.multinomial(probs,num_samples=1)
            input_ids = torch.cat([input_ids,next_token])
            
            
    generated_text = tokenizer.decode(input_ids)
    return generate_text

# Text generation function
def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50):
    """
    Generate text using the trained model
    
    Args:
        model: Trained GPT2 model
        tokenizer: Tokenizer used for training
        prompt: Starting text prompt
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only consider top k tokens for sampling
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            
            next_token_logits = logits[0, -1, :] / temperature
            
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

prompt = "Once upon a time"
generated = generate_text(llm, toker, prompt, max_length=50, temperature=0.3)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")

prompts = [
    "The fourier of the transform x(t) = 1(t) is ",
    "Machine Learning is ",
    "GPU code",
]

for prompt in prompts:
    generated = generate_text(llm, toker, prompt, max_length=30, temperature=0.3)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")