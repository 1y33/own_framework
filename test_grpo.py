# run_grpo.py
import re, random, torch, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from grpo_trainer import GRPOTrainer, Config        # ← the file above
# ──────────────────────────────────────────────────────────────────────────────
# 1.  Reward functions
# ──────────────────────────────────────────────────────────────────────────────
def _extract_last_number(txt: str) -> str | None:
    m = re.findall(r"\d+\.\d+|\d+/\d+|\d+", txt)
    return m[-1] if m else None

def reward_math_correct(sample: dict, completion: str) -> float:
    return 1.0 if _extract_last_number(completion) == sample["A"] else -1.0

def reward_format(sample: dict, completion: str) -> float:
    pat = r"^<think>[\s\S]*?</think><answer>[\s\S]*?</answer>$"
    return 1.0 if re.fullmatch(pat, completion.strip()) else -1.0

def reward_reasoning_depth(sample: dict, completion: str,
                            min_tokens: int = 20) -> float:
    m = re.search(r"<think>(.*?)</think>", completion, flags=re.DOTALL)
    if not m: return -0.25
    return 0.25 if len(m.group(1).split()) >= min_tokens else 0.0

REWARD_FUNCS = [reward_math_correct, reward_format, reward_reasoning_depth]
def total_reward(sample, completion):
    return sum(fn(sample, completion) for fn in REWARD_FUNCS)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Tiny dataset (subset of GSM-8K) – adjust `n_examples` for bigger runs
# ──────────────────────────────────────────────────────────────────────────────
def load_tiny_gsm8k(n_examples: int = 500):
    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{n_examples}]")
    return [{"Q": q, "A": a.split('####')[-1].strip()}
            for q, a in zip(ds["question"], ds["answer"])]

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  main()
# ──────────────────────────────────────────────────────────────────────────────
def main(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    group_size:  int = 4,
    steps:       int = 50,
):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    print("Loading model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    model      = AutoModelForCausalLM.from_pretrained(model_name,
                                                      torch_dtype="auto",
                                                      trust_remote_code=True)
    ref_model  = AutoModelForCausalLM.from_pretrained(model_name,
                                                      torch_dtype="auto",
                                                      trust_remote_code=True)

    train_items = load_tiny_gsm8k(500)
    train_ds    = PromptDataset(train_items)

    gen_cfg = GenerationConfig(
        max_new_tokens       = 512,
        do_sample            = True,
        temperature          = 0.9,
        num_return_sequences = group_size,
        pad_token_id         = tokenizer.pad_token_id,
        eos_token_id         = tokenizer.eos_token_id,
        min_length           = 30,     # ensures the model doesn't quit too early
    )

    cfg = Config(
        model             = model,
        train_dataset     = train_ds,
        batch_size        = 1,     # 1 prompt → group_size completions
        lr                = 5e-6,
        epochs            = 1,     # we’ll loop manually for `steps`
        amp               = True,
        save_every_n_epochs = 0,
        scheduler         = None,
        max_grad_norm     = 1.0,
    )

    trainer = GRPOTrainer(
        run_name      = "demo_grpo",
        cfg           = cfg,
        tokenizer     = tokenizer,
        ref_model     = ref_model,
        reward_fn     = total_reward,
        num_samples   = group_size,
        beta          = 0.01,
        generation_cfg= gen_cfg,
    )

    # Manual training loop (short demo)
    for step in range(steps):
        batch = [train_items[random.randrange(len(train_items))]]
        loss  = trainer.compute_loss(batch)

        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

        if step % 10 == 0:
            print(f"[{step:03d}] loss = {loss.item():.4f}")

    print("Done!")

if __name__ == "__main__":
    main()
