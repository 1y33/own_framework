# dataset_mathx.py
from datasets import load_dataset, Features, Value
import re
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional

BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")

def _strip_boxed(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = BOX_RE.search(s)
    if m:
        return m.group(1).strip()
    # fallback: last number-like span
    m2 = re.findall(r"[-+]?\d*\.?\d+(?:/[1-9]\d*)?", s)
    return m2[-1] if m2 else s.strip()

import re

THINK_OPEN  = "<think>"
THINK_CLOSE = "</think>"

def _bounded(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def thinking_reward(
    text: str,
    tokenizer,
    min_tokens: int = 8,
    max_tokens: int = 128,
    penalize_missing: float = 0.0,   # set >0 to punish when tags are absent
    weight_presence: float = 0.5,
    weight_length: float = 0.5,
    penalize_leak: float = 0.25       # penalty if final answer appears inside think
) -> float:
    """
    Heuristic reward in [0,1] for 'thinking tags' quality:
    + presence of <think>...</think>
    + length inside tags within [min_tokens, max_tokens]
    - penalty if likely final answer appears inside the thinking block (answer leak)
    """
    s = text or ""
    has_open  = THINK_OPEN in s
    has_close = THINK_CLOSE in s
    if not (has_open and has_close):
        return _bounded(1.0 - penalize_missing) if penalize_missing > 0 else 0.0

    # extract first block only
    start = s.index(THINK_OPEN) + len(THINK_OPEN)
    end   = s.index(THINK_CLOSE, start) if THINK_CLOSE in s[start:] else len(s)
    inner = s[start:end].strip()

    # token length score
    n_tok = len(tokenizer.encode(inner, add_special_tokens=False))
    if n_tok <= 0:
        len_score = 0.0
    elif n_tok < min_tokens:
        len_score = n_tok / float(min_tokens)            # ramp up to 1
    elif n_tok > max_tokens:
        len_score = max(0.0, 1.0 - (n_tok - max_tokens) / float(max_tokens))
    else:
        len_score = 1.0

    presence_score = 1.0  # we already checked tags exist

    # simple leakage check: numeric span inside <think>...</think>
    leak = re.search(r"\\boxed\{[^}]*\}", inner) or re.search(r"[-+]?\d*\\?\.?\d+", inner)
    leak_pen = penalize_leak if leak else 0.0

    score = weight_presence * presence_score + weight_length * len_score
    score = _bounded(score - leak_pen, 0.0, 1.0)
    return score

def _unify_row(ex: Dict[str, Any]) -> Dict[str, str]:
    # normalize field names across shards
    problem = ex.get("problem") or ex.get("question") or ex.get("prompt") or ""
    expected = ex.get("expected_answer") or ex.get("expected_answear") or ex.get("answer") or ""
    gen_sol = ex.get("generated_solution") or ex.get("solution") or ex.get("reasoning") or ""
    target = _strip_boxed(expected if expected else gen_sol)
    return {
        "problem": problem,
        "expected_text": target,
        "generated_solution": gen_sol,
    }

class MathXRL(Dataset):
    """
    Materializes up to `max_examples` rows into memory with unified schema,
    ready for your RL trainer.

    Yields each item:
      {
        "prompt_text": str,
        "expected_text": str,
        "input_ids": LongTensor,
        "attention_mask": LongTensor,
        "raw": dict
      }
    """
    def __init__(
        self,
        hf_name: str = "XenArcAI/MathX-5M",
        split: str = "train",
        tokenizer=None,
        max_prompt_len: int = 1024,
        max_examples: int = 500,
        prompt_template: Optional[str] = "Problem: {problem}\nAnswer:",
        streaming_fallback: bool = True,
    ):
        assert tokenizer is not None, "Pass a HF tokenizer"
        self.tok = tokenizer
        self.max_prompt_len = max_prompt_len
        self.tmpl = prompt_template
        self.rows = []

        def _encode_and_append(ex: Dict[str, Any]):
            uni = _unify_row(ex)
            prompt_text = self.tmpl.format(problem=uni["problem"])
            enc = self.tok(
                prompt_text,
                truncation=True,
                max_length=self.max_prompt_len,
                add_special_tokens=True,
                return_tensors="pt",
            )
            self.rows.append({
                "prompt_text": prompt_text,
                "expected_text": uni["expected_text"],
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "raw": ex,
            })

        # try non-streaming small slice first (fast in local dev)
        tried_streaming = False
        ds = load_dataset(hf_name, split=split + "[:1%]")  # small subset
        for i, ex in enumerate(ds):
            _encode_and_append(ex)
            if len(self.rows) >= max_examples:
                break
        

        # If 1% non-streaming worked but you want more, optionally stream the rest
        if not tried_streaming and len(self.rows) < max_examples and streaming_fallback:
            ds_stream = load_dataset(hf_name, split=split, streaming=True)
            for ex in ds_stream:
                _encode_and_append(ex)
                if len(self.rows) >= max_examples:
                    break

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

def rl_collate_fn(batch, pad_token_id: int):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_text": [b["prompt_text"] for b in batch],
        "expected_text": [b["expected_text"] for b in batch],
        "raw": [b["raw"] for b in batch],
    }
# rl_trainer.py
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

def _last_number_like(s: str) -> str:
    m = re.findall(r"[-+]?\d*\.?\d+(?:/[1-9]\d*)?", s)
    return m[-1] if m else s.strip()

def _normalize_target(s: str) -> str:
    # strip boxed; fallback to last number-like
    s = s or ""
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m: return m.group(1).strip()
    return _last_number_like(s)

def _greedy_or_sample(model, tokenizer, input_ids, attention_mask, max_new_tokens, do_sample=True, temperature=1.0):
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Return only the generated continuation (exclude the prompt part)
    cont = gen[:, input_ids.size(1):]
    return cont

def _sum_logprob_on_given_continuation(model, input_ids, attention_mask, cont_ids):
    """
    Teacher-forced log p(y|x): run model on x||y, sum logprobs over y tokens only.
    """
    device = input_ids.device
    full = torch.cat([input_ids, cont_ids], dim=1)
    full_attn = torch.cat([attention_mask, torch.ones_like(cont_ids, dtype=attention_mask.dtype)], dim=1)

    out = model(full, attention_mask=full_attn, use_cache=False)
    logits = out.logits  # [B, T_full, V]
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # predict next
    # labels are full[:,1:], but we only sum over the continuation slice
    labels = full[:, 1:]
    # Build mask that selects the continuation positions
    T_prompt = input_ids.size(1)
    L_full = labels.size(1)
    cont_mask = torch.zeros_like(labels, dtype=torch.bool)
    cont_mask[:, T_prompt-1:] = True  # starting from the first cont token's label position

    # gather logprobs at labels
    lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T_full-1]
    lp = torch.where(cont_mask, lp, torch.zeros_like(lp))
    return lp.sum(dim=1)  # [B]

def _tokenwise_kl_and_entropy(model, ref_model, input_ids, attention_mask):
    with torch.no_grad():
        ref_logits = ref_model(input_ids, attention_mask=attention_mask, use_cache=False).logits
    logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
    p = F.log_softmax(logits, dim=-1).exp()
    q = F.log_softmax(ref_logits, dim=-1).exp()
    # KL(p||q) per position
    kl = (p * (torch.log(p + 1e-9) - torch.log(q + 1e-9))).sum(dim=-1)   # [B, T]
    H  = -(p * torch.log(p + 1e-9)).sum(dim=-1)                          # [B, T]
    # mask on prompt (we'll use prompt+continuation contexts we actually conditioned on)
    mask = attention_mask[:, 1:].float()  # align with logits[:-1]
    kl_mean = (kl[:, :-1] * mask).sum() / (mask.sum() + 1e-9)
    H_mean  = (H[:, :-1]  * mask).sum() / (mask.sum() + 1e-9)
    return kl_mean, H_mean

from trainer import Trainer, Config
class CustomRL(Trainer):
    """
    Your Trainer subclass with a custom RL loss:
    - completion-level reward (per generated answer)
    - PPO-style clip on sequence ratio
    - KL to reference
    - entropy bonus
    Expects extra attributes set on self:
      self.tokenizer (HF tokenizer)
      self.ref_model (HF CausalLM, frozen)
      self.old_model (HF CausalLM, frozen)
      self.K, self.max_new_tokens, self.epsilon_clip, self.kl_weight, self.entropy_coef
    """
    def compute_loss(self, batch) -> torch.Tensor:
        # Unpack batch prepared by rl_collate_fn
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        expected_texts = batch["expected_text"]

        B = input_ids.size(0)
        device = input_ids.device
        tok = self.tokenizer

        # 1) Sample K candidates from π_old
        with torch.no_grad():
            conts = []
            for _ in range(self.K):
                y = _greedy_or_sample(
                    self.old_model, tok,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True, temperature=1.0
                )
                conts.append(y)  # list of [B, Tnew]

        # 2) Rewards per completion (exact-match on normalized targets; adapt as needed)
        # decode candidates and compute reward in [0,1]
# 2) Composite rewards: exact-match + thinking-tags bonus
        targets = [_normalize_target(t) for t in expected_texts]
        
        w_exact  = getattr(self, "w_exact", 1.0)   # or set on trainer
        w_think  = getattr(self, "w_think", 0.2)   # small bonus by default
        
        rewards = []
        for i in range(self.K):
            gen_texts = tok.batch_decode(conts[i], skip_special_tokens=True)
            r_i = []
            for g, tgt in zip(gen_texts, targets):
                # exact-match reward
                pred = _normalize_target(g)
                r_exact = 1.0 if pred == tgt else 0.0
        
                # thinking-tags reward (heuristic in [0,1])
                r_think = thinking_reward(
                    g, tok,
                    min_tokens=8, max_tokens=128,
                    penalize_missing=0.0,         # set >0 to require tags
                    weight_presence=0.5, weight_length=0.5,
                    penalize_leak=0.2
                )
        
                r_total = w_exact * r_exact + w_think * r_think
                # keep the final reward in [0,1] for stability
                r_i.append(min(1.0, max(0.0, r_total)))
            rewards.append(torch.tensor(r_i, device=device, dtype=torch.float))

        # 3) Group baseline (leave-one-out mean)
        A_list = []
        for i in range(self.K):
            others = torch.stack([rewards[j] for j in range(self.K) if j != i], dim=0).mean(dim=0)
            A_list.append(rewards[i] - others)  # [B]
        # stop grad through rewards/baseline
        A_list = [a.detach() for a in A_list]

        # 4) Sequence ratio rho and PPO-style clipped actor term
        actor_terms = []
        for i in range(self.K):
            # log p(y|x) under new and old
            logp_new = _sum_logprob_on_given_continuation(self.model, input_ids, attention_mask, conts[i])
            with torch.no_grad():
                logp_old = _sum_logprob_on_given_continuation(self.old_model, input_ids, attention_mask, conts[i])
            rho = torch.exp(logp_new - logp_old)  # [B]
            Ai  = A_list[i]
            unclipped = rho * Ai
            clipped = torch.clamp(rho, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * Ai
            term = torch.minimum(unclipped, clipped)  # [B]
            actor_terms.append(term)
        J_actor = torch.stack(actor_terms, dim=0).mean()  # scalar

        # 5) KL to reference + entropy (measured on the prompt contexts)
        KL_mean, H_mean = _tokenwise_kl_and_entropy(self.model, self.ref_model, input_ids, attention_mask)

        # 6) Final objective: maximize J = J_actor − λ KL + c_H H  → loss = −J
        J = J_actor - self.kl_weight * KL_mean + self.entropy_coef * H_mean
        loss = -J
        return loss

# train_rl.py (sketch)
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
# Models
student_ckpt  = "Qwen/Qwen2.5-3B"           # policy π_θ (cfg.model)
reference_ckpt= "Qwen/Qwen2.5-3B"           # π_ref (anchor; frozen)
teacher_ckpt  = "Qwen/Qwen2.5-Math-7B"      # optional; not required by this loss
# 'old_model' starts as a copy of student; refresh periodically outside compute_loss

tokenizer = AutoTokenizer.from_pretrained(student_ckpt, use_fast=True)
student   = AutoModelForCausalLM.from_pretrained(student_ckpt, torch_dtype="auto")
ref_model = AutoModelForCausalLM.from_pretrained(reference_ckpt, torch_dtype="auto")
old_model = AutoModelForCausalLM.from_pretrained(student_ckpt, torch_dtype="auto")  # init π_old = π_θ

# Freeze ref & old
ref_model.eval(); 
for p in ref_model.parameters(): p.requires_grad_(False)
old_model.eval();
for p in old_model.parameters(): p.requires_grad_(False)

# Dataset & loader
train_ds = MathXRL(
    split="train",
    tokenizer=tokenizer,
    max_prompt_len=1024,
    max_examples=400,          # start small, scale up
    streaming_fallback=True,   # <- key to avoid cast errors
)
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda b: rl_collate_fn(b, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id),
)

# Your Config & CustomRL
cfg = Config(
    model=student,
    train_dataset=train_ds,
    valid_dataset=None,
    batch_size=2,
    lr=1e-5,
    epochs=1,
    amp=True,
)

trainer = CustomRL(run_name="rl_mathx_demo", cfg=cfg)
# attach extra bits required by compute_loss
trainer.tokenizer = tokenizer
trainer.ref_model = ref_model
trainer.old_model = old_model
trainer.K = 2
trainer.max_new_tokens = 64
trainer.epsilon_clip = 0.2
trainer.kl_weight = 0.05
trainer.entropy_coef = 1e-3

# (Optional) periodically refresh old_model = student.state_dict() copy in a callback
def refresh_old_every_epoch(tr):
    tr.old_model.load_state_dict(tr.model.state_dict())
trainer.cfg.callbacks.append(refresh_old_every_epoch)

# Now trainer.fit() will call CustomRL.compute_loss() per batch
trainer.fit()
