from __future__ import annotations
import torch, torch.nn.functional as F
from typing import List, Dict, Callable
from transformers import AutoTokenizer, GenerationConfig
from trainer import Trainer, Config

def get_per_token_logps(logits: torch.Tensor,
                        input_ids: torch.Tensor) -> torch.Tensor:
    log_probs = logits.log_softmax(-1)
    gathered  = torch.gather(log_probs, -1, input_ids.unsqueeze(-1))
    return gathered.squeeze(-1)


class GRPOTrainer(Trainer):
    """
    Group-Relative Policy Optimisation à la DeepSeek.
    The DataLoader must yield dicts containing at least "Q" (prompt) and "A"
    (ground-truth answer string); rewards are supplied via `reward_fn`.
    """

    def __init__(
        self,
        run_name: str,
        cfg: Config,
        tokenizer: AutoTokenizer,
        ref_model: torch.nn.Module,
        reward_fn: Callable[[Dict, str], float],
        num_samples:int= 8,
        beta: float= 0.04,
        clip_eps: float= 0.2,
        generation_cfg: GenerationConfig | None = None,
    ):
        super().__init__(run_name, cfg)

        self.tokenizer   = tokenizer
        self.ref_model   = ref_model.to(self.device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.reward_fn = reward_fn
        self.G = num_samples
        self.beta = beta
        self.clip_eps = clip_eps
        self.gen_cfg = generation_cfg or GenerationConfig(
            max_new_tokens= 512,
            do_sample= True,
            temperature= 0.9,
            num_return_sequences= num_samples,
            pad_token_id= tokenizer.pad_token_id,
        )

    def compute_loss(self, batch: List[Dict]) -> torch.Tensor:
        prompts: List[str] = [item["Q"] for item in batch]

        tip_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            padding_side="left",
        ).to(self.device)

        with torch.no_grad():
            gen_ids = self.model.generate(**tip_inputs, generation_config=self.gen_cfg)

        prompt_len  = tip_inputs["input_ids"].shape[-1]
        completion_ids= gen_ids[:, prompt_len:]
        completions_txt = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        rewards: List[float] = []
        for i, sample in enumerate(batch):
            slice_i = completions_txt[i*self.G:(i+1)*self.G]
            rewards.extend(self.reward_fn(sample, ans) for ans in slice_i)

        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        advantages = rewards.unsqueeze(1)

        rep_prompts = tip_inputs["input_ids"].repeat_interleave(self.G, 0)
        merged_ids = torch.cat([rep_prompts, completion_ids], dim=1)
        compl_mask = (completion_ids != self.tokenizer.pad_token_id).int()

        logits_new = self.model(merged_ids).logits[:, :-1, :]
        logps_new = get_per_token_logps(logits_new, merged_ids[:, 1:])
        logps_new = logps_new[:, prompt_len-1]

        with torch.no_grad():
            logits_ref = self.ref_model(merged_ids).logits[:, :-1, :]
        logps_ref = get_per_token_logps(logits_ref, merged_ids[:, 1:])
        logps_ref = logps_ref[:, prompt_len-1:]

        kl_token = torch.exp(logps_ref - logps_new) - (logps_ref - logps_new) - 1

        pg_token = advantages

        token_loss = -(pg_token - self.beta * kl_token)
        seq_loss = (token_loss * compl_mask).sum(1) / compl_mask.sum(1)
        return seq_loss.mean()
