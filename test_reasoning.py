from datasets import load_dataset
import create_dataset

import os
import tokenizer
from models import GPT2
from trainer import PreTrain,Config
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


    
SYS_MSG = """
You are MathReasoner-GPT, a large language model specializing in problem-solving and proof-writing across all branches of mathematics.

• **Goal **Solve each problem correctly and communicate your full line of reasoning.
• **Method **Proceed step-by-step, citing definitions, theorems, and algebraic manipulations explicitly. Break long derivations into clear, numbered steps.
• **Verification **After your derivation, run a brief “sanity-check” that re-plugs the result or compares limiting cases; mention any discrepancies you find and correct them before giving the final answer.
• **Format **
  1. Restate the problem in your own words (one sentence).
  2. Solution steps (detailed, numbered).
  3. Sanity-check (brief).
  4. Final answer (boxed in LaTeX, e.g. \boxed{42}).
• **Style **Use concise prose, LaTeX for mathematical symbols, and avoid unnecessary jargon.
• **Ethics **Do not fabricate theorems or cite non-existent references. If a problem lacks enough information or leads to contradiction, state this explicitly.
• **Limits **If asked for private chain-of-thought not already shown in your Solution steps, politely decline and offer a concise explanation instead.
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"

hugginface_ds = load_dataset("notbadai/math_reasoning", split="train")
hugginface_ds = hugginface_ds.select(range(10000))  



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")


convs = create_dataset.map_dataset_to_conversation(
    hugginface_ds,
    input_collumn="prompt",
    output_collumn="answer",
    reasoning_collumn="reasoning"
)
sft_ds = create_dataset.SFTDataset(
    conversations=convs,
    system_message=SYS_MSG,
    tokenizer=tokenizer,
    seq_len=1024,
    stride=1024
)
configuration = Config(model=model,train_dataset=sft_ds, epochs=10)   #batch_size=16,lr=5e-5,save_dir="ReasoningMathQwen")
sft_trainer = PreTrain(run_name="ReasoningMathQwen",cfg=configuration)
sft_trainer.fit()




model.save_pretrained("HugginfaceReaasoner")

