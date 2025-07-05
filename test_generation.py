from transformers import AutoTokenizer, AutoModelForCausalLM

# Terminal colors for better visibility
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# 0️⃣  Paths ─────────────────────────────────────────────────────────────
ckpt_dir = "HugginfaceReaasoner"        # <— directory saved by model.save_pretrained

# 1️⃣  Load tokenizer & model ────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained( "Qwen/Qwen3-0.6B")
model     = AutoModelForCausalLM.from_pretrained(ckpt_dir)
model.eval()

# 2️⃣  System message (same string you used at fine-tuning) ─────────────
SYS_MSG = """
You are MathReasoner-GPT, a large language model specializing in problem-solving and proof-writing across all branches of mathematics.

• **Goal **Solve each problem correctly and communicate your full line of reasoning.
• **Method **Proceed step-by-step, citing definitions, theorems, and     algebraic manipulations explicitly. Break long derivations into clear, numbered steps.
• **Verification **After your derivation, run a brief “sanity-check” that re-plugs the result or compares limiting cases; mention any discrepancies you find and correct them before giving the final answer.
• **Format **
  1. Restate the problem in your own words (one sentence).  
  2. Solution steps (detailed, numbered).  
  3. Sanity-check (brief).  
  4. Final answer (boxed in LaTeX, e.g. \\boxed{42}).  
• **Style **Use concise prose, LaTeX for mathematical symbols, and avoid unnecessary jargon.
• **Ethics **Do not fabricate theorems or cite non-existent references. If a problem lacks enough information or leads to contradiction, state this explicitly.
• **Limits **If asked for private chain-of-thought not already shown in your Solution steps, politely decline and offer a concise explanation instead.
""".strip()

user_problem = "Compute the integral of x^2 / x^2 + 1 )."

# 4️⃣  Build conversation prompt  (Qwen-style tags) ─────────────────────
dialogue = (
    "<|system|>\n"    + SYS_MSG + 
    "\n<|user|>\n"    + user_problem + 
    "\n<|assistant|>\n" + "<thinking>"
)

inputs = tokenizer(dialogue, return_tensors="pt")

# 5️⃣  Generate answer ──────────────────────────────────────────────────
gen_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,         # tweak as you wish
    top_p=0.95,
    do_sample=True,
)

# 6️⃣  Extract only the newly-generated tokens and decode ───────────────
start = inputs["input_ids"].shape[-1]
reply = tokenizer.decode(gen_ids[0][start:], skip_special_tokens=False)

# 7️⃣  Display results in colorful format ─────────────────────────────────
print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
print(f"{Colors.HEADER}{Colors.BOLD}🧮 MATH REASONER TEST RESULTS{Colors.ENDC}")
print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

print(f"{Colors.OKBLUE}{Colors.BOLD}📝 USER PROMPT:{Colors.ENDC}")
print(f"{Colors.OKCYAN}{user_problem}{Colors.ENDC}\n")

print(f"{Colors.OKGREEN}{Colors.BOLD}🤖 MODEL OUTPUT:{Colors.ENDC}")
print(f"{Colors.OKGREEN}{reply}{Colors.ENDC}\n")

# print(f"{Colors.WARNING}{Colors.BOLD}📊 GENERATION DETAILS:{Colors.ENDC}")
# print(f"{Colors.WARNING}• Temperature: 0.7{Colors.ENDC}")
# print(f"{Colors.WARNING}• Top-p: 0.95{Colors.ENDC}")
# print(f"{Colors.WARNING}• Max new tokens: 256{Colors.ENDC}")
# print(f"{Colors.WARNING}• Model: {ckpt_dir}{Colors.ENDC}")

print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
print(f"{Colors.HEADER}{Colors.BOLD}TEST COMPLETED{Colors.ENDC}")
print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")