from transformers import AutoTokenizer, AutoModelForCausalLM

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

ckpt_dir = "model"

tokenizer = AutoTokenizer.from_pretrained("nvidia/AceReason-Nemotron-7B",device_map="cuda")
model = AutoModelForCausalLM.from_pretrained(ckpt_dir,device_map="cuda")
model.eval()

SYS_MSG = """
You are a helpfull assitant that can reason about math problems.
""".strip()

user_problem = "Solve what you want XD"

dialogue = (
    "<|system|>\n"    + SYS_MSG + 
    "\n<|user|>\n"    + user_problem + 
    "\n<|assistant|>\n" + "<thinking>"
)

inputs = tokenizer(dialogue, return_tensors="pt").to("cuda")

gen_ids = model.generate(
    **inputs,
    max_new_tokens=4096, 
    temperature=0.7,         
    top_p=0.95,
    do_sample=True, 
)

start = inputs["input_ids"].shape[-1]
reply = tokenizer.decode(gen_ids[0][start:], skip_special_tokens=False)

print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
print(f"{Colors.HEADER}{Colors.BOLD}🧮 MATH REASONER TEST RESULTS{Colors.ENDC}")
print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

print(f"{Colors.OKBLUE}{Colors.BOLD}📝 USER PROMPT:{Colors.ENDC}")
print(f"{Colors.OKCYAN}{user_problem}{Colors.ENDC}\n")

print(f"{Colors.OKGREEN}{Colors.BOLD}🤖 MODEL OUTPUT:{Colors.ENDC}")
print(f"{Colors.OKGREEN}{reply}{Colors.ENDC}\n")


print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
print(f"{Colors.HEADER}{Colors.BOLD}TEST COMPLETED{Colors.ENDC}")
print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")