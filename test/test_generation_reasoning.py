import torch
import tokenizer
import models.GPT as model
from create_dataset import create_prompt_message, tokenize_message

# System message used during training
SYS_MSG = """
You are MathReasoner-GPT, a large language model specializing in problem-solving and proof-writing across all branches of mathematics.

• **Goal **Solve each problem correctly and communicate your full line of reasoning.
• **Method **Proceed step-by-step, citing definitions, theorems, and algebraic manipulations explicitly. Break long derivations into clear, numbered steps.
• **Verification **After your derivation, run a brief "sanity-check" that re-plugs the result or compares limiting cases; mention any discrepancies you find and correct them before giving the final answer.
• **Format **
  1. Restate the problem in your own words (one sentence).
  2. Solution steps (detailed, numbered).
  3. Sanity-check (brief).
  4. Final answer (boxed in LaTeX, e.g. \boxed{42}).
• **Style **Use concise prose, LaTeX for mathematical symbols, and avoid unnecessary jargon.
• **Ethics **Do not fabricate theorems or cite non-existent references. If a problem lacks enough information or leads to contradiction, state this explicitly.
• **Limits **If asked for private chain-of-thought not already shown in your Solution steps, politely decline and offer a concise explanation instead.
"""

# Load tokenizer and model
toker = tokenizer.get_tokenizer("RWKV/RWKV7-Goose-World2.8-0.1B-HF")

llm = model.GPT2(vocab_size=toker.vocab_size, n_layers=6, d_model=512, max_seq_len=1024)
ckpt = torch.load("gpt_reasoning_please/epoch_9.pt", map_location="cuda", weights_only=False)
llm.load_state_dict(ckpt["model_state"])
llm.eval()

def generate_reasoning_response(model, tokenizer, system_message, user_prompt, max_length=300, temperature=0.7, top_k=50):
    """
    Generate a reasoning response using the same format as the SFT dataset
    """
    # Create the conversation in the same format as training
    messages = create_prompt_message(
        system_message=system_message,
        input_message=user_prompt,
        output_message="",  # Empty since we're generating this
        reasoning_messages=None
    )[:-1]  # Remove the empty assistant message, we'll generate it
    
    # Convert to text format
    prompt_text = tokenize_message(tokenizer, messages)
    
    # Add the assistant prefix to signal generation start
    # Note: Adjust this based on your tokenizer's chat format
    if hasattr(tokenizer, 'apply_chat_template'):
        # If using chat template, add assistant start
        assistant_msg = [{"role": "assistant", "content": ""}]
        assistant_start = tokenizer.apply_chat_template(assistant_msg, tokenize=False, add_generation_prompt=True)
        prompt_text += assistant_start.split("assistant")[-1].strip()
    else:
        prompt_text += "\n\nAssistant: "
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    
    with torch.no_grad():
        generated_ids = input_ids.clone()
        
        for _ in range(max_length):
            logits = model(generated_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for end tokens
            if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the full response
    full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "Assistant:" in full_response:
        assistant_response = full_response.split("Assistant:")[-1].strip()
    else:
        # Fallback: take everything after the original prompt
        original_prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        assistant_response = full_response.replace(original_prompt_text, "").strip()
    
    return assistant_response

# Test reasoning problems
math_problems = [
    "Solve for x: 2x + 5 = 13",
    "Find the derivative of f(x) = x³ + 2x² - 4x + 1",
    "What is the sum of the first 10 positive integers?",
    "Prove that the square root of 2 is irrational",
    "Solve the quadratic equation: x² - 5x + 6 = 0",
    "Find the integral of ∫(3x² + 2x - 1)dx",
    "What is the limit of (sin x)/x as x approaches 0?"
]

print("Testing Math Reasoning Model")
print("=" * 60)

for i, problem in enumerate(math_problems, 1):
    print(f"\n[Problem {i}]")
    print(f"Question: {problem}")
    print("-" * 40)
    
    response = generate_reasoning_response(
        model=llm,
        tokenizer=toker,
        system_message=SYS_MSG,
        user_prompt=problem,
        max_length=200,
        temperature=0.7,
        top_k=40
    )
    
    print(f"Response:\n{response}")
    print("=" * 60)

# Test with different temperatures for creativity vs accuracy
print("\n\nTesting with different temperatures:")
test_problem = "Find the area of a circle with radius 5"

for temp in [0.3, 0.7, 1.0]:
    print(f"\nTemperature: {temp}")
    print("-" * 30)
    response = generate_reasoning_response(
        model=llm,
        tokenizer=toker,
        system_message=SYS_MSG,
        user_prompt=test_problem,
        max_length=150,
        temperature=temp,
        top_k=50
    )
    print(f"Response: {response}")