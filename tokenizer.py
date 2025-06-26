#
# Tokenizer utils for easier way to tokenize stuff around
#

from transformers import AutoTokenizer
from typing import List, Dict, Union, Optional
import torch
import json

def get_tokenizer(name :str = None):
    if name == None :
        return AutoTokenizer.from_pretrained("openai-community/gpt2",trust_remote_code=True)
    return AutoTokenizer.from_pretrained(name,trust_remote_code=True)


def get_tokenizer_properties(tokenizer: AutoTokenizer):
    return {
        'vocab_size': tokenizer.vocab_size,
        'model_max_length': tokenizer.model_max_length,
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token,
        'bos_token': tokenizer.bos_token,
        'unk_token': tokenizer.unk_token,
        'sep_token': tokenizer.sep_token,
        'cls_token': tokenizer.cls_token,
        'mask_token': tokenizer.mask_token,
        'padding_side': tokenizer.padding_side,
        'truncation_side': tokenizer.truncation_side
    }

def to_chat_text(tokenizer,system_message,input,reasoining,output, *, add_generation_prompt=False) -> str:

    assistant_content = ""
    if reasoining:
        assistant_content += f"<thinking>{reasoining}</thinking>"
    assistant_content += output

    messages = [
        {"role": "system",     "content": system_message},
        {"role": "user",       "content": input},
        {"role": "assistant",  "content": assistant_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )

