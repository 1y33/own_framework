import torch
from torch.utils.data import Dataset
from typing import Iterable, List, Sequence
from tokenizer import to_chat_text
from typing import List, Union, Optional
import time
from tqdm import tqdm



class TextDataset(Dataset):
    '''
    This dataset is used like this : 
    texts : its a list containing all the text files you have 
    tokenizer : tokenizer
    
    what it does :
    it iterates over text files
    for each iteration its creating a sliding window :
    from 0 to len(ids)=len(tokens in the text)-stride. with the step of stride:
    [0 , stride] [stride + 1 , stride * 2 ] ....
    '''
   
    def __init__(self, texts, tokenizer, *, seq_len=1024, stride=1, add_eos=True):
        samples = []
        for t in texts:
            if add_eos and tokenizer.eos_token:
                t += tokenizer.eos_token
            ids = tokenizer(t, add_special_tokens=False)["input_ids"]

            for start in range(0, len(ids) - stride, stride):
                window = ids[start : start + seq_len]
                if len(window) < seq_len:
                    break                           
                samples.append(torch.tensor(window, dtype=torch.long))

        self.input_ids = torch.stack(samples)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        x = self.input_ids[idx]
        return {"input_ids": x, "labels": x.clone()}
    

# The text dataset now needs to be done for this:
# SFT - prompt message , actual conversation

def _formatted_message(role: str, content: str):
    return {
        "role": role,
        "content": content
    }

def create_prompt_message(
    system_message: str,
    input_message: Union[List[str], str],
    output_message: Union[List[str], str],
    reasoning_messages: Optional[Union[List[str], str]] = None
):
    messages = []

    messages.append(_formatted_message("system", system_message))

    if isinstance(input_message, str):
        input_message = [input_message]

    if isinstance(output_message, str):
        output_message = [output_message]

    if reasoning_messages is not None:
        if isinstance(reasoning_messages, str):
            reasoning_messages = [reasoning_messages]
        
        if len(reasoning_messages) != len(input_message):
            raise ValueError("reasoning_messages must have the same length as input_message.")
    
    if len(input_message) != len(output_message):
        raise ValueError("input_message and output_message must have the same length.")

    for i in range(len(input_message)):
        messages.append(_formatted_message("user", input_message[i]))
        if reasoning_messages is not None:
            messages.append(_formatted_message("assistant", f"<thinking>{reasoning_messages[i]}</thinking>"))
        messages.append(_formatted_message("assistant", output_message[i]))

    return messages

def tokenize_message(tokenizer,message):
    '''
    Return the message with the tokenization of the model
    '''    
    return tokenizer.apply_chat_template(message,tokenize=False)

def map_dataset_to_conversation(hugginface_datatest,
                                input_collumn,
                                output_collumn,
                                reasoning_collumn=False)->List:
    
    def row_to_conv(example):
        conv = {
            "inputs" : example[input_collumn],
            "outputs" : example[output_collumn]
        }
        if reasoning_collumn and reasoning_collumn in example and example[reasoning_collumn] is not None:
            conv["reasoning"] = example[reasoning_collumn]

        return conv
    conversations = [row_to_conv(row) for row in hugginface_datatest]
    return conversations

class SFTDataset(TextDataset):
    '''
    This Dataset is used so that it creates a TextDataset from a collection of conversation
    This is used for SFT(Instruction following etc). This is cool 
    
    Also need to have the conversation in a specific way.
    Use the other tool to format the conversation
    '''
    def __init__(
        self,
        conversations: Sequence[dict],
        tokenizer,
        system_message,
        *,
        seq_len: int = 1024,
        stride: int = 1,
        add_eos: bool = False,            
    ):
        rendered_texts: List[str] = []
        start_time = time.time()
        for conv in tqdm(conversations, desc="Rendering conversations"):
            rendered_texts.append(
            tokenize_message(tokenizer, create_prompt_message(
                system_message=system_message,
                input_message=conv["inputs"],
                output_message=conv["outputs"],
                reasoning_messages=conv.get("reasoning"),))
            )
        elapsed_time = time.time() - start_time
        print(f"Rendered {len(rendered_texts)} conversations in {elapsed_time:.2f} seconds")

        super().__init__(
            texts=rendered_texts,
            tokenizer=tokenizer,
            seq_len=seq_len,
            stride=stride,
            add_eos=add_eos,
        )