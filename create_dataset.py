import torch
from torch.utils.data import Dataset
from typing import Iterable, List, Sequence
from tokenizer import to_chat_text
from typing import Sequence, List, Optional, Union, Dict
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

            for start in range(0, len(ids), stride):
                window = ids[start : start + seq_len]
                if len(window) == 0:
                    break
                if len(window) < seq_len:
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    window += [pad_token_id] * (seq_len - len(window))
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
        
from torch.utils.data import IterableDataset, get_worker_info
from datasets import IterableDataset as HFIterableDataset 

class StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        hf_ds_iter: "HFIterableDataset",
        tokenizer,
        *,
        seq_len: int = 1024,
        stride: int = 1,
        add_eos: bool = True,
        text_column: str = "text",
    ):
        self.dataset   = hf_ds_iter
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.stride    = stride
        self.add_eos   = add_eos
        self.text_col  = text_column

    def _tokenise_and_yield(self, text: str):
        if self.add_eos and self.tokenizer.eos_token:
            text += self.tokenizer.eos_token
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        for start in range(0, len(ids) - self.seq_len + 1, self.stride):
            window = ids[start : start + self.seq_len]
            if len(window) == self.seq_len:
                tensor = torch.tensor(window, dtype=torch.long)
                yield {"input_ids": tensor, "labels": tensor.clone()}
    # main iterator -----------------------------------------
    def __iter__(self):
        worker = get_worker_info()
        if worker is not None:
            shard_ds = self.dataset.shard(
                num_shards = worker.num_workers,
                index      = worker.id,
                contiguous = True,
            )
        else:
            shard_ds = self.dataset


        for row in shard_ds:
            yield from self._tokenise_and_yield(row[self.text_col])

class StreamingSFTDataset(IterableDataset):

    def __init__(
        self,hf_ds_iter: "HFIterableDataset",tokenizer,system_message: str,
        *,
        input_column: str = "inputs",output_column: str = "outputs",reasoning_column: Optional[str] = None,seq_len: int = 1024,stride : int = 1,add_eos: bool = False,
    ):
        self.dataset = hf_ds_iter
        self.tokenizer = tokenizer
        self.sys_msg = system_message
        self.in_col = input_column
        self.out_col = output_column
        self.reason_col = reasoning_column
        self.seq_len = seq_len
        self.stride = stride
        self.add_eos = add_eos

    from tokenizer import to_chat_text
    def _fmt(role, content): return {"role": role, "content": content}

    def _create_prompt(self, user, assistant, reasoning=None):
        msgs = [self._fmt("system", self.sys_msg)]
        if reasoning is not None:
            reasoning = [reasoning] if isinstance(reasoning, str) else reasoning
        if isinstance(user, str):      user      = [user]
        if isinstance(assistant, str): assistant = [assistant]
        if len(user) != len(assistant):
            raise ValueError("input/output length mismatch")

        for i in range(len(user)):
            msgs.append(self._fmt("user", user[i]))
            if reasoning is not None:
                msgs.append(self._fmt("assistant", f"<thinking>{reasoning[i]}</thinking>"))
            msgs.append(self._fmt("assistant", assistant[i]))
        return msgs

    # ──────────────────────────────────────────────────────────
    def _render_and_tokenise(self, row: Dict):
        # Format chat → plain string using HF chat template
        rendered = self.tokenizer.apply_chat_template(
            self._create_prompt(
                row[self.in_col],
                row[self.out_col],
                row.get(self.reason_col) if self.reason_col else None,
            ),
            tokenize=False,
        )
        if self.add_eos and self.tokenizer.eos_token:
            rendered += self.tokenizer.eos_token

        ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        for start in range(0, len(ids) - self.seq_len + 1, self.stride):
            window = ids[start : start + self.seq_len]
            if len(window) == self.seq_len:
                t = torch.tensor(window, dtype=torch.long)
                yield {"input_ids": t, "labels": t.clone()}

    def __iter__(self):
        worker = get_worker_info()
        ds = self.dataset.shard(worker.num_workers, worker.id, True) if worker else self.dataset
        for row in ds:
            yield from self._render_and_tokenise(row)
