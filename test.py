import parse_files
import create_dataset
import tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

directory =  "director_random"

texts = parse_files.parse_directory(directory)
# word_count = parse_files.count_words(texts)
# word_stats = parse_files.print_word_statistics(word_count)

toker = tokenizer.get_tokenizer("RWKV/RWKV7-Goose-World2.8-0.1B-HF")
ds = create_dataset.TextDataset(texts=texts,tokenizer=toker,stride=1024)
# print(ds[1])
# print(ds[1]["input_ids"])
# print(ds[1]["labels"])
# print(toker.decode(ds[1]["input_ids"][:10]))
# print(toker.decode(ds[1]["labels"]))
# print("==" * 10 )
# print(toker.decode(ds[2]["input_ids"][:10]))
# print(toker.decode(ds[2]["labels"]))
        
        # shift_s = student_logits[..., :-1, :].contiguous()
        # shift_t = teacher_logits[..., :-1, :].contiguous()
        # shift_l = labels[..., 1:].contiguous()

        # B, L, V = shift_s.shape
        # flat_s = shift_s.view(-1, V)       # (B*(L), V)
        # flat_t = shift_t.view(-1, V)       # (B*(L), V)
        # flat_l = shift_l.view(-1)          # (B*(L),)

from models import GPT2
from trainer import Config, PreTrain
import torch


# import lora

# llm = lora.LoraApplier(llm,"simple",rank=12,device=configuration.device)

# teacher_model = AutoModelForCausalLM.from_pretrained("RWKV/RWKV7-Goose-World2.8-0.1B-HF",trust_remote_code=True)
# teacher_model.to("cuda")
llm = GPT2(vocab_size=toker.vocab_size,n_layers=6,d_model=512)
# ckpt = torch.load("gpt_training_1/epoch_19.pt", map_location="cpu",weights_only=False)
# llm.load_state_dict(ckpt["model_state"])
configuration = Config(model=llm, train_dataset=ds, epochs=50, batch_size=1, lr=1e-3,save_dir="gpt_1024stride_50epochs")

# kdl_trainer = KDL(run_name="kdl_training", cfg=configuration, teacher_model=teacher_model)
# kdl_trainer.fit()

language_trainer = PreTrain(run_name="gpt_training_1",cfg=configuration)
language_trainer.fit()
