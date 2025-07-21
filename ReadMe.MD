# LLM Training Framework

A flexible framework for training, fine-tuning, and evaluating large language models (LLMs) on a variety of datasets, supporting both HuggingFace models and custom PyTorch models.

---

## Features

- **Model Agnostic:** Use HuggingFace models or your own PyTorch models (e.g., GPT2, DeepSeek).
- **Dataset Flexibility:** Train on HuggingFace datasets, plain text, code, or your own data.
- **Custom Trainers:** Easily implement new training objectives by overriding a single method.
- **Multi-GPU Training:** Built-in support for distributed (DDP) training.
- **Rich Logging:** Colorful terminal logs and live loss visualization.

---

## Getting Started

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Prepare Data

- **Text/Code:** Place your files in a directory and use the provided parsing utilities.
- **HuggingFace Datasets:** Load any dataset using the `datasets` library.

### 3. Tokenization

- Use any HuggingFace tokenizer or your own.
- Example:
  ```python
  from tokenizer import get_tokenizer
  tokenizer = get_tokenizer("RWKV/RWKV7-Goose-World2.8-0.1B-HF")
  ```

### 4. Dataset Creation

- **Plain Text:** Use `TextDataset` to create a dataset from your files.
- **Instruction/Reasoning:** Use `SFTDataset` for chat-style or reasoning data.
- **Streaming:** Use `StreamingTextDataset` or `StreamingSFTDataset` for large datasets.

### 5. Model Selection

- **HuggingFace:** Load any model with `AutoModelForCausalLM`.
- **Custom:** Use provided models like `GPT2` ([`models.GPT2`](models/GPT.py)) or implement your own.

### 6. Training

- **Configuration:** Set up a [`Config`](trainer/trainer.py) object with your model, dataset, and hyperparameters.
- **Trainer:** Use a built-in trainer (e.g., `PreTrain`, `KDL`, `GRPOTrainer`) or create your own.
- **Run Training:**
  ```python
  from trainer import PreTrain, Config
  trainer = PreTrain(run_name="my_run", cfg=configuration)
  trainer.fit()
  ```

### 7. Multi-GPU Training

- Enable DDP by passing `gpu_ids` in your config and launching with `torch.multiprocessing`.
- See the DDP pattern in the codebase for launching multi-process training.

---

## Custom Trainers

To implement a new training objective, subclass [`Trainer`](trainer/trainer.py) and override the `compute_loss(self, batch)` method:

```python
class MyCustomTrainer(Trainer):
    def compute_loss(self, batch):
        # Your custom loss logic here
        return loss
```

---

## Example Workflows

- **Pretraining on Text:** Use `TextDataset` and `PreTrain`.
- **Supervised Fine-Tuning (SFT):** Use `SFTDataset` and `PreTrain`.
- **Knowledge Distillation:** Use `KDL` trainer with a teacher model.
- **Reinforcement Learning (GRPO):** Use `GRPOTrainer` with a custom reward function.

---

## Logging & Visualization

- Training logs are colorized and informative.
- Live loss plots can be enabled for real-time monitoring.

---

## Saving & Loading

- Checkpoints are saved automatically.
- Resume or evaluate by loading checkpoints with the provided methods.

---

## Extending

- Add new models in the `models/` directory.
- Add new trainers in `trainer/custom_trainers/`.
- Use or extend dataset utilities in `create_dataset.py` and `parse_files.py`.

---

## License

MIT License

---

## Acknowledgements

- Built on PyTorch and HuggingFace Transformers.