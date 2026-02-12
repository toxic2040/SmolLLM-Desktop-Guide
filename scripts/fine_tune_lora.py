# fine_tune_lora.py: LoRA fine-tuning for TA adaptation. Focus: Low-VRAM on 7900 GRE.
# Details: Uses Unsloth; bfloat16/4bit. ~2-4 hours on subset. Integrations: LlamaFactory YAML configs.
# Usage: python scripts/fine_tune_lora.py --model qwen2.5-vl:7b-q5_K_M --dataset annotations.json --output btc-vl-lora

import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
import torch
import yaml  # For config (LlamaFactory-inspired)

parser = argparse.ArgumentParser(description="Fine-tune LoRA for Bitcoin TA.")
parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model.")
parser.add_argument("--dataset", default="examples/btc_dataset_sample/annotations.json", help="JSON dataset.")
parser.add_argument("--output", default="models/btc-vl-lora", help="Output dir.")
parser.add_argument("--config", default=None, help="YAML config file.")
args = parser.parse_args()

# Load config if provided (e.g., r=16, epochs=3)
config = yaml.safe_load(open(args.config)) if args.config else {"r": 16, "epochs": 3, "batch": 4}

model, tokenizer = FastLanguageModel.from_pretrained(args.model, dtype=torch.bfloat16, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=config["r"])  # LoRA

dataset = load_dataset("json", data_files=args.dataset, split="train")
trainer = SFTTrainer(model=model, train_dataset=dataset, tokenizer=tokenizer, max_seq_length=2048, args={"num_train_epochs": config["epochs"], "per_device_train_batch_size": config["batch"]})
trainer.train()

model.save_pretrained(args.output)
print(f"LoRA adapters saved to {args.output}. Merge for Ollama use.")
