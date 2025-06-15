import torch
import torch.nn as nn
import argparse
import deepspeed
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
import os

from transformers import DataCollatorForSeq2Seq

# Argument parser
def add_argument():
    parser = argparse.ArgumentParser(description='Simple CNN Training')
    parser.add_argument('-b',
                        '--batch_size',
                        default=2,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=5,
                        type=int,
                        help='number of total epochs (default: 5)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--log_interval',
                        type=int,
                        default=1,
                        help='output logging information at a given interval')
    parser.add_argument('--model_path',
                        type=str,
                        default='/root/autodl-tmp/models/Qwen2-0.5B-Instruct',
                        help='output logging information at a given interval')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }

# Data processing function
def process_func(example, tokenizer):
    MAX_LENGTH = 128
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        "\n".join(["<|im_start|>system", "现在你要扮演皇帝身边的女人--甄嬛.<|im_end|>" + "\n<|im_start|>user\n" + example["instruction"] + example["input"] + "<|im_end|>\n"]).strip(),
        add_special_tokens=False
    )
    response = tokenizer("<|im_start|>assistant\n" + example["output"] + "<|im_end|>\n", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Main training loop
def main():
    args = add_argument()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and process data
    with open('/root/autodl-tmp/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    tokenized_data = [process_func(example, tokenizer) for example in data]

    # Create dataset and dataloader
    trainset = CustomDataset(tokenized_data)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.half
    )

    # Initialize DeepSpeed
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters
    )

    # Training loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, batch in enumerate(trainloader):
            inputs = {
                "input_ids": batch["input_ids"].to(model_engine.local_rank),
                "attention_mask": batch["attention_mask"].to(model_engine.local_rank),
                "labels": batch["labels"].to(model_engine.local_rank)
            }
            print(input)
            outputs = model_engine(**inputs)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item()
            print(loss)
            # if i % args.log_interval == (args.log_interval - 1):
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / args.log_interval:.3f}')
            #     running_loss = 0.0

    # Test model
    model_engine.eval()
    with torch.no_grad():
        inputs = tokenizer(
            "<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛.<|im_end|>\n<|im_start|>user\n你是谁<|im_end|>\n",
            return_tensors="pt",
            add_special_tokens=False
        ).to(model_engine.local_rank)
        outputs = model_engine.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.6
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print("Model response:", response)

if __name__ == '__main__':
    main()