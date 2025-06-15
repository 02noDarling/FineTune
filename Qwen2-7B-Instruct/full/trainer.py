from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, HfArgumentParser, Trainer
import os
import torch
import json
from dataclasses import dataclass, field
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

@dataclass
class FinetuneArguments:
    # 微调参数
    model_path: str = field(default="/root/Qwen2-7B-Instruct")

# 用于处理数据的函数
def process_func(example):
    MAX_LENGTH = 128  # 最大长度，适应中文分词
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

if __name__ == "__main__":
    # 解析命令行参数
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 直接读取 JSON 文件
    with open('/root/autodl-tmp/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理数据为 token 形式
    tokenized_data = [process_func(example) for example in data]

    # 创建模型并以半精度加载
    print(os.environ.get("LOCAL_RANK"))
    model = AutoModelForCausalLM.from_pretrained(
        finetune_args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.half,
        # device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}
    )

    # 使用 Trainer 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()  # 开始训练

    # 测试模型
    # response, history = model.chat(tokenizer, "你是谁", history=[], system="现在你要扮演皇帝身边的女人--甄嬛.")
    # print(response)