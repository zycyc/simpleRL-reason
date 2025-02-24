import json
from datasets import load_dataset
from tqdm import tqdm
from utils import PROMPT_TEMPLATES

prompt_temp = PROMPT_TEMPLATES["qwen25-math-cot"]
input_template, output_template, splitter = (
    prompt_temp[0],
    prompt_temp[1],
    prompt_temp[2],
)

data = load_dataset("GAIR/LIMR")


def process_example(example):
    example["input"] = input_template.format(input=example["prompt"]).strip()
    return example


data["train"] = data["train"].map(process_example)
data["train"] = data["train"].rename_column("prompt", "question")

output_path = (
    "/data/alan-g491/simpleRL-reason/train/data/limr_processed_with_qwen_prompt.json"
)

examples = list(data["train"])
for example in examples:
    example["gt_answer"] = example["answer"]
    example["ground_truth_answer"] = example["answer"]
    example["target"] = example["answer"]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=4)
