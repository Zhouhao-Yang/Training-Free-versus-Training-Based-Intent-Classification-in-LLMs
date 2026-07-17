import json
import random
import re
from functools import partial
from pathlib import Path

import numpy as np
import requests
from datasets import load_dataset

# Only keep the minimal set of functions needed for the simplified main

_lang_re = re.compile(r"```([a-z]+)\b")

lang_map = {
    "cpp": ["cpp", "c"],
    "csharp": ["csharp"],
    "java": ["java"],
    "php": ["php"],
    "python": ["python"],
    "rust": ["rust"],
    "shell": ["bash", "powershell"],
    "swift": ["swift"],
    "typescript": ["typescript", "javascript", "jsx", "tsx", "ts"],
}


def lang_filter(example, lang: str = None):
    if lang is None:
        return True

    if example["lang"] != lang:
        return False

    code_lang = _lang_re.search(example["solution"])

    if not code_lang:
        return False
    else:
        return code_lang.group(1) in lang_map[lang]


def load_gsm8k_problems(split="train"):
    """
    Load problems from the GSM8K dataset.
    Args:
        num_samples: Number of samples to load.
    Returns:
        List of problems.
    Remark: For now, we only keep the questions. Answers can be included in the samples as well.
    """
    if split == "train":
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    elif split == "test":
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    else:
        raise ValueError(f"Unknown split: {split}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = response.text.strip().split("\n")
    problems = [json.loads(line)["question"] for line in lines]
    return problems


def load_mmlu_problems(subject="logical_fallacies"):
    """
    Load problems from the MMLU dataset.
    Args:
        subject: Subject to load.
        num_samples: Number of samples to load.
    Returns:
        List of problems.
    """
    dataset = load_dataset("cais/mmlu", subject, split="test")
    problems = []
    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        problem = f"{question}\n" + "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        problems.append(problem)
    return problems


def load_humaneval_problems():
    """
    Load problems from the HumanEval dataset.
    Args:
        num_samples: Number of samples to load.
    Returns:
        List of problems.
    """
    dataset = load_dataset("openai_humaneval", split="test")
    problems = [f"# Write a Python function\n{item['prompt']}" for item in dataset]
    return problems


def load_math500_problems(split="test"):
    """
    Load problems from the HuggingFaceH4/MATH-500 dataset.
    Args:
        num_samples: Number of samples to load.
        split: Which split to use (default 'test').
    Returns:
        List of problems (questions as strings).
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
    problems = [item["problem"] for item in dataset]
    return problems


def load_magicoder_problems(split="train", language=None):
    """
    Load problems from the ise-uiuc/Magicoder-OSS-Instruct-75K dataset.
    Args:
        num_samples: Number of samples to load.
        split: Which split to use (default 'test').
    Returns:
        List of problems (prompts as strings).
    """
    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split=split)
    dataset = dataset.filter(partial(lang_filter, lang=language))
    problems = [item["solution"] for item in dataset]
    return problems


def load_competition_math_problems(topic=None):
    dataset = load_dataset("qwedsacf/competition_math", split="train")
    if topic:
        dataset = dataset.filter(lambda x: x["type"] == topic)
    problems = [item["problem"] for item in dataset]
    return problems


def load_aya_problems(language=None):
    dataset = load_dataset("CohereLabs/aya_dataset", split="train")
    if language:
        dataset = dataset.filter(lambda x: x["language"] == language)
    problems = [item["inputs"] for item in dataset]
    return problems


def load_magicoder_combined_problems(num_samples=500, k=5, split="train", language=None):
    """
    Load problems from the Magicoder dataset, concatenating k samples for each problem.
    Args:
        num_samples: Number of concatenated samples to return.
        k: Number of samples to concatenate for each problem.
        split: Which split to use (default 'train').
        language: Optional language filter.
    Returns:
        List of concatenated problems (as strings).
    """
    total_needed = num_samples * k
    # Get enough samples to concatenate
    base_samples = load_magicoder_problems(split=split, language=language)
    if len(base_samples) < total_needed:
        raise ValueError(
            f"Requested {total_needed} Magicoder samples, but only {len(base_samples)} are available"
        )
    random.shuffle(base_samples)
    combined = []
    for i in range(num_samples):
        group = base_samples[i * k : (i + 1) * k]
        combined.append("\n\n".join(group))
    return combined


def prepare_batch(problems, tokenizer, max_length=256):
    """
    Prepare a batch of problems for the model.
    Args:
        problems: List of problems.
        tokenizer: Tokenizer.
        max_length: Maximum sequence length.
    Returns:
        Dictionary of encoded problems.
    """
    encoded = tokenizer(
        problems,
        padding=True,
        pad_to_multiple_of=8,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return encoded


def get_dataset(dataset_name="math", num_samples=10, tokenizer=None, split="train", seed: int = 42):
    """
    Meta function to load a dataset.
    Args:
        dataset_name: Name of the dataset.
        num_samples: Number of samples to load.
        tokenizer: Tokenizer.
        split: Split to load.
    Returns:
        List of problems.
    """
    # dataset loading
    if dataset_name == "gsm8k":
        problems = load_gsm8k_problems(split)
    elif dataset_name == "mmlu_logic":
        problems = load_mmlu_problems("logical_fallacies")
    elif dataset_name == "mmlu_history" and split == "train":
        problems = load_mmlu_problems("high_school_european_history")
    elif dataset_name == "mmlu_history" and split == "test":
        problems = load_mmlu_problems("high_school_us_history")
    elif dataset_name == "humaneval":
        problems = load_humaneval_problems()
    elif dataset_name == "math500":
        problems = load_math500_problems()
    elif dataset_name == "magicoder":
        problems = load_magicoder_problems()
    elif dataset_name == "magicoder_combined":
        problems = load_magicoder_combined_problems(num_samples)
    elif "magicoder:" in dataset_name:
        lang = dataset_name.split("magicoder:")[1]
        problems = load_magicoder_problems(language=lang)
    elif "comp_math:" in dataset_name:
        topic = dataset_name.split("comp_math:")[1].replace("_", " ")
        problems = load_competition_math_problems(topic=topic)
    elif "aya:" in dataset_name:
        lang = dataset_name.split("aya:")[1].replace("_", " ")
        problems = load_aya_problems(language=lang)
    elif dataset_name in ["adv_math500_easy", "adv_math500_medium", "adv_math500_hard"]:
        difficulty = dataset_name.split("adv_math500_")[1]
        ds = load_dataset("nanchennn/Adv_MATH500", difficulty)
        problems = [item["text"] for item in ds["train"]]
    elif dataset_name in ["adv_math500", "math500_c1"]:
        project_root = Path(__file__).parent.parent
        filename = {
            "adv_math500": "adv_math500.json",
            "math500_c1": "math500.json",
        }[dataset_name]
        file_path = project_root / "adv_datasets" / filename
        with file_path.open(encoding="utf-8") as f:
            data_dict = json.load(f)
        problems = data_dict["problem"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # postprocessing
    if num_samples is None or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    num_samples = min(num_samples, len(problems))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(problems)).tolist()

    if split == "test" and any(name in dataset_name for name in ["magicoder", "comp_math", "aya"]):
        return [problems[i] for i in perm[-num_samples:]]
    else:
        return [problems[i] for i in perm[:num_samples]]
