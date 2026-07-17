"""Evaluate a causal LLM with the paper's zero-shot or few-shot intent prompts."""

import argparse
import json
import logging
import random
import re
import string
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import get_dataset
from src.direct_prompts import (
    LABEL_ORDER,
    label_to_letter,
    render_chat_messages,
    render_prompt,
    select_examples,
)
from src.helper import ResultsDB, set_seed, setup_logging
from src.paths import slugify_path_component


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "configs" / "configurations.json"
FEW_SHOT_SETTINGS = {"L1", "L1:Adv"}


def parse_prediction(text, valid_letters):
    """Extract a class letter while retaining the raw generation for auditing."""

    stripped = text.strip().upper()
    match = re.search(r"\b([A-Z])\b", stripped)
    if match and match.group(1) in valid_letters:
        return match.group(1)
    if stripped and stripped[0] in valid_letters:
        return stripped[0]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model identifier")
    parser.add_argument(
        "--setting",
        required=True,
        choices=["L1", "L1:Adv", "L2:PLang", "L2:Math", "L2:NatLang-5"],
    )
    parser.add_argument(
        "--nbsamples", type=int, default=100, help="Maximum samples per test dataset"
    )
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument(
        "--seqlen",
        type=int,
        default=4096,
        help="Maximum prompt length after tokenization",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-of-shots",
        type=int,
        default=0,
        help="In-context examples per L1 class (the paper uses 0 or 3)",
    )
    parser.add_argument(
        "--icl-pool-size",
        type=int,
        default=2000,
        help="Maximum calibration examples loaded per class for few-shot selection",
    )
    parser.add_argument(
        "--chat-model", action="store_true", help="Render prompts with the tokenizer chat template"
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode (disabled by default to match the paper)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--record-dir", type=Path, default=Path("exp_records"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/direct_llm"))
    return parser.parse_args()


def main():
    args = parse_args()
    positive_values = {
        "--nbsamples": args.nbsamples,
        "--batchsize": args.batchsize,
        "--seqlen": args.seqlen,
        "--max-new-tokens": args.max_new_tokens,
        "--icl-pool-size": args.icl_pool_size,
    }
    for option, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{option} must be positive")
    if args.num_of_shots < 0:
        raise ValueError("--num-of-shots cannot be negative")
    if args.num_of_shots and args.setting not in FEW_SHOT_SETTINGS:
        raise ValueError(
            "The paper's few-shot baseline is defined only for L1 and L1:Adv; "
            "use --num-of-shots 0 for fine-grained settings"
        )

    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = json.load(f)[args.setting]
    ordered_labels = LABEL_ORDER[args.setting]
    if set(ordered_labels) != set(config["label2id"]):
        raise ValueError(f"Prompt labels do not match the configuration for {args.setting}")
    label_letters = label_to_letter(args.setting)
    valid_letters = set(label_letters.values())

    set_seed(args.seed)
    prompt_rng = random.Random(args.seed)
    setup_logging(REPO_ROOT / "logs" / "direct_llm")
    logging.info("Arguments: %s", args)

    in_context_pools = {}
    if args.num_of_shots:
        pool_size = max(args.icl_pool_size, args.num_of_shots)
        for label in ordered_labels:
            in_context_pools[label] = get_dataset(
                config["training set"][label],
                num_samples=pool_size,
                split="train",
                seed=args.seed,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.eval()
    input_device = next(model.parameters()).device

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setting_slug = slugify_path_component(args.setting)
    database = ResultsDB(args.record_dir / f"{setting_slug}_direct_llm.duckdb")

    for dataset_name, label, available in config["test set"]:
        sample_count = min(args.nbsamples, available)
        problems = get_dataset(
            dataset_name,
            num_samples=sample_count,
            split="test",
            seed=args.seed,
        )
        expected = label_letters[label]
        predictions = []
        raw_outputs = []

        for start in tqdm(range(0, len(problems), args.batchsize), desc=dataset_name):
            batch_problems = problems[start : start + args.batchsize]
            prompts = []
            for offset, problem in enumerate(batch_problems):
                examples = []
                if args.num_of_shots:
                    examples = select_examples(
                        in_context_pools,
                        label_letters,
                        query_index=start + offset,
                        shots_per_class=args.num_of_shots,
                        rng=prompt_rng,
                    )

                if args.chat_model:
                    if not tokenizer.chat_template:
                        raise ValueError(
                            "--chat-model was set, but the tokenizer has no chat template"
                        )
                    template_kwargs = {}
                    if "qwen" in args.model.lower():
                        template_kwargs["enable_thinking"] = args.enable_thinking
                    prompt = tokenizer.apply_chat_template(
                        render_chat_messages(args.setting, problem, examples),
                        tokenize=False,
                        add_generation_prompt=True,
                        **template_kwargs,
                    )
                else:
                    prompt = render_prompt(args.setting, problem, examples)
                prompts.append(prompt)

            tokens = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.seqlen,
            )
            tokens = {key: value.to(input_device) for key, value in tokens.items()}
            with torch.no_grad():
                generated = model.generate(
                    **tokens,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = generated[:, tokens["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            raw_outputs.extend(decoded)
            predictions.extend(parse_prediction(text, valid_letters) for text in decoded)

        correct = sum(prediction == expected for prediction in predictions)
        accuracy = correct / max(len(predictions), 1)
        logging.info("Task %s (%s): accuracy %.4f", dataset_name, label, accuracy)

        model_slug = slugify_path_component(args.model.split("/")[-1])
        dataset_slug = slugify_path_component(dataset_name)
        output_path = args.output_dir / (
            f"{model_slug}_{dataset_slug}_K{args.num_of_shots}_S{args.seed}.json"
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "setting": args.setting,
                    "dataset": dataset_name,
                    "label": label,
                    "expected_letter": expected,
                    "num_shots_per_class": args.num_of_shots,
                    "accuracy": accuracy,
                    "predictions": predictions,
                    "raw_outputs": raw_outputs,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        method = (
            "LLM:zero-shot" if args.num_of_shots == 0 else f"LLM:{args.num_of_shots}-shot-per-class"
        )
        database.log(
            model=args.model,
            dataset=dataset_name,
            method=method,
            n_layers=model.config.num_hidden_layers,
            seqlen=args.seqlen,
            seed=args.seed,
            accuracy=accuracy,
            n_samples=len(predictions),
            batchsize=args.batchsize,
            nbsamples=0,
        )


if __name__ == "__main__":
    main()
