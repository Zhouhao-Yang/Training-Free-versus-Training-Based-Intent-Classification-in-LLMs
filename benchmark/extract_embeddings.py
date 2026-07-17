"""Extract frozen LLM embeddings for the learned intent classifiers."""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.utils import embedding_filename
from src.data import get_dataset, prepare_batch
from src.helper import set_seed, setup_logging


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "configurations.json"


def compute_weighted_average(last_hidden_state, attention_mask):
    batch_size, seq_len, _ = last_hidden_state.shape
    position_weights = (
        torch.arange(
            1,
            seq_len + 1,
            device=last_hidden_state.device,
            dtype=torch.float,
        )
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    masked_weights = position_weights * attention_mask.float()
    normalized_weights = masked_weights / masked_weights.sum(dim=-1, keepdim=True).clamp_min(1)
    return torch.sum(last_hidden_state * normalized_weights.unsqueeze(-1), dim=-2)


def parse_layer_indices(value, num_hidden_layers):
    if value:
        indices = sorted({int(item) for item in value.split(",")})
    else:
        indices = list(range(4, num_hidden_layers + 1, 4))
        if num_hidden_layers not in indices:
            indices.append(num_hidden_layers)
    invalid = [idx for idx in indices if idx < 1 or idx > num_hidden_layers]
    if invalid:
        raise ValueError(f"Layer indices must be between 1 and {num_hidden_layers}; got {invalid}")
    return indices


def dataset_plan(config, mode, nbsamples):
    if mode == "train":
        default_size = config["train_nbsamples"]
        size = nbsamples or default_size
        return [
            (label, dataset_name, size) for label, dataset_name in config["training set"].items()
        ]

    plan = []
    for dataset_name, _label, available in config["test set"]:
        size = available if nbsamples is None else min(available, nbsamples)
        plan.append((dataset_name, dataset_name, size))
    return plan


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model identifier")
    parser.add_argument(
        "--setting",
        required=True,
        choices=["L1", "L1:Adv", "L2:PLang", "L2:Math", "L2:NatLang-5"],
    )
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--nbsamples", type=int, default=None, help="Optional per-dataset cap")
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all", action="store_true", help="Save selected intermediate layers as dictionaries"
    )
    parser.add_argument(
        "--layer-indices", help="Comma-separated hidden-state indices used with --all"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.nbsamples is not None and args.nbsamples <= 0:
        raise ValueError("--nbsamples must be positive")

    with CONFIG_PATH.open(encoding="utf-8") as f:
        config_map = json.load(f)
    config = config_map[args.setting]
    plan = dataset_plan(config, args.mode, args.nbsamples)

    if args.output_dir is None:
        args.output_dir = REPO_ROOT / "artifacts" / "embeddings" / args.mode
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    setup_logging(REPO_ROOT / "logs" / "benchmark")
    logging.info("Arguments: %s", args)
    logging.info("Dataset plan: %s", plan)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    layer_indices = None
    if args.all:
        layer_indices = parse_layer_indices(args.layer_indices, model.config.num_hidden_layers)
        logging.info("Saving hidden-state indices: %s", layer_indices)

    model.eval()
    input_device = next(model.parameters()).device
    for task, dataset_name, size in plan:
        problems = get_dataset(
            dataset_name=dataset_name,
            num_samples=size,
            tokenizer=tokenizer,
            split=args.mode,
            seed=args.seed,
        )
        if not problems:
            raise ValueError(f"No samples loaded for {dataset_name}")
        logging.info("Loaded %d samples from %s", len(problems), dataset_name)

        dataloader = DataLoader(
            problems,
            batch_size=args.batchsize,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=partial(prepare_batch, tokenizer=tokenizer, max_length=args.seqlen),
        )

        if layer_indices is None:
            last_tokens, avg_tokens, weighted_avg_tokens = [], [], []
        else:
            last_tokens = {idx: [] for idx in layer_indices}
            avg_tokens = {idx: [] for idx in layer_indices}
            weighted_avg_tokens = {idx: [] for idx in layer_indices}

        for batch in tqdm(dataloader, desc=dataset_name):
            batch = {key: value.to(input_device) for key, value in batch.items()}
            with torch.no_grad():
                output = model(**batch, output_hidden_states=True)

            attention_mask = batch["attention_mask"]
            last_indices = attention_mask.sum(dim=1).long() - 1
            batch_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)

            if layer_indices is None:
                hidden_state = output.hidden_states[-1]
                last_tokens.append(hidden_state[batch_indices, last_indices].cpu())
                sequence_sum = torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=-2)
                avg_tokens.append(
                    (sequence_sum / attention_mask.sum(dim=-1, keepdim=True).clamp_min(1)).cpu()
                )
                weighted_avg_tokens.append(
                    compute_weighted_average(hidden_state, attention_mask).cpu()
                )
            else:
                for idx in layer_indices:
                    hidden_state = output.hidden_states[idx]
                    last_tokens[idx].append(hidden_state[batch_indices, last_indices].cpu())
                    sequence_sum = torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=-2)
                    avg_tokens[idx].append(
                        (sequence_sum / attention_mask.sum(dim=-1, keepdim=True).clamp_min(1)).cpu()
                    )
                    weighted_avg_tokens[idx].append(
                        compute_weighted_average(hidden_state, attention_mask).cpu()
                    )

        if layer_indices is None:
            result = {
                "last_tokens": torch.cat(last_tokens),
                "avg_tokens": torch.cat(avg_tokens),
                "weighted_avg_tokens": torch.cat(weighted_avg_tokens),
            }
        else:
            result = {
                "last_tokens": {idx: torch.cat(values) for idx, values in last_tokens.items()},
                "avg_tokens": {idx: torch.cat(values) for idx, values in avg_tokens.items()},
                "weighted_avg_tokens": {
                    idx: torch.cat(values) for idx, values in weighted_avg_tokens.items()
                },
            }

        result["_metadata"] = {
            "model": args.model,
            "num_hidden_layers": model.config.num_hidden_layers,
            "dataset": dataset_name,
            "task": task,
            "mode": args.mode,
            "seed": args.seed,
            "seqlen": args.seqlen,
            "layer_indices": layer_indices,
        }

        result_path = args.output_dir / embedding_filename(
            args.model,
            task,
            args.seed,
            args.mode,
            seqlen=args.seqlen,
            all_layers=args.all,
        )
        torch.save(result, result_path)
        logging.info("Saved embeddings to %s", result_path)


if __name__ == "__main__":
    main()
