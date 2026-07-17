import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import get_dataset, prepare_batch
from src.aggregation import MetricAccumulator, average_batch_metrics, pool_metrics
from src.helper import setup_logging, set_seed
from src.metrics import calculate_nfn_scores, calculate_projection_scores


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs"


def average_metrics(
    metrics_list,
    record_dist=False,
    method_type="norm",
    legacy_implementation=False,
):
    if legacy_implementation:
        return average_batch_metrics(metrics_list, method_type, record_dist=record_dist)
    return pool_metrics(metrics_list, method_type, record_dist=record_dist)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save baseline metrics for each dataset."
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model handle")
    parser.add_argument(
        "--setting",
        required=True,
        choices=["L1", "L2:PLang", "L2:Math", "L2:NatLang", "L2:NatLang-5"],
        help="Intent-classification setting",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("anchors"), help="Directory for baseline metrics"
    )
    parser.add_argument("--batchsize", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--nbsamples",
        type=int,
        default=None,
        help="Samples per class (default: the setting's value in configurations.json)",
    )
    parser.add_argument("--seqlen", type=int, default=256, help="Sequence length")
    parser.add_argument("--record_dist", action="store_true", help="Record distribution of norms")
    parser.add_argument(
        "--method_type",
        type=str,
        default="norm",
        choices=["norm", "projection"],
        help="Method type: norm (Method 1) or projection (Method 2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--train-percent",
        type=float,
        default=100.0,
        help="Percentage of sampled calibration data to use (0, 100]",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from the model repository",
    )
    parser.add_argument(
        "--legacy-implementation",
        action="store_true",
        help="Reproduce submitted-code scaling and per-batch anchor averaging",
    )

    args = parser.parse_args()

    with (CONFIG_DIR / "configurations.json").open(encoding="utf-8") as f:
        configuration_map = json.load(f)
    if args.nbsamples is None:
        args.nbsamples = configuration_map.get(args.setting, {}).get("train_nbsamples", 100)

    if not 0 < args.train_percent <= 100:
        parser.error("--train-percent must be in the interval (0, 100]")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    log_dir = REPO_ROOT / "logs" / "baselines"
    setup_logging(log_dir)
    logging.info(args)

    # Load model and tokenizer
    logging.info(f"Loading model: {args.model}")
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

    # Baseline datasets
    if args.setting == "L1":
        dataset_map = {
            "code": "magicoder",
            "text": "mmlu_history",
            "math": "gsm8k",
        }
    elif args.setting == "L2:PLang":
        lang_list = [
            "cpp",
            "csharp",
            "java",
            "php",
            "python",
            "rust",
            "shell",
            "swift",
            "typescript",
        ]
        dataset_map = {lang: f"magicoder:{lang}" for lang in lang_list}
    elif args.setting == "L2:Math":
        topic_list = [
            "Algebra",
            "Counting_&_Probability",
            "Geometry",
            "Intermediate_Algebra",
            "Number_Theory",
            "Prealgebra",
            "Precalculus",
        ]
        dataset_map = {topic: f"comp_math:{topic}" for topic in topic_list}
    elif args.setting == "L2:NatLang":
        with (CONFIG_DIR / "L2_NatLang.json").open(encoding="utf-8") as f:
            lang_list = json.load(f)["selected_lang"]
        dataset_map = {lang: f"aya:{lang}" for lang in lang_list}
    elif args.setting == "L2:NatLang-5":
        with (CONFIG_DIR / "L2_NatLang_5.json").open(encoding="utf-8") as f:
            lang_list = json.load(f)["selected_lang"]
        dataset_map = {lang: f"aya:{lang}" for lang in lang_list}
    else:
        raise KeyError(f"Unknown setting {args.setting}")

    logging.info(f"Dataset Map: {dataset_map}")

    for task, dataset_name in dataset_map.items():
        logging.info(f"\nProcessing baseline for task: {task}")
        problems = get_dataset(
            dataset_name=dataset_name,
            num_samples=args.nbsamples,
            tokenizer=tokenizer,
            split="train",
            seed=args.seed,
        )

        if args.train_percent < 100:
            num_train_samples = max(1, int(len(problems) * args.train_percent / 100.0))
            problems = problems[:num_train_samples]

        if not problems:
            raise ValueError(f"No calibration samples were loaded for {dataset_name}")

        logging.info(f"Sampled {len(problems)} problems from {dataset_name}")
        logging.info(f"Sample Input: {problems[0]}")
        batches = [
            problems[i : i + args.batchsize] for i in range(0, len(problems), args.batchsize)
        ]
        accumulator = MetricAccumulator(args.method_type, record_dist=args.record_dist)
        legacy_metrics = []
        for i, batch_problems in enumerate(batches):
            batch = prepare_batch(batch_problems, tokenizer, max_length=args.seqlen)

            if args.method_type == "norm":
                metrics = calculate_nfn_scores(
                    model,
                    batch,
                    record_dist=args.record_dist,
                    legacy_norm_width=args.legacy_implementation,
                )
            elif args.method_type == "projection":
                metrics = calculate_projection_scores(model, batch, record_dist=args.record_dist)
            else:
                raise KeyError(f"Unknown method type {args.method_type}")

            if args.legacy_implementation:
                legacy_metrics.append(metrics)
            else:
                accumulator.update(metrics)
        avg_metrics = (
            average_batch_metrics(
                legacy_metrics,
                args.method_type,
                record_dist=args.record_dist,
            )
            if args.legacy_implementation
            else accumulator.finalize()
        )

        method_suffix = "_proj" if args.method_type == "projection" else ""
        training_ratio_suffix = f"_{args.train_percent:g}" if args.train_percent != 100 else ""
        implementation_suffix = "_legacy" if args.legacy_implementation else ""
        metrics_path = args.output_dir / (
            args.model.split("/")[-1]
            + "_"
            + task
            + "_"
            + str(args.seed)
            + method_suffix
            + training_ratio_suffix
            + implementation_suffix
            + "_metrics.json"
        )
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(avg_metrics, f, indent=2)
        logging.info(f"Saved baseline metrics to {metrics_path}")


if __name__ == "__main__":
    main()
