import argparse
import json
import logging
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import get_dataset, prepare_batch
from src.helper import setup_logging, set_seed, ResultsDB
from src.metrics import (
    calculate_nfn_scores,
    calculate_projection_scores,
    infer_task_from_projection_scores,
    infer_task_from_scores,
)
from src.paths import slugify_path_component


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs"


def main():
    parser = argparse.ArgumentParser(description="Classify tasks using alignment metrics.")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model handle")
    parser.add_argument(
        "--setting",
        required=True,
        choices=["L1", "L2:PLang", "L2:Math", "L2:NatLang", "L2:NatLang-5"],
        help="Intent-classification setting",
    )
    parser.add_argument("--batchsize", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--nbsamples", type=int, default=100, help="Number of samples to use from dataset"
    )
    parser.add_argument(
        "--nb_trainsamples",
        type=int,
        default=None,
        help="Number of training samples to use from dataset",
    )
    parser.add_argument("--seqlen", type=int, default=256, help="Sequence length")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results"), help="Directory to save results"
    )
    parser.add_argument(
        "--record_dir",
        type=Path,
        default=Path("exp_records"),
        help="Directory to save experiment records",
    )
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset to use for testing")
    parser.add_argument("--label", type=str, default="code", help="Label to use for classification")
    parser.add_argument(
        "--baseline_dir",
        type=Path,
        default=Path("anchors"),
        help="Directory containing baseline metrics",
    )
    parser.add_argument(
        "--method", default="KL", choices=["mean", "KL"], help="Distance/similarity method"
    )
    parser.add_argument(
        "--method_type",
        type=str,
        default="norm",
        choices=["norm", "projection"],
        help="Method type: norm (Method 1) or projection (Method 2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n_layers", type=int, default=None, help="Number of hidden layers to collect information."
    )

    parser.add_argument(
        "--train-percent",
        type=float,
        default=100.0,
        help="Percentage used when generating anchors (0, 100]",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from the model repository",
    )
    parser.add_argument(
        "--legacy-implementation",
        action="store_true",
        help="Use anchors and NormStat scaling compatible with the submitted code",
    )

    args = parser.parse_args()

    if not 0 < args.train_percent <= 100:
        parser.error("--train-percent must be in the interval (0, 100]")
    with (CONFIG_DIR / "configurations.json").open(encoding="utf-8") as f:
        configuration_map = json.load(f)
    if args.nb_trainsamples is None and args.setting in configuration_map:
        args.nb_trainsamples = configuration_map[args.setting]["train_nbsamples"]

    if args.setting == "L1":
        baseline_tasks = ["code", "text", "math"]
        args.record_db = args.record_dir / "L1_record.duckdb"
    elif args.setting == "L2:PLang":
        p_lang = args.dataset.split("magicoder:")[1]
        assert args.label == p_lang

        baseline_tasks = [
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

        with (CONFIG_DIR / "L2_PLang.json").open(encoding="utf-8") as f:
            test_size_map = json.load(f)

        args.nbsamples = min(test_size_map[p_lang] - args.nb_trainsamples, args.nbsamples)
        args.record_db = args.record_dir / "L2_lang_record.duckdb"
    elif args.setting == "L2:Math":
        topic = args.dataset.split("comp_math:")[1]
        assert args.label == topic

        baseline_tasks = [
            "Algebra",
            "Counting_&_Probability",
            "Geometry",
            "Intermediate_Algebra",
            "Number_Theory",
            "Prealgebra",
            "Precalculus",
        ]

        with (CONFIG_DIR / "L2_Math.json").open(encoding="utf-8") as f:
            test_size_map = json.load(f)

        args.nbsamples = min(test_size_map[topic] - args.nb_trainsamples, args.nbsamples)
        args.record_db = args.record_dir / "L2_math_record.duckdb"
    elif args.setting == "L2:NatLang":
        nat_lang = args.dataset.split("aya:")[1]
        assert args.label == nat_lang

        if args.nb_trainsamples is None:
            parser.error("--nb_trainsamples is required for L2:NatLang")
        with (CONFIG_DIR / "L2_NatLang.json").open(encoding="utf-8") as f:
            nat_lang_meta = json.load(f)
        baseline_tasks = nat_lang_meta["selected_lang"]
        test_size_map = nat_lang_meta["data_size"]

        args.nbsamples = min(test_size_map[nat_lang] - args.nb_trainsamples, args.nbsamples)
        args.record_db = args.record_dir / "L2_natlang_record.duckdb"
    elif args.setting == "L2:NatLang-5":
        nat_lang = args.dataset.split("aya:")[1]
        assert args.label == nat_lang

        with (CONFIG_DIR / "L2_NatLang_5.json").open(encoding="utf-8") as f:
            nat_lang_meta = json.load(f)
        baseline_tasks = nat_lang_meta["selected_lang"]
        test_size_map = nat_lang_meta["data_size"]

        args.nbsamples = min(test_size_map[nat_lang] - args.nb_trainsamples, args.nbsamples)
        args.record_db = args.record_dir / "L2_natlang_record_5.duckdb"
    else:
        raise KeyError(f"Unknown setting {args.setting}")

    if args.nbsamples <= 0:
        parser.error("No held-out samples remain; reduce --nb_trainsamples")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    log_dir = REPO_ROOT / "logs" / "classifier"
    setup_logging(log_dir)
    logging.info(args)
    logging.info(f"Baseline tasks: {baseline_tasks}")

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

    layer_indices = list(range(model.config.num_hidden_layers))
    if args.n_layers is not None:
        assert args.n_layers <= len(layer_indices), f"n_layers must be <= {len(layer_indices)}"
        layer_indices = layer_indices[: args.n_layers]
    args.n_layers = len(layer_indices)

    # Load test set
    logging.info(f"Loading test set: {args.dataset}")
    problems = get_dataset(
        args.dataset, num_samples=args.nbsamples, tokenizer=tokenizer, split="test", seed=args.seed
    )
    logging.info(f"Loaded {len(problems)} problems from {args.dataset}")
    logging.info(f"Sample Input: {problems[0]}")

    # Create DataLoader
    dataloader = DataLoader(
        problems,
        batch_size=args.batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(prepare_batch, tokenizer=tokenizer, max_length=args.seqlen),
    )
    logging.info(f"Processing {len(dataloader)} batches of size up to {args.batchsize}")

    # Load baselines (math, code)
    model_short = args.model.split("/")[-1]
    baseline_scores = {}
    method_suffix = "_proj" if args.method_type == "projection" else ""
    training_ratio_suffix = f"_{args.train_percent:g}" if args.train_percent != 100 else ""
    implementation_suffix = "_legacy" if args.legacy_implementation else ""
    for task in baseline_tasks:
        baseline_path = (
            args.baseline_dir
            / f"{model_short}_{task}_{args.seed}{method_suffix}{training_ratio_suffix}{implementation_suffix}_metrics.json"
        )
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline file not found for {task}: {baseline_path}")
        with baseline_path.open(encoding="utf-8") as f:
            baseline_scores[task] = json.load(f)
    if not baseline_scores:
        raise ValueError(
            "No baseline scores found. Please provide baseline metrics in the baseline_dir."
        )

    label_to_idx = {k: i for i, k in enumerate(baseline_scores.keys())}

    # For each batch, calculate metrics and classify
    y_true = [label_to_idx[args.label]] * len(problems)
    y_pred, dist_list, metrics = [], [], None
    for batch in tqdm(dataloader, total=len(dataloader)):
        if args.method_type == "norm":
            metrics = calculate_nfn_scores(
                model=model,
                batch=batch,
                mode="test",
                allowed_layers=layer_indices,
                legacy_norm_width=args.legacy_implementation,
            )
            pred_task, distances = infer_task_from_scores(
                metrics, baseline_scores, method=args.method
            )  # (B,) (B, 3)
        elif args.method_type == "projection":
            metrics = calculate_projection_scores(
                model=model, batch=batch, mode="test", allowed_layers=layer_indices
            )
            pred_task, distances = infer_task_from_projection_scores(
                metrics, baseline_scores, method=args.method
            )  # (B,) (B, 3)
        else:
            raise ValueError(f"Unknown method type {args.method_type}")

        y_pred.extend(pred_task.cpu().tolist())
        dist_list.append(distances)

    logging.info(f"Monitored Layers: {metrics.keys()}")

    # Compute accuracy
    y_true = torch.tensor(y_true).int()
    y_pred = torch.tensor(y_pred).int()
    accuracy = torch.mean((y_true == y_pred).float())
    logging.info(f"Classification accuracy: {accuracy:.3f}")

    # Save results
    results = {
        "accuracy": accuracy,
        "n_samples": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
        "dist_list": torch.cat(dist_list, dim=0),
    }
    method_suffix = f"_{args.method_type}_{args.method}"
    implementation_suffix = "_legacy" if args.legacy_implementation else ""
    dataset_slug = slugify_path_component(args.dataset)
    out_path = (
        args.output_dir
        / f"{model_short}_{dataset_slug}_{args.seed}_L{args.n_layers}_S{args.seqlen}{method_suffix}{training_ratio_suffix}{implementation_suffix}_classification_results.pt"
    )
    torch.save(results, out_path)
    logging.info(f"Saved classification results to {out_path}")

    args.method = f"P:{args.method}" if args.method_type == "projection" else args.method
    if args.legacy_implementation:
        args.method = f"{args.method}:legacy"
    db = ResultsDB(args.record_db)
    db.log(
        model=args.model,
        dataset=args.dataset,
        method=args.method,
        n_layers=args.n_layers,
        seqlen=args.seqlen,
        seed=args.seed,  # use your parsed seed
        accuracy=accuracy,
        n_samples=len(y_true),
        batchsize=args.batchsize,
        nbsamples=max(1, int(args.nb_trainsamples * args.train_percent / 100.0)),
    )


if __name__ == "__main__":
    main()
