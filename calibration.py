import argparse
import re
from pathlib import Path

import torch
import numpy as np
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import get_dataset, prepare_batch
from src.helper import setup_logging, set_seed
from src.metrics import calculate_nfn_scores, calculate_projection_scores
from src.aggregation import MetricAccumulator, pool_metrics


def compute_calibration_error_projection(true_metrics, estimated_metrics):
    """
    Compute calibration error for projection method.

    Args:
        true_metrics: Ground truth metrics dict {layer_name: {'mean_vec': [...], 'var_vec': [...]}}
        estimated_metrics: Estimated metrics dict with same structure

    Returns:
        dict: Calibration errors per layer and aggregated
    """
    errors = {}

    for layer_name in true_metrics:
        if layer_name not in estimated_metrics:
            continue

        true_mean = np.array(true_metrics[layer_name]["mean_vec"])
        est_mean = np.array(estimated_metrics[layer_name]["mean_vec"])

        true_var = np.array(true_metrics[layer_name]["var_vec"])
        est_var = np.array(estimated_metrics[layer_name]["var_vec"])

        mean_error = np.linalg.norm(true_mean - est_mean)

        var_error = np.max(np.abs(true_var - est_var))

        errors[layer_name] = {"mean_error": float(mean_error), "var_error": float(var_error)}

    return errors


def compute_calibration_error_norm(true_metrics, estimated_metrics):
    """
    Compute calibration error for norm method.

    Args:
        true_metrics: Ground truth metrics dict {layer_name: {'mean': x, 'std': y, 'median': z}}
        estimated_metrics: Estimated metrics dict with same structure

    Returns:
        dict: Calibration errors per layer and aggregated
    """
    errors = {}

    for layer_name in true_metrics:
        if layer_name not in estimated_metrics:
            continue

        true_mean = true_metrics[layer_name]["mean"]
        est_mean = estimated_metrics[layer_name]["mean"]

        true_std = true_metrics[layer_name]["std"]
        est_std = estimated_metrics[layer_name]["std"]

        mean_error = abs(true_mean - est_mean)

        std_error = abs(true_std - est_std)

        errors[layer_name] = {"mean_error": float(mean_error), "std_error": float(std_error)}

    return errors


def aggregate_errors(errors, aggregation_type="average"):
    """
    Aggregate calibration errors across layers.

    Args:
        errors: Dict of errors per layer
        aggregation_type: 'average' or 'single_layer'

    Returns:
        dict: Aggregated errors
    """
    if aggregation_type == "single_layer":
        # Find the last layer (assume it's the one with highest index/name)
        layer_names = list(errors.keys())
        if not layer_names:
            return {}

        transformer_layers = []
        for layer_name in layer_names:
            match = re.search(r"\blayers\.(\d+)\b", layer_name)
            if match:
                transformer_layers.append((int(match.group(1)), layer_name))
        last_layer = max(transformer_layers)[1] if transformer_layers else sorted(layer_names)[-1]

        return {"layer": last_layer, "errors": errors[last_layer]}

    elif aggregation_type == "average":
        # Average errors across all layers
        if not errors:
            return {}

        first_layer_keys = list(errors[list(errors.keys())[0]].keys())
        avg_errors = {}

        for error_type in first_layer_keys:
            values = [errors[layer][error_type] for layer in errors]
            avg_errors[error_type] = float(np.mean(values))

        return {"aggregation": "average", "errors": avg_errors, "num_layers": len(errors)}

    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")


def run_calibration_experiment(
    model,
    tokenizer,
    dataset_name,
    method_type,
    sample_sizes,
    aggregation_type,
    seqlen,
    batchsize,
    seed,
    target_layers=None,
):
    """
    Run calibration experiment for a given dataset and method.

    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset_name: Name of dataset ('gsm8k' or 'magicoder')
        method_type: 'norm' or 'projection'
        sample_sizes: List of sample sizes to test
        aggregation_type: 'average' or 'single_layer'
        seqlen: Sequence length
        batchsize: Batch size
        seed: Random seed
        target_layers: List of specific layer names to use (if None, uses all layers)

    Returns:
        dict: Calibration errors for each sample size
    """
    # Determine max_samples based on dataset
    if dataset_name == "gsm8k":
        # GSM8K train has 7473 samples
        max_samples = 7000  # Use most of the training data as ground truth
    elif dataset_name == "magicoder":
        # Magicoder has 75k samples, use 75k for ground truth
        max_samples = 75000
    else:
        # Default fallback
        max_samples = 2000

    logging.info(f"Auto-determined max_samples={max_samples} for dataset {dataset_name}")
    logging.info(f"Loading {max_samples} samples from {dataset_name} as ground truth")

    # Load maximum samples as ground truth
    all_problems = get_dataset(
        dataset_name=dataset_name,
        num_samples=max_samples,
        tokenizer=tokenizer,
        split="train",
        seed=seed,
    )

    # Compute ground truth metrics with all samples
    all_batches = [all_problems[i : i + batchsize] for i in range(0, len(all_problems), batchsize)]
    true_accumulator = MetricAccumulator(method_type, target_layers=target_layers)

    logging.info(f"Computing ground truth metrics with {len(all_problems)} samples")
    for i, batch_problems in enumerate(all_batches):
        batch = prepare_batch(batch_problems, tokenizer, max_length=seqlen)
        if method_type == "norm":
            metrics = calculate_nfn_scores(model, batch, record_dist=False)
        elif method_type == "projection":
            metrics = calculate_projection_scores(model, batch, record_dist=False)
        true_accumulator.update(metrics)

        if (i + 1) % 10 == 0:
            logging.info(f"Processed batch {i+1}/{len(all_batches)} for ground truth")

    # Average ground truth metrics
    true_metrics = true_accumulator.finalize()

    # Test different sample sizes
    results = {}

    for sample_size in sample_sizes:
        if sample_size > len(all_problems):
            logging.warning(
                f"Sample size {sample_size} > available samples {len(all_problems)}, skipping"
            )
            continue

        logging.info(f"Testing sample size: {sample_size}")

        # Use first sample_size problems
        subset_problems = all_problems[:sample_size]
        subset_batches = [
            subset_problems[i : i + batchsize] for i in range(0, len(subset_problems), batchsize)
        ]

        subset_accumulator = MetricAccumulator(method_type, target_layers=target_layers)
        for batch_problems in subset_batches:
            batch = prepare_batch(batch_problems, tokenizer, max_length=seqlen)
            if method_type == "norm":
                metrics = calculate_nfn_scores(model, batch, record_dist=False)
            elif method_type == "projection":
                metrics = calculate_projection_scores(model, batch, record_dist=False)
            subset_accumulator.update(metrics)

        # Average subset metrics
        estimated_metrics = subset_accumulator.finalize()

        # Compute calibration errors
        if method_type == "projection":
            layer_errors = compute_calibration_error_projection(true_metrics, estimated_metrics)
        elif method_type == "norm":
            layer_errors = compute_calibration_error_norm(true_metrics, estimated_metrics)

        # Aggregate errors
        aggregated_errors = aggregate_errors(layer_errors, aggregation_type)

        results[sample_size] = {
            "layer_errors": layer_errors,
            "aggregated_errors": aggregated_errors,
        }

    return results


def average_metrics_simple(metrics_list, method_type, target_layers=None):
    """Pool per-batch sufficient statistics over all selected tokens."""
    return pool_metrics(metrics_list, method_type, target_layers=target_layers)


def main():
    parser = argparse.ArgumentParser(description="Calibration error experiment")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model handle")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("calibration_results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "magicoder"],
        default="gsm8k",
        help="Dataset to use",
    )
    parser.add_argument(
        "--method_type",
        type=str,
        choices=["norm", "projection"],
        default="projection",
        help="Method type",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["average", "single_layer"],
        default="average",
        help="How to aggregate across layers",
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        help="Sample sizes to test",
    )
    parser.add_argument("--batchsize", type=int, default=8, help="Batch size")
    parser.add_argument("--seqlen", type=int, default=256, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--target_layers",
        type=str,
        nargs="*",
        help="Specific layer names to use for calibration (default: all layers)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from the model repository",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    log_dir = Path(__file__).resolve().parent / "logs" / "calibration"
    setup_logging(log_dir)
    logging.info(f"Arguments: {args}")

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

    # Run calibration experiment
    results = run_calibration_experiment(
        model=model,
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        method_type=args.method_type,
        sample_sizes=args.sample_sizes,
        aggregation_type=args.aggregation,
        seqlen=args.seqlen,
        batchsize=args.batchsize,
        seed=args.seed,
        target_layers=args.target_layers,
    )

    # Save results
    layers_suffix = f"_layers_{len(args.target_layers)}" if args.target_layers else ""
    output_filename = f"{args.model.split('/')[-1]}_{args.dataset}_{args.method_type}_{args.aggregation}_{args.seed}{layers_suffix}_calibration.pt"
    output_path = args.output_dir / output_filename

    torch.save(results, output_path)

    logging.info(f"Calibration results saved to: {output_path}")

    # Print summary
    print(f"\nCalibration Error Summary ({args.aggregation} aggregation):")
    print(f"Dataset: {args.dataset}, Method: {args.method_type}")
    print("-" * 50)

    for sample_size, result in results.items():
        if not result["aggregated_errors"]:
            logging.warning("No matching layers for N=%s; skipping summary", sample_size)
            continue
        agg_errors = result["aggregated_errors"]["errors"]
        if args.method_type == "projection":
            print(
                f"N={sample_size:3d}: Mean Error={agg_errors['mean_error']:.4f}, Var Error={agg_errors['var_error']:.4f}"
            )
        elif args.method_type == "norm":
            print(
                f"N={sample_size:3d}: Mean Error={agg_errors['mean_error']:.4f}, Std Error={agg_errors['std_error']:.4f}"
            )


if __name__ == "__main__":
    main()
