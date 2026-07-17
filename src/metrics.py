import torch
from collections import defaultdict
import numpy as np
from math import sqrt
from typing import List
from functools import partial
import re

_layer_idx_re = re.compile(r"\blayers\.(\d+)\b")


def filter_layers(
    name: str,
    allowed_layers: List[int] = None,
    allowed_non_transformers_modules: bool = True,
) -> bool:
    m = _layer_idx_re.search(name)
    if m is None:
        return allowed_non_transformers_modules

    idx = int(m.group(1))
    if allowed_layers is None:
        return True
    return idx in allowed_layers


def masked_mean(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(data.dtype)
    count = mask.sum(dim=-1).clamp(min=1)  # (N,)
    mean = (data * mask).sum(dim=-1) / count
    return mean


def masked_std(data: torch.Tensor, mask: torch.Tensor, masked_mean: torch.Tensor) -> torch.Tensor:
    diff = torch.where(mask, data - masked_mean.unsqueeze(-1), torch.zeros_like(data))

    count = mask.to(data.dtype).sum(dim=-1).clamp(min=1)

    var_samp = diff.pow(2).sum(dim=-1) / (count - 1).clamp(min=1)

    return var_samp.sqrt()


def calculate_nfn_scores(
    model,
    batch,
    record_dist: bool = False,
    mode: str = "train",
    allowed_layers: List[int] = None,
    legacy_norm_width: bool = False,
):
    """
    Calculate NFN scores for all weight matrices.
    Args:
        model: Model to calculate NFN scores for.
        batch: Batch of problems.
        record_dist: Whether to record the distribution of norms. This is used for analysis, not recommended for use case inference.
        mode: Mode of NFN calculation.
    Returns:
        Dictionary of NFN scores for all weight matrices.
    """
    # Move batch to GPU if needed
    if next(model.parameters()).device != batch["input_ids"].device:
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}

    # Initialize metrics dictionary
    metrics = defaultdict(dict)

    if mode == "train":
        flat_mask = batch["attention_mask"].reshape(-1).bool()
    else:
        flat_mask = batch["attention_mask"].bool()

    l_filter_fn = partial(filter_layers, allowed_layers=allowed_layers)

    # Define hook function to calculate NFN scores for each weight matrix
    def hook_fn(name):
        """
        Hook function to calculate NFN scores for each weight matrix.
        Args:
            name: Name of the weight matrix.
        Returns:
            Hook function to calculate NFN scores for each weight matrix.
        """
        # Define inner hook function to calculate NFN scores for each weight matrix
        def hook(module, input, output):
            """
            Inner hook function to calculate NFN scores for each weight matrix.
            Args:
                module: Module to calculate NFN scores for.
                input: Input to the module.
                output: Output from the module (won't be used here).
            """
            if hasattr(module, "weight") and module.weight is not None:
                # Get input and weight matrices
                z = input[0] if isinstance(input, tuple) else input
                W = module.weight
                z = z.float()  # B, N, D_in
                W = W.float()  # (D_in, D_out)

                # Calculate NFN scores
                try:
                    # We calculate the Frobenius norm of W to normalize W for stability, but it is not necessary.
                    W_norm = (W**2).mean().sqrt()
                    z_norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)  # B, N, 1
                    W_normalized = W / (W_norm + 1e-8)
                    z_normalized = z / (z_norm + 1e-8)
                    Wz = torch.matmul(z_normalized, W_normalized.t())  # B, N, D_out
                    norms = torch.norm(Wz, dim=-1)  # B, N
                    # The submitted experiment code used the input width here. The
                    # paper defines the statistic on Wz, whose output width is the
                    # appropriate scale. Keep the historical behavior opt-in so old
                    # table-generation runs can still be replicated.
                    feature_scale = sqrt(z.shape[-1] if legacy_norm_width else Wz.shape[-1])

                    if mode == "train":
                        norms = norms.view(-1)  # B * N
                        valid_norms = norms[flat_mask] / feature_scale
                        metrics[name]["count"] = valid_norms.numel()
                        metrics[name]["sum"] = valid_norms.sum().item()
                        metrics[name]["sumsq"] = valid_norms.square().sum().item()
                        metrics[name]["mean"] = valid_norms.mean().item()
                        metrics[name]["std"] = valid_norms.std().item()
                        if record_dist:
                            metrics[name]["dist"] = valid_norms.cpu().numpy()
                    else:
                        tmp_mean = masked_mean(norms, flat_mask)
                        tmp_std = masked_std(norms, flat_mask, tmp_mean)
                        metrics[name]["mean"] = tmp_mean / feature_scale
                        metrics[name]["std"] = tmp_std / feature_scale
                except RuntimeError as e:
                    print(f"Error in layer {name}:")
                    print(f"Input shape: {z.shape}")
                    print(f"Weight shape: {W.shape}")
                    raise e

        return hook

    hooks = []
    for name, module in model.named_modules():
        embedding_filter = isinstance(module, torch.nn.Embedding)
        ln_filter = isinstance(module, torch.nn.LayerNorm) or "norm" in name.lower()
        if (
            hasattr(module, "weight")
            and (module.weight is not None)
            and (not embedding_filter)
            and (not ln_filter)
            and l_filter_fn(name)
        ):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    model.eval()
    try:
        with torch.no_grad():
            _ = model(**batch)
    finally:
        for hook in hooks:
            hook.remove()

    return metrics


def calculate_projection_scores(
    model,
    batch,
    record_dist: bool = False,
    mode: str = "train",
    allowed_layers: List[int] = None,
):
    """
    Calculate projection-based scores for all weight matrices using Method 2.
    Args:
        model: Model to calculate projection scores for.
        batch: Batch of problems.
        record_dist: Whether to record the distribution of projections.
        mode: Mode of calculation.
        allowed_layers: List of allowed layer indices.
    Returns:
        Dictionary of projection scores for all weight matrices.
    """
    # Move batch to GPU if needed
    if next(model.parameters()).device != batch["input_ids"].device:
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}

    # Initialize metrics dictionary
    metrics = defaultdict(dict)

    if mode == "train":
        flat_mask = batch["attention_mask"].reshape(-1).bool()
    else:
        flat_mask = batch["attention_mask"].float().unsqueeze(-1)

    l_filter_fn = partial(filter_layers, allowed_layers=allowed_layers)

    def hook_fn(name):
        def hook(module, input, output):
            if hasattr(module, "weight") and module.weight is not None:
                # Get input and weight matrices
                z = input[0] if isinstance(input, tuple) else input
                W = module.weight
                z = z.float()  # B, N, D_in
                W = W.float()  # (D_in, D_out)

                try:
                    # Normalization (same as Method 1)
                    W_norm = (W**2).mean().sqrt()
                    z_norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)  # B, N, 1
                    W_normalized = W / (W_norm + 1e-8)
                    z_normalized = z / (z_norm + 1e-8)

                    # Projection computation: Wz_l = z_normalized @ W_normalized^T
                    Wz = torch.matmul(z_normalized, W_normalized.t())  # B, N, D_out

                    if mode == "train":
                        Wz = Wz.view(-1, Wz.shape[-1])  # (B*N), D_out
                        Wz_masked = Wz[flat_mask]  # (valid_tokens), D_out

                        metrics[name]["count"] = Wz_masked.shape[0]
                        metrics[name]["sum_vec"] = Wz_masked.sum(dim=0).cpu()
                        metrics[name]["sumsq_vec"] = Wz_masked.square().sum(dim=0).cpu()

                        # Average vector computation
                        mean_vec = Wz_masked.mean(dim=0)  # D_out

                        # Variance computation (diagonal of covariance matrix only)
                        centered = Wz_masked - mean_vec  # (valid_tokens), D_out
                        variance_vec = centered.square().sum(dim=0) / max(Wz_masked.shape[0] - 1, 1)

                        metrics[name]["mean_vec"] = mean_vec.cpu()
                        metrics[name]["var_vec"] = variance_vec.cpu()

                        if record_dist:
                            metrics[name]["projections"] = Wz_masked.cpu().numpy()

                    else:
                        mean_vecs = torch.sum(Wz * flat_mask, dim=1) / torch.sum(flat_mask, dim=1)
                        metrics[name]["mean_vec"] = mean_vecs

                        centered = Wz - mean_vecs.unsqueeze(1)
                        centered_squared = centered**2
                        metrics[name]["var_vec"] = torch.sum(
                            centered_squared * flat_mask, dim=1
                        ) / (torch.sum(flat_mask, dim=1) - 1).clamp(min=1)

                except RuntimeError as e:
                    print(f"Error in layer {name}:")
                    print(f"Input shape: {z.shape}")
                    print(f"Weight shape: {W.shape}")
                    raise e

        return hook

    hooks = []
    for name, module in model.named_modules():
        embedding_filter = isinstance(module, torch.nn.Embedding)
        ln_filter = isinstance(module, torch.nn.LayerNorm) or "norm" in name.lower()
        if (
            hasattr(module, "weight")
            and (module.weight is not None)
            and (not embedding_filter)
            and (not ln_filter)
            and l_filter_fn(name)
        ):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    try:
        with torch.no_grad():
            _ = model(**batch)
    finally:
        for hook in hooks:
            hook.remove()

    return metrics


def infer_task_from_projection_scores(
    dataset_scores, baseline_scores_dict, keys=None, threshold=1.1, method="mean"
):
    """
    Compare dataset projection scores to baseline scores for each task and infer the closest task.
    Args:
        dataset_scores: dict of projection scores for the dataset
        baseline_scores_dict: dict mapping task name to baseline projection score dict
        keys: list of keys to compare (if None, use intersection of all keys)
        threshold: threshold for distance (not used in projection methods)
        method: method to use for distance calculation: 'mean' (cosine similarity), 'KL' (dimension-wise KL divergence)
    Returns:
        closest_task: task name with smallest distance (or highest similarity for 'mean')
        distances: dict of task -> distance/similarity
    """
    if keys is None:
        keys = list(dataset_scores.keys())

    device = dataset_scores[keys[0]]["mean_vec"].device
    distances = []

    for task in baseline_scores_dict.keys():
        baseline = baseline_scores_dict[task]

        if method == "mean":
            # Cosine similarity approach
            similarities = []
            for k in keys:
                # Get mean vectors
                vec1 = dataset_scores[k]["mean_vec"]  # B, D_out or D_out
                vec2 = torch.tensor(baseline[k]["mean_vec"]).to(device)  # D_out

                if vec1.dim() == 1:  # Training mode
                    vec1 = vec1.unsqueeze(0)  # 1, D_out
                if vec2.dim() == 1:
                    vec2 = vec2.unsqueeze(0)  # 1, D_out

                # Compute cosine similarity
                vec1_norm = torch.norm(vec1, dim=-1, keepdim=True)  # B, 1
                vec2_norm = torch.norm(vec2, dim=-1, keepdim=True)  # 1, 1

                denominator = (vec1_norm.squeeze(-1) * vec2_norm.squeeze(-1)).clamp_min(1e-8)
                cosine_sim = torch.sum(vec1 * vec2, dim=-1) / denominator  # B
                similarities.append(cosine_sim)

            # Average similarity across layers
            avg_similarity = torch.stack(similarities, dim=0).mean(dim=0)  # B
            distances.append(-avg_similarity)  # Convert to distance (negative similarity)

        elif method == "KL":
            # Dimension-wise KL divergence approach
            kl_divs = []
            for k in keys:
                # Get mean and variance vectors
                vec1_mean = dataset_scores[k]["mean_vec"]  # B, D_out or D_out
                vec1_var = dataset_scores[k]["var_vec"]  # B, D_out or D_out
                vec2_mean = torch.tensor(baseline[k]["mean_vec"]).to(device)  # D_out
                vec2_var = torch.tensor(baseline[k]["var_vec"]).to(device)  # D_out

                if vec1_mean.dim() == 1:  # Training mode
                    vec1_mean = vec1_mean.unsqueeze(0)  # 1, D_out
                    vec1_var = vec1_var.unsqueeze(0)  # 1, D_out
                if vec2_mean.dim() == 1:
                    vec2_mean = vec2_mean.unsqueeze(0)  # 1, D_out
                    vec2_var = vec2_var.unsqueeze(0)  # 1, D_out

                # Dimension-wise KL divergence
                vec1_std = torch.sqrt(vec1_var + 1e-8)  # B, D_out
                vec2_std = torch.sqrt(vec2_var + 1e-8)  # 1, D_out

                kl_dim = (
                    torch.log(vec1_std / vec2_std)
                    + (vec2_var + (vec1_mean - vec2_mean) ** 2) / (2 * vec1_var + 1e-8)
                    - 0.5
                )  # B, D_out

                # Average over dimensions for each layer
                kl_layer = kl_dim.mean(dim=-1)  # B
                kl_divs.append(kl_layer)

            # Sum across layers
            total_kl = torch.stack(kl_divs, dim=0).sum(dim=0)  # B
            distances.append(total_kl)

        else:
            raise ValueError(f"Invalid method: {method}")

    distances = torch.stack(distances, dim=1)  # B, num_tasks

    if method == "mean":
        # For cosine similarity (stored as negative), find minimum distance (maximum similarity)
        closest_task = torch.argmin(distances, dim=1)  # B
    else:
        # For KL divergence, find minimum distance
        closest_task = torch.argmin(distances, dim=1)  # B

    return closest_task, distances


def infer_task_from_scores(
    dataset_scores, baseline_scores_dict, keys=None, threshold=1.1, method="mean"
):
    """
    Compare dataset scores to baseline scores for each task and infer the closest task.
    Args:
        dataset_scores: dict of scores for the dataset (e.g., aggregated by type)
        baseline_scores_dict: dict mapping task name to baseline score dict (same format as dataset_scores)
        keys: list of keys to compare (if None, use intersection of all keys)
        threshold: threshold for distance
        method: method to use for distance calculation: 'mean' or 'KL'
    Returns:
        closest_task: task name with smallest distance
        distances: dict of task -> distance
    """
    if keys is None:
        # Use intersection of all layer keys
        keys = list(dataset_scores.keys())

    device = dataset_scores[keys[0]]["mean"].device

    distances = []
    for task in baseline_scores_dict.keys():
        baseline = baseline_scores_dict[task]
        if method == "mean":
            vec1 = torch.stack([dataset_scores[k][method] for k in keys], dim=0)  # L, B
            vec2 = (
                torch.tensor([baseline[k][method] for k in keys]).unsqueeze(-1).to(device)
            )  # L, 1
            dist = torch.mean((vec1 > threshold) * (vec1 - vec2) ** 2, dim=0)  # B
        elif method == "KL":
            vec1_mean = torch.stack([dataset_scores[k]["mean"] for k in keys], dim=0)  # L, B
            vec2_mean = (
                torch.tensor([baseline[k]["mean"] for k in keys]).unsqueeze(-1).to(device)
            )  # L, 1
            vec1_std = torch.stack([dataset_scores[k]["std"] for k in keys], dim=0).clamp_min(
                1e-8
            )  # L, B
            vec2_std = (
                torch.tensor([baseline[k]["std"] for k in keys])
                .unsqueeze(-1)
                .to(device)
                .clamp_min(1e-8)
            )  # L, 1

            dist = torch.sum(
                torch.log(vec1_std / vec2_std)
                + (vec2_std**2 + (vec1_mean - vec2_mean) ** 2) / vec1_std**2 / 2
                - 1 / 2,
                dim=0,
            )  # B
        else:
            raise ValueError(f"Invalid method: {method}")
        distances.append(dist)

    distances = torch.stack(distances, dim=1)  # B, 3
    closest_task = torch.argmin(distances, dim=1)  # B,

    return closest_task, distances


def infer_task_probs_from_scores(
    dataset_scores, baseline_scores_dict, keys=None, temperature=1.0, threshold=1.1
):
    """
    Frequentist approach: Convert distances to a probability distribution over tasks using softmax over negative distances.
    Args:
        dataset_scores: dict of scores for the dataset (e.g., aggregated by type)
        baseline_scores_dict: dict mapping task name to baseline score dict (same format as dataset_scores)
        keys: list of keys to compare (if None, use intersection of all keys)
        temperature: softmax temperature (default 1.0)
    Returns:
        probs_dict: dict of task -> probability
        distances: dict of task -> distance
    """
    import numpy as np

    if keys is None:
        # key_sets = [set(scores.keys()) for scores in baseline_scores_dict.values()]# + [set(dataset_scores.keys())]
        # keys = set.intersection(*key_sets)
        keys = dataset_scores.keys()
    distances = {}
    for task, baseline in baseline_scores_dict.items():
        vec1 = np.array([dataset_scores[k]["actual"] for k in keys])
        vec2 = np.array([baseline[k]["actual"] for k in keys])
        dist = np.sum((vec1 - vec2) ** 2 * (vec1 > threshold)) / len(keys)
        distances[task] = dist
    # Softmax over negative distances
    task_list = list(distances.keys())
    logits = -np.array([distances[task] for task in task_list]) / temperature
    exp_logits = np.exp(logits - np.max(logits))  # numerical stability
    probs = exp_logits / exp_logits.sum()
    probs_dict = dict(zip(task_list, probs))
    return probs_dict, distances
