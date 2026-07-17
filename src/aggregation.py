"""Online pooling utilities for statistics computed over mini-batches."""

from typing import Iterable, Optional


class MetricAccumulator:
    """Pool sufficient statistics without retaining every batch in memory."""

    def __init__(
        self,
        method_type: str,
        *,
        record_dist: bool = False,
        target_layers: Optional[Iterable[str]] = None,
    ):
        if method_type not in {"norm", "projection"}:
            raise ValueError(f"Unknown method type: {method_type}")
        self.method_type = method_type
        self.record_dist = record_dist
        self.target_layers = set(target_layers) if target_layers is not None else None
        self._states = {}

    def update(self, metrics):
        if self.method_type == "projection":
            import torch
        if self.record_dist:
            import numpy as np

        for key, batch in metrics.items():
            if self.target_layers is not None and key not in self.target_layers:
                continue

            count = int(batch["count"])
            if count == 0:
                continue

            if key not in self._states:
                if self.method_type == "norm":
                    self._states[key] = {
                        "count": 0,
                        "sum": 0.0,
                        "sumsq": 0.0,
                        "dist": [],
                    }
                else:
                    self._states[key] = {
                        "count": 0,
                        "sum_vec": torch.zeros_like(torch.as_tensor(batch["sum_vec"]).cpu()),
                        "sumsq_vec": torch.zeros_like(torch.as_tensor(batch["sumsq_vec"]).cpu()),
                        "projections": [],
                    }

            state = self._states[key]
            state["count"] += count
            if self.method_type == "norm":
                state["sum"] += float(batch["sum"])
                state["sumsq"] += float(batch["sumsq"])
                if self.record_dist:
                    state["dist"].append(np.asarray(batch["dist"]))
            else:
                state["sum_vec"].add_(torch.as_tensor(batch["sum_vec"]).cpu())
                state["sumsq_vec"].add_(torch.as_tensor(batch["sumsq_vec"]).cpu())
                if self.record_dist:
                    state["projections"].append(np.asarray(batch["projections"]))

    def finalize(self):
        if self.record_dist:
            import numpy as np

        pooled = {}
        for key in sorted(self._states):
            state = self._states[key]
            count = state["count"]
            if self.method_type == "norm":
                total = state["sum"]
                total_sq = state["sumsq"]
                mean = total / count
                variance = max(0.0, (total_sq - total * total / count) / max(count - 1, 1))
                pooled[key] = {"mean": mean, "std": variance**0.5}
                if self.record_dist:
                    pooled[key]["dist"] = np.concatenate(state["dist"]).tolist()
            else:
                total = state["sum_vec"]
                total_sq = state["sumsq_vec"]
                mean = total / count
                variance = (total_sq - total.square() / count) / max(count - 1, 1)
                pooled[key] = {
                    "mean_vec": mean.tolist(),
                    "var_vec": variance.clamp_min(0).tolist(),
                }
                if self.record_dist:
                    pooled[key]["projections"] = np.concatenate(
                        state["projections"], axis=0
                    ).tolist()
        return pooled


def pool_metrics(
    metrics_iterable,
    method_type: str,
    *,
    record_dist: bool = False,
    target_layers: Optional[Iterable[str]] = None,
):
    """Pool an iterable of per-batch sufficient statistics."""
    accumulator = MetricAccumulator(
        method_type,
        record_dist=record_dist,
        target_layers=target_layers,
    )
    for metrics in metrics_iterable:
        accumulator.update(metrics)
    return accumulator.finalize()


def average_batch_metrics(metrics_iterable, method_type: str, *, record_dist: bool = False):
    """Reproduce the submitted code's unweighted average over mini-batches.

    This compatibility path intentionally gives a short final mini-batch the same
    weight as a full batch. New runs should use :func:`pool_metrics` instead.
    """

    metrics_list = list(metrics_iterable)
    if not metrics_list:
        return {}
    if method_type not in {"norm", "projection"}:
        raise ValueError(f"Unknown method type: {method_type}")
    if method_type == "projection":
        import torch
    if record_dist:
        import numpy as np

    keys = sorted(set().union(*(metrics.keys() for metrics in metrics_list)))
    averaged = {}
    for key in keys:
        batches = [metrics[key] for metrics in metrics_list if key in metrics]
        if method_type == "norm":
            averaged[key] = {
                "mean": sum(float(batch["mean"]) for batch in batches) / len(batches),
                "std": sum(float(batch["std"]) for batch in batches) / len(batches),
            }
            if record_dist:
                averaged[key]["dist"] = np.concatenate(
                    [np.asarray(batch["dist"]) for batch in batches]
                ).tolist()
        else:
            mean_vecs = torch.stack([torch.as_tensor(batch["mean_vec"]).cpu() for batch in batches])
            var_vecs = torch.stack([torch.as_tensor(batch["var_vec"]).cpu() for batch in batches])
            averaged[key] = {
                "mean_vec": mean_vecs.mean(dim=0).tolist(),
                "var_vec": var_vecs.mean(dim=0).tolist(),
            }
            if record_dist:
                averaged[key]["projections"] = np.concatenate(
                    [np.asarray(batch["projections"]) for batch in batches], axis=0
                ).tolist()
    return averaged
