"""Utilities for the learned embedding classifiers used in the paper."""

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.paths import slugify_path_component


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class EmbeddingClassifier(nn.Module):
    """A linear probe or two-layer MLP over frozen LLM embeddings."""

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers: int = 2):
        super().__init__()
        if num_layers == 1:
            self.classifier = nn.Linear(input_dim, num_classes, bias=False)
        elif num_layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")

    def forward(self, inputs):
        return self.classifier(inputs)


def embedding_filename(
    model_name: str,
    task: str,
    seed: int,
    mode: str,
    *,
    seqlen: int = 512,
    all_layers: bool = False,
) -> str:
    """Build the portable filename shared by extraction and training."""
    model_slug = slugify_path_component(model_name.split("/")[-1])
    task_slug = slugify_path_component(task)
    length_suffix = "" if seqlen == 512 else f"_S{seqlen}"
    layer_suffix = "_all" if all_layers else ""
    return f"{model_slug}_{task_slug}_{seed}_{mode}{length_suffix}{layer_suffix}.pt"


def load_train_data(
    baseline_tasks: List[str],
    model_name: str,
    seed: int,
    embed_type: str,
    train_percent: float = 100.0,
    num_layers: Optional[int] = None,
    seqlen: int = 512,
    input_dir: Path = Path("artifacts/embeddings/train"),
):
    if not 0 < train_percent <= 100:
        raise ValueError("train_percent must be in the interval (0, 100]")

    input_dir = Path(input_dir)
    train_embed = []
    train_labels = []
    for idx, task in enumerate(baseline_tasks):
        embed_path = input_dir / embedding_filename(
            model_name,
            task,
            seed,
            "train",
            seqlen=seqlen,
            all_layers=num_layers is not None,
        )
        saved = torch.load(embed_path, map_location="cpu", weights_only=True)
        tmp_embed = saved[embed_type] if num_layers is None else saved[embed_type][num_layers]

        if train_percent < 100:
            count = max(1, int(tmp_embed.shape[0] * train_percent / 100.0))
            tmp_embed = tmp_embed[:count]

        train_embed.append(tmp_embed)
        train_labels.append(torch.full((tmp_embed.shape[0],), idx, dtype=torch.long))

    return torch.cat(train_embed, dim=0).float(), torch.cat(train_labels, dim=0)


def get_method_name(embed_type: str, num_layers: int = 2) -> str:
    prefixes = {1: "Linear:", 2: "MLP:"}
    names = {
        "last_tokens": "Last",
        "avg_tokens": "Avg",
        "weighted_avg_tokens": "WAvg",
    }
    if num_layers not in prefixes:
        raise ValueError(f"Unknown number of classifier layers: {num_layers}")
    if embed_type not in names:
        raise ValueError(f"Unknown embedding type: {embed_type}")
    return prefixes[num_layers] + names[embed_type]


def get_n_layers(model_name: str) -> int:
    """Return layer counts for the models evaluated in the paper."""
    if "Qwen3-1.7B" in model_name:
        return 28
    if "Llama-3.2-3B" in model_name:
        return 28
    if "Llama-3.2-1B" in model_name:
        return 16
    if "Qwen3-8B" in model_name or "Qwen3-4B" in model_name:
        return 36
    if "Qwen3-32B" in model_name:
        return 64
    raise ValueError(
        f"Unknown model type {model_name!r}; pass --model-layers when training the classifier"
    )
