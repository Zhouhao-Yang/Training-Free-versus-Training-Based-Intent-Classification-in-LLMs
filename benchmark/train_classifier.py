"""Train and evaluate the paper's linear-probe and MLP baselines."""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from benchmark.utils import (
    EmbeddingClassifier,
    EmbeddingDataset,
    embedding_filename,
    get_method_name,
    get_n_layers,
    load_train_data,
)
from src.helper import ResultsDB, set_seed, setup_logging
from src.paths import slugify_path_component


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "configurations.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model identifier")
    parser.add_argument(
        "--setting",
        required=True,
        choices=["L1", "L1:Adv", "L2:PLang", "L2:Math", "L2:NatLang-5"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embed-type",
        required=True,
        choices=["last_tokens", "avg_tokens", "weighted_avg_tokens"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--num-layers", type=int, choices=[1, 2], default=2, help="Classifier depth"
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-percent", type=float, default=100.0)
    parser.add_argument(
        "--embedding-layer", type=int, default=None, help="Intermediate hidden-state index"
    )
    parser.add_argument(
        "--model-layers", type=int, default=None, help="Override model layer count in records"
    )
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--train-dir", type=Path, default=Path("artifacts/embeddings/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("artifacts/embeddings/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/probes"))
    parser.add_argument("--record-dir", type=Path, default=Path("test_records"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.train_percent <= 100:
        raise ValueError("--train-percent must be in the interval (0, 100]")

    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = json.load(f)[args.setting]
    baseline_tasks = list(config["training set"])
    label_to_idx = {task: idx for idx, task in enumerate(baseline_tasks)}

    set_seed(args.seed)
    setup_logging(REPO_ROOT / "logs" / "benchmark")
    logging.info("Arguments: %s", args)
    logging.info("Classes: %s", baseline_tasks)

    train_embeds, train_labels = load_train_data(
        baseline_tasks=baseline_tasks,
        model_name=args.model,
        seed=args.seed,
        embed_type=args.embed_type,
        train_percent=args.train_percent,
        num_layers=args.embedding_layer,
        seqlen=args.seqlen,
        input_dir=args.train_dir,
    )
    train_dataset = EmbeddingDataset(train_embeds, train_labels)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingClassifier(
        input_dim=train_embeds.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=len(baseline_tasks),
        num_layers=args.num_layers,
    ).to(device)

    decay_params, no_decay_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (
            no_decay_params if len(parameter.shape) == 1 or name.endswith(".bias") else decay_params
        ).append(parameter)
    optimizer = optim.Adam(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for embeddings, labels in train_dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(embeddings), labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        logging.info(
            "Epoch %d/%d - loss %.6f",
            epoch + 1,
            args.epochs,
            epoch_loss / max(len(train_dataloader), 1),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = slugify_path_component(args.model.split("/")[-1])
    setting_slug = slugify_path_component(args.setting)
    method_name = get_method_name(args.embed_type, args.num_layers)
    checkpoint_path = args.output_dir / (
        f"{model_slug}_{setting_slug}_{slugify_path_component(method_name)}_S{args.seed}.pt"
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": baseline_tasks,
            "embed_type": args.embed_type,
            "classifier_layers": args.num_layers,
            "hidden_dim": args.hidden_dim,
        },
        checkpoint_path,
    )
    logging.info("Saved classifier to %s", checkpoint_path)

    if args.model_layers is not None:
        recorded_layers = args.model_layers
    elif args.embedding_layer is not None:
        recorded_layers = args.embedding_layer
    else:
        try:
            recorded_layers = get_n_layers(args.model)
        except ValueError:
            metadata_path = args.train_dir / embedding_filename(
                args.model,
                baseline_tasks[0],
                args.seed,
                "train",
                seqlen=args.seqlen,
            )
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=True).get(
                "_metadata", {}
            )
            if "num_hidden_layers" not in metadata:
                raise ValueError("Pass --model-layers for an unrecognized model")
            recorded_layers = int(metadata["num_hidden_layers"])

    record_db = args.record_dir / f"{setting_slug}_record.duckdb"
    db = ResultsDB(record_db)
    model.eval()
    with torch.no_grad():
        for task, label, _available in config["test set"]:
            embed_path = args.test_dir / embedding_filename(
                args.model,
                task,
                args.seed,
                "test",
                seqlen=args.seqlen,
                all_layers=args.embedding_layer is not None,
            )
            saved = torch.load(embed_path, map_location="cpu", weights_only=True)
            test_embed = (
                (
                    saved[args.embed_type]
                    if args.embedding_layer is None
                    else saved[args.embed_type][args.embedding_layer]
                )
                .float()
                .to(device)
            )
            logits = model(test_embed)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == label_to_idx[label]).float().mean()

            db.log(
                model=args.model,
                dataset=task,
                method=method_name,
                n_layers=recorded_layers,
                seqlen=args.seqlen,
                seed=args.seed,
                accuracy=accuracy,
                n_samples=len(predictions),
                batchsize=args.batchsize,
                nbsamples=len(train_dataset),
            )
            logging.info("Task %s (%s): accuracy %.4f", task, label, accuracy)


if __name__ == "__main__":
    main()
