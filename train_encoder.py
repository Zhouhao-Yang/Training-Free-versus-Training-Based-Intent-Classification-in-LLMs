import argparse
import json
import logging
from pathlib import Path
from copy import deepcopy
from functools import partial

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evaluate
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from src.encoder_utils import construct_dataset, prompt_tokenization, compute_metrics
from src.helper import setup_logging, set_seed
from src.paths import slugify_path_component


REPO_ROOT = Path(__file__).resolve().parent
with (REPO_ROOT / "configs" / "configurations.json").open(encoding="utf-8") as f:
    CONFIGURATION_MAP = json.load(f)


def main(args):
    config = deepcopy(CONFIGURATION_MAP[args.setting])
    config["seed"] = args.seed

    # load dataset
    raw_data, test_set_name = construct_dataset(config)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"FacebookAI/{args.base_model}")

    data_preprocess = partial(
        prompt_tokenization, processing_class=tokenizer, max_seq_length=args.seqlen
    )
    tokenized_dataset = raw_data.map(
        data_preprocess,
        batched=True,
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # load metric
    metric = evaluate.load("accuracy")

    # construct model
    encoder_config = AutoConfig.from_pretrained(
        f"FacebookAI/{args.base_model}",
        num_labels=len(config["label2id"]),
    )
    encoder_config.label2id = config["label2id"]
    encoder_config.id2label = {idx: l for l, idx in config["label2id"].items()}

    base_model = AutoModelForSequenceClassification.from_pretrained(
        f"FacebookAI/{args.base_model}",
        config=encoder_config,
    )

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim=args.optim,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        save_strategy="no",
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=False,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_num_workers=args.num_workers,
        label_names=["labels"],
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="no",
        report_to=["wandb"] if args.enable_log else [],
        disable_tqdm=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=base_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset={d_name: tokenized_dataset[d_name] for d_name in test_set_name},
        processing_class=tokenizer,
        compute_metrics=partial(compute_metrics, metric=metric),
    )

    trainer.train()

    final_metric = trainer.evaluate()

    # save results
    result_dict = dict(
        lr=args.learning_rate,
        seed=args.seed,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
    )

    for name in test_set_name:
        result_dict[name] = final_metric[f"eval_{name}_accuracy"]

    with args.result_pth.open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-model", type=str, choices=["roberta-base", "roberta-large"], default="roberta-base"
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=["L1", "L2:PLang", "L2:Math", "L2:NatLang-5", "L1:Adv"],
        default="L1",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--enable-log", action="store_true")
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 training on supported GPUs"
    )
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 on supported NVIDIA GPUs")
    parser.add_argument("--optim", default="adamw_torch", help="Transformers optimizer name")
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    setting_slug = slugify_path_component(args.setting)
    args.save_dir = REPO_ROOT / "checkpoints" / args.base_model / setting_slug
    args.save_dir.mkdir(parents=True, exist_ok=True)

    lr_tag = f"lr{args.learning_rate:.0e}"
    args.result_pth = args.save_dir / f"{setting_slug}_{lr_tag}_S{args.seed}.json"

    args.log_dir = REPO_ROOT / "logs" / args.base_model / setting_slug
    setup_logging(args.log_dir)
    logging.info(args)

    set_seed(args.seed, deterministic=False)

    main(args)
