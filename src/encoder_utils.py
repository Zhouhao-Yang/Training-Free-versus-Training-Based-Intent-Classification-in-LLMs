from src.data import get_dataset
from datasets import Dataset, DatasetDict
from transformers import EvalPrediction
import numpy as np


def construct_dataset(config):
    dataset_collection = {}

    # loading training data
    training_data = []
    for task, dataset_name in config["training set"].items():
        problems = get_dataset(
            dataset_name=dataset_name,
            num_samples=config["train_nbsamples"],
            tokenizer=None,
            split="train",
            seed=config["seed"],
        )
        label = config["label2id"][task]

        for p in problems:
            training_data.append({"prompt": str(p), "label": label})

    dataset_collection["train"] = Dataset.from_list(training_data).shuffle(seed=config["seed"])

    # load evaluation data
    test_set_name = []
    for (dataset_name, task, nbsamples) in config["test set"]:
        test_data = []
        problems = get_dataset(
            dataset_name, num_samples=nbsamples, tokenizer=None, split="test", seed=config["seed"]
        )
        label = config["label2id"][task]

        for p in problems:
            test_data.append({"prompt": str(p), "label": label})

        collection_name = f"{dataset_name}"
        dataset_collection[collection_name] = Dataset.from_list(test_data)
        test_set_name.append(collection_name)

    raw_data = DatasetDict(dataset_collection)

    return raw_data, test_set_name


def prompt_tokenization(
    batch,
    processing_class,
    max_seq_length: int = 512,
):
    result = processing_class(
        batch["prompt"], truncation=True, padding=False, max_length=max_seq_length
    )
    result["labels"] = batch["label"]
    return result


def compute_metrics(eval_pred: EvalPrediction, metric):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    result = metric.compute(predictions=preds, references=labels)
    return result
