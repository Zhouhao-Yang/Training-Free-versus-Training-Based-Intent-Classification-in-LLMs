# Training-Free versus Training-Based Intent Classification in LLMs: Accuracy, Robustness, and Failure Modes

Official code release for the COLM 2026 paper **"Training-Free versus Training-Based Intent Classification in LLMs: Accuracy, Robustness, and Failure Modes"**.

**Nan Chen\***, **Zhouhao Yang\***, and **Soufiane Hayou**<br>
Department of Applied Mathematics and Statistics, Johns Hopkins University<br>
\* Equal contribution

The paper is accepted at the **Conference on Language Modeling (COLM) 2026**.

## Overview

Intent classification assigns a prompt to a predefined class - such as mathematics, code, or general text - before generation. This repository compares classifiers built from pretrained language-model representations:

- **NormStat** and **VecStat** estimate class-level activation statistics from calibration prompts without gradient updates.
- **Avg-MLP**, **Tail-MLP**, and **Avg-Linear** fit a small classifier to frozen LLM embeddings.
- A RoBERTa classifier and zero-/few-shot causal-LLM evaluator are included as additional baselines.

| Method | CLI value | Prompt summary per monitored module | Comparison |
|---|---|---|---|
| **NormStat** | `--method_type norm` | Mean and variance of token-level activation norms | One-dimensional Gaussian KL |
| **VecStat** | `--method_type projection` | Coordinate-wise mean and variance | Diagonal-Gaussian KL or cosine similarity |

For $m$ classes and representation width $d$, the paper analyzes class-statistic storage of $O(m)$ for NormStat and $O(md)$ for VecStat. Both reuse features from the LLM prefill pass and leave the underlying model unchanged.

## Main findings

- Training-free and training-based methods often approach saturation on coarse general-text vs. mathematics vs. code classification.
- Training-based classifiers are generally stronger on fine-grained distinctions. NormStat struggles when classes differ mainly by direction rather than activation scale, such as programming-language identification.
- Training-free methods are generally more stable on mixed-intent and adversarially rephrased prompts. VecStat provides the strongest mixed-intent uncertainty estimates in the reported experiment.
- No method dominates every regime; the hardest adversarial MATH-500 tier remains challenging for all lightweight classifiers.

## Installation

Run commands from the repository root.

```bash
git clone https://github.com/Zhouhao-Yang/Training-Free-versus-Training-Based-Intent-Classification-in-LLMs.git
cd Training-Free-versus-Training-Based-Intent-Classification-in-LLMs

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

The activation-statistics and embedding-extraction workflows are intended for a CUDA-capable NVIDIA GPU. Model weights and most datasets are downloaded on first use. Authenticate with Hugging Face when using gated models, and review each model and dataset's terms.

The paper's reference environment used Python 3.12.0, PyTorch 2.7.0, and CUDA 12.6. Models up to 4B parameters were evaluated on an NVIDIA L40S (48 GB); the 8B and 32B models were evaluated on a GH200 system with an H100 GPU (96 GB). These are reference configurations, not minimum requirements. The repository's syntax/portability checks also run on Python 3.10.

## Tasks and datasets

| Setting | Task | Classes | Calibration data | Evaluation data |
|---|---|---|---|---|
| `L1` | General intent | Text, math, code | MMLU European History, GSM8K, Magicoder | MMLU US History, GSM8K, MATH-500, Magicoder, HumanEval |
| `L2:PLang` | Programming language | 9 languages | Magicoder | Held-out Magicoder |
| `L2:NatLang-5` | Natural language | 5 languages | Aya | Held-out Aya |
| `L2:Math` | Mathematical subfield | 7 subfields | Competition Math | Held-out Competition Math |

The five natural-language classes are Sinhala, Tamil, English, Moroccan Arabic, and Japanese. The programming-language classes are C++, C#, Java, PHP, Python, Rust, Shell, Swift, and TypeScript. The mathematical subfields are Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, and Precalculus.

The loaders use:

- `cais/mmlu`
- OpenAI's `grade-school-math` JSONL data
- `HuggingFaceH4/MATH-500`
- `ise-uiuc/Magicoder-OSS-Instruct-75K`
- `openai_humaneval`
- `qwedsacf/competition_math`
- `CohereLabs/aya_dataset`

An expanded 38-language Aya configuration is available as `L2:NatLang`, separate from the five-language task reported in the main experiments.

Adversarial evaluation uses the `easy`, `medium`, and `hard` configurations of [nanchennn/Adv_MATH500](https://huggingface.co/datasets/nanchennn/Adv_MATH500). Paired local prompts and the generation template are documented in [`adv_datasets/`](adv_datasets/README.md). The paper also studies constructed mixed math/code prompts; this release does not include a standalone mixed-intent generator.

## Training-free workflow

### 1. Estimate class anchors

NormStat on the `L1` setting:

```bash
python baselines.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --setting L1 \
  --output_dir anchors \
  --method_type norm \
  --nbsamples 2000 \
  --batchsize 8 \
  --seqlen 512 \
  --seed 42
```

`--nbsamples` is an upper bound per class. If a dataset contains fewer examples, all available examples are used.

### 2. Classify evaluation prompts

```bash
python classifier.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --setting L1 \
  --dataset math500 \
  --label math \
  --baseline_dir anchors \
  --output_dir results \
  --method_type norm \
  --method KL \
  --nbsamples 500 \
  --batchsize 8 \
  --seqlen 512 \
  --seed 42
```

For VecStat, use `--method_type projection` in both commands. With VecStat, `--method KL` selects diagonal-Gaussian KL and `--method mean` selects cosine similarity. NormStat uses KL in the paper's main experiments.

Model, setting, seed, method type, and `--train-percent` must match between anchor estimation and classification. Fine-grained dataset selectors include a class after a colon, for example `magicoder:python`, `comp_math:Number_Theory`, or `aya:Moroccan_Arabic`; pass the corresponding class to `--label`. Artifact filenames are sanitized automatically for Windows.

The configuration file supplies the paper's default calibration counts for fine-grained evaluation. Use `--nb_trainsamples` only when anchors were estimated with a different absolute count.

## Training-based LLM probes

The learned-probe workflow first caches frozen LLM embeddings, then trains and evaluates a classifier head.

### 1. Extract calibration and evaluation embeddings

```bash
python -m benchmark.extract_embeddings \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --setting L1 \
  --mode train \
  --nbsamples 2000 \
  --batchsize 8 \
  --seqlen 512 \
  --seed 42

python -m benchmark.extract_embeddings \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --setting L1 \
  --mode test \
  --batchsize 8 \
  --seqlen 512 \
  --seed 42
```

Embeddings are written to `artifacts/embeddings/train/` and `artifacts/embeddings/test/`. Add `--all` to save selected intermediate layers, optionally controlled with `--layer-indices 4,8,12`.

### 2. Fit and evaluate a probe

```bash
python -m benchmark.train_classifier \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --setting L1 \
  --embed-type avg_tokens \
  --num-layers 2 \
  --epochs 10 \
  --seqlen 512 \
  --seed 42
```

| Paper name | `--embed-type` | `--num-layers` |
|---|---|---:|
| Avg-MLP | `avg_tokens` | `2` |
| Tail-MLP | `last_tokens` | `2` |
| Avg-Linear | `avg_tokens` | `1` |

Trained heads are saved under `artifacts/probes/`. Embeddings can consume substantial disk space, particularly for large models and multiple seeds.

## Other baselines

### RoBERTa encoder

```bash
python train_encoder.py \
  --base-model roberta-base \
  --setting L1 \
  --batch-size 32 \
  --eval-batch-size 256 \
  --epochs 1 \
  --seed 42
```

Use `--bf16 --tf32 --optim adamw_torch_fused` on compatible NVIDIA hardware to match the reduced-precision/fused configuration. The portable defaults use standard AdamW and no persistent DataLoader workers.

### Zero- and three-shot causal LLM

```bash
python evaluate_direct_llm.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --setting L1 \
  --chat-model \
  --num-of-shots 0 \
  --nbsamples 100 \
  --batchsize 8 \
  --seed 42
```

For the paper's three-shot-per-class baseline, change `--num-of-shots 0` to
`--num-of-shots 3`. This supplies three demonstrations from each L1 class (nine
demonstrations in total), following the original experiment code. Few-shot evaluation is
defined for `L1` and `L1:Adv`; the fine-grained settings use zero-shot prompts.

The descriptive class prompts and class-to-letter mappings reported in the paper are kept
in `src/direct_prompts.py`. In particular, L1 maps general text, mathematics, and code to
`A`, `B`, and `C`; this prompt-specific order is intentionally separate from the training
configuration's internal label IDs. The evaluator records raw outputs, parsed predictions,
accuracy, and a DuckDB experiment row. Qwen thinking mode is disabled by default, matching
the paper; `--enable-thinking` opts into it and generally requires a larger
`--max-new-tokens` value. Increase `--seqlen` if custom demonstrations are longer than the
default 4096-token prompt budget.

## Calibration convergence

`calibration.py` compares estimates from smaller subsets against a larger reference estimate:

```bash
python calibration.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset magicoder \
  --method_type projection \
  --aggregation average \
  --sample_sizes 512 1024 2048 4096 \
  --output_dir calibration_results \
  --batchsize 8 \
  --seqlen 512 \
  --seed 42
```

This is more expensive than the quickstart: it first computes a large reference estimate and then recomputes each requested subset.



## Citation

```bibtex
@inproceedings{chen2026trainingfree,
  title     = {Training-Free versus Training-Based Intent Classification in {LLM}s: Accuracy, Robustness, and Failure Modes},
  author    = {Chen, Nan and Yang, Zhouhao and Hayou, Soufiane},
  booktitle = {Conference on Language Modeling ({COLM})},
  year      = {2026}
}
```

