# Adversarial MATH-500 data

This directory contains the local paired data used by the original experiments:

- `math500.json`: the 500 source MATH-500 prompts.
- `adv_math500.json`: the corresponding adversarial rewrites in the same order.
- `prompt_template.txt`: the prompt template used to construct adversarial variants.

The paper's released difficulty-stratified dataset is hosted as
[nanchennn/Adv_MATH500](https://huggingface.co/datasets/nanchennn/Adv_MATH500).
The experiment loader downloads the `easy`, `medium`, and `hard` configurations from that
dataset when the identifiers `adv_math500_easy`, `adv_math500_medium`, or
`adv_math500_hard` are selected.

These bundled JSON files were inherited from the authors' experiment repository. No
standalone data license was present in that source snapshot; users should also follow the
terms of the original MATH-500 dataset.
