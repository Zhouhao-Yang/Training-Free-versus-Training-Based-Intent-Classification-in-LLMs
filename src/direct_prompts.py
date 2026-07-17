"""Prompt templates used by the paper's direct causal-LLM baselines."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence


LABEL_ORDER = {
    "L1": ("text", "math", "code"),
    "L1:Adv": ("text", "math", "code"),
    "L2:PLang": (
        "cpp",
        "csharp",
        "java",
        "php",
        "python",
        "rust",
        "shell",
        "swift",
        "typescript",
    ),
    "L2:Math": (
        "Algebra",
        "Counting_&_Probability",
        "Geometry",
        "Intermediate_Algebra",
        "Number_Theory",
        "Prealgebra",
        "Precalculus",
    ),
    "L2:NatLang-5": (
        "Sinhala",
        "Tamil",
        "English",
        "Moroccan_Arabic",
        "Japanese",
    ),
}


L1_PROMPT_TEMPLATE = """
You are a classifier.
Your task is to look at the user's input text and decide which of these three categories it belongs to:

1. General text – natural language content like sentences, questions, explanations, stories, or instructions that are not primarily mathematics or code.
2. Math – text that is primarily mathematical expressions, equations, formulas, or word problems where the main focus is on mathematics.
3. Code – text that is primarily programming code or pseudocode (any programming language, including configuration snippets or shell commands).

Output rules:

- If the input is general text, output: A
- If the input is math, output: B
- If the input is code, output: C

Important Notes:

- Output only a single letter: A, B, or C.
- Do not output anything else (no explanation, no punctuation, no spaces).
{examples_block}
Now classify the following input accordingly and output just one letter.

Input:
{user_text}
Output:
"""


L2_PLANG_PROMPT_TEMPLATE = """
You are a classifier.
Your task is to look at the user's input code snippet and decide which programming language it is written in.

The possible languages are:

1. C++ – output: A
2. C# – output: B
3. Java – output: C
4. PHP – output: D
5. Python – output: E
6. Rust – output: F
7. Shell – output: G
8. Swift – output: H
9. TypeScript – output: I

Output rules:

- Output only a single letter: A, B, C, D, E, F, G, H, or I.
- Do not output anything else (no explanation, no punctuation, no spaces).

Now classify the following input accordingly and output just one letter.

Input:
{user_text}
Output:
"""


L2_MATH_PROMPT_TEMPLATE = """
You are a classifier.
Your task is to look at the user's input math problem and decide which of these seven categories it belongs to:

1. Algebra – problems involving polynomial equations, inequalities, functions, sequences, series, logarithms, and algebraic manipulations at an introductory level.
2. Counting & Probability – problems involving combinatorics, permutations, combinations, probability calculations, and counting principles.
3. Geometry – problems involving shapes, angles, areas, volumes, coordinate geometry, triangles, circles, and geometric proofs.
4. Intermediate Algebra – problems involving more advanced algebraic topics such as complex numbers, polynomial roots, rational functions, systems of equations, and advanced inequalities.
5. Number Theory – problems involving prime numbers, divisibility, modular arithmetic, greatest common divisors, least common multiples, and integer properties.
6. Prealgebra – problems involving basic arithmetic, fractions, decimals, percentages, ratios, and foundational mathematical concepts.
7. Precalculus – problems involving trigonometry, vectors, matrices, conic sections, polar coordinates, and topics that bridge algebra and calculus.

Output rules:

- If the input is Algebra, output: A
- If the input is Counting & Probability, output: B
- If the input is Geometry, output: C
- If the input is Intermediate Algebra, output: D
- If the input is Number Theory, output: E
- If the input is Prealgebra, output: F
- If the input is Precalculus, output: G

Important Notes:

- Output only a single letter: A, B, C, D, E, F, or G.
- Do not output anything else (no explanation, no punctuation, no spaces).

Now classify the following input accordingly and output just one letter.

Input:
{user_text}
Output:
"""


L2_NATLANG5_PROMPT_TEMPLATE = """
You are a classifier.
Your task is to look at the user's input text and decide which natural language it is written in.

The possible languages are:

1. Sinhala – output: A
2. Tamil – output: B
3. English – output: C
4. Moroccan Arabic – output: D
5. Japanese – output: E

Output rules:

- Output only a single letter: A, B, C, D, or E.
- Do not output anything else (no explanation, no punctuation, no spaces).

Now classify the following input accordingly and output just one letter.

Input:
{user_text}
Output:
"""


PROMPT_TEMPLATES = {
    "L1": L1_PROMPT_TEMPLATE,
    "L1:Adv": L1_PROMPT_TEMPLATE,
    "L2:PLang": L2_PLANG_PROMPT_TEMPLATE,
    "L2:Math": L2_MATH_PROMPT_TEMPLATE,
    "L2:NatLang-5": L2_NATLANG5_PROMPT_TEMPLATE,
}


def label_to_letter(setting: str) -> dict[str, str]:
    """Return the paper's prompt-specific class-letter mapping."""

    return {label: chr(ord("A") + index) for index, label in enumerate(LABEL_ORDER[setting])}


def select_examples(
    pools: Mapping[str, Sequence[str]],
    label_letters: Mapping[str, str],
    *,
    query_index: int,
    shots_per_class: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Interleave deterministically shuffled classes, as in the paper code."""

    examples: list[tuple[str, str]] = []
    labels = list(label_letters)
    for shot_index in range(shots_per_class):
        round_labels = labels.copy()
        rng.shuffle(round_labels)
        for label in round_labels:
            pool = pools[label]
            if not pool:
                raise ValueError(f"No in-context examples are available for {label!r}")
            example_index = (query_index + shot_index) % len(pool)
            examples.append((pool[example_index], label_letters[label]))
    return examples


def format_examples(examples: Sequence[tuple[str, str]]) -> str:
    """Format demonstrations for a plain-text few-shot prompt."""

    if not examples:
        return ""
    rendered = [
        f"Example {index}:\nInput:\n{text}\nOutput:\n{letter}\n"
        for index, (text, letter) in enumerate(examples, start=1)
    ]
    return "\nBelow are some classification examples for this task:\n" + "\n".join(rendered)


def render_prompt(setting: str, user_text: str, examples: Sequence[tuple[str, str]]) -> str:
    """Render the paper's descriptive prompt as one text sequence."""

    return PROMPT_TEMPLATES[setting].format(
        user_text=user_text,
        examples_block=format_examples(examples),
    )


def render_chat_messages(
    setting: str,
    user_text: str,
    examples: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    """Render demonstrations as valid alternating chat turns."""

    if not examples:
        return [{"role": "user", "content": render_prompt(setting, user_text, [])}]

    first_text, first_letter = examples[0]
    messages = [
        {
            "role": "user",
            "content": PROMPT_TEMPLATES[setting].format(
                user_text=first_text,
                examples_block="",
            ),
        },
        {"role": "assistant", "content": first_letter},
    ]
    for example_text, example_letter in examples[1:]:
        messages.extend(
            [
                {"role": "user", "content": f"Input:\n{example_text}\nOutput:\n"},
                {"role": "assistant", "content": example_letter},
            ]
        )
    messages.append({"role": "user", "content": f"Input:\n{user_text}\nOutput:\n"})
    return messages
