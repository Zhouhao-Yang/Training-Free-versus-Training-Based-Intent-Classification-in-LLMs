import random
import unittest

from src.direct_prompts import label_to_letter, render_chat_messages, render_prompt, select_examples


class DirectPromptTests(unittest.TestCase):
    def test_l1_uses_paper_letter_mapping(self):
        self.assertEqual(label_to_letter("L1"), {"text": "A", "math": "B", "code": "C"})

    def test_three_shots_means_three_examples_per_class(self):
        letters = label_to_letter("L1")
        pools = {label: [f"{label}-{index}" for index in range(4)] for label in letters}
        examples = select_examples(
            pools,
            letters,
            query_index=0,
            shots_per_class=3,
            rng=random.Random(42),
        )
        self.assertEqual(len(examples), 9)
        self.assertEqual({letter for _, letter in examples}, {"A", "B", "C"})

    def test_descriptions_and_chat_turns_are_retained(self):
        prompt = render_prompt("L1", "query", [("demo", "A")])
        self.assertIn("General text", prompt)
        self.assertIn("Example 1", prompt)
        messages = render_chat_messages("L1", "query", [("demo", "A")])
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user"])


if __name__ == "__main__":
    unittest.main()
