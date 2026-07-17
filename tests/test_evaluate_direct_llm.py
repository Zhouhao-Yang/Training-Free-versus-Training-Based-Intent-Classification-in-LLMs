import ast
import re
import unittest
from pathlib import Path


def load_parse_prediction():
    source = Path("evaluate_direct_llm.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_prediction"
    )
    namespace = {"re": re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), source, "exec"), namespace)
    return namespace["parse_prediction"]


parse_prediction = load_parse_prediction()


class ParsePredictionTests(unittest.TestCase):
    def test_finds_valid_letter_after_explanation(self):
        self.assertEqual(parse_prediction("I think B", {"A", "B", "C"}), "B")

    def test_uses_answer_letter_in_nine_class_setting(self):
        valid_letters = set("ABCDEFGHI")
        self.assertEqual(parse_prediction("I think E", valid_letters), "E")

    def test_ignores_standalone_invalid_letters(self):
        self.assertEqual(parse_prediction("X, so B", {"A", "B", "C"}), "B")


if __name__ == "__main__":
    unittest.main()
