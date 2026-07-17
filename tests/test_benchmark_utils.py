import ast
import unittest
from pathlib import Path


def load_get_n_layers():
    source = Path("benchmark/utils.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_n_layers"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), source, "exec"), namespace)
    return namespace["get_n_layers"]


get_n_layers = load_get_n_layers()


class ModelLayerTests(unittest.TestCase):
    def test_llama_32_sizes_have_distinct_layer_counts(self):
        self.assertEqual(get_n_layers("meta-llama/Llama-3.2-1B-Instruct"), 16)
        self.assertEqual(get_n_layers("meta-llama/Llama-3.2-3B-Instruct"), 28)


if __name__ == "__main__":
    unittest.main()
