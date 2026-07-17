import unittest

from src.paths import slugify_path_component


class SlugifyPathComponentTests(unittest.TestCase):
    def test_replaces_colons_and_spaces(self):
        self.assertEqual(slugify_path_component("L2:Natural Language"), "L2_Natural_Language")

    def test_keeps_portable_identifiers(self):
        self.assertEqual(slugify_path_component("Qwen3-1.7B"), "Qwen3-1.7B")

    def test_rejects_empty_components(self):
        with self.assertRaises(ValueError):
            slugify_path_component(":::")


if __name__ == "__main__":
    unittest.main()
