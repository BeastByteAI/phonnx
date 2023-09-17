import unittest
import numpy as np
from phonnx.cuops.strings.json_extractor import (
    string_json_label_extractor_v1_impl,
    _RANDOM,
)

_NONE = ""


class TestStringJSONLabelExtractorV1Impl(unittest.TestCase):
    def test_valid_extract(self):
        json_strings = np.array([['{"label": "dog"}'], ['{"label": "cat"}']])
        candidate_labels = np.array(["dog", "cat", "fish"])
        probabilities = np.array([0.5, 0.4, 0.1])

        result = string_json_label_extractor_v1_impl(
            [json_strings, candidate_labels, probabilities],
            default_label=_NONE,
            extract_key="label",
            size=1,
        )

        expected = np.array([["dog"], ["cat"]], dtype=np.str_)
        np.testing.assert_array_equal(result, expected)

    def test_random_label(self):
        json_strings = np.array([['{"label": "elephant"}']])
        candidate_labels = np.array(["dog", "cat", "fish"])
        probabilities = np.array([0.0, 1.0, 0.0])

        result = string_json_label_extractor_v1_impl(
            [json_strings, candidate_labels, probabilities],
            default_label=_RANDOM,
            extract_key="label",
            size=1,
        )

        expected = np.array([["cat"]], dtype=np.str_)
        np.testing.assert_array_equal(result, expected)

    def test_default_label(self):
        json_strings = np.array([['{"label": "elephant"}']])
        candidate_labels = np.array(["dog", "cat", "fish"])
        probabilities = np.array([1.0, 0.0, 0.0])

        result = string_json_label_extractor_v1_impl(
            [json_strings, candidate_labels, probabilities],
            default_label="default",
            extract_key="label",
            size=1,
        )

        expected = np.array([["default"]], dtype=np.str_)
        np.testing.assert_array_equal(result, expected)

    def test_multi_label(self):
        json_strings = np.array(
            [
                ['{"label": ["dog", "cat"]}'],
                ['{"label": ["fish"]}'],
                ['{"label": ["fish", "dog", "cat"]}'],
                ['{"label": something else}'],
            ]
        )
        candidate_labels = np.array(["dog", "cat", "fish"])
        probabilities = np.array([1.0, 0.0, 0.0])

        result = string_json_label_extractor_v1_impl(
            [json_strings, candidate_labels, probabilities],
            default_label=_NONE,
            extract_key="label",
            size=2,
        )

        expected = np.array(
            [["dog", "cat"], ["fish", ""], ["fish", "dog"], ["", ""]], dtype=np.str_
        )
        np.testing.assert_array_equal(result, expected)

    def test_empty_json(self):
        json_strings = np.array([["{}"]])
        candidate_labels = np.array(["dog", "cat", "fish"])
        probabilities = np.array([0.5, 0.4, 0.1])

        result = string_json_label_extractor_v1_impl(
            [json_strings, candidate_labels, probabilities],
            default_label="default",
            extract_key="label",
            size=1,
        )

        expected = np.array([["default"]], dtype=np.str_)
        np.testing.assert_array_equal(result, expected)

    def test_mismatched_candidates_probabilities(self):
        json_strings = np.array([['{"label": "dog"}']])
        candidate_labels = np.array(["dog", "cat"])
        probabilities = np.array([1.0])  # Mismatched size

        with self.assertRaises(AssertionError):
            string_json_label_extractor_v1_impl(
                [json_strings, candidate_labels, probabilities],
                default_label=_NONE,
                extract_key="label",
                size=1,
            )

    def test_missing_label_key(self):
        json_strings = np.array([['{"not_label": "dog"}']])
        candidate_labels = np.array(["dog", "cat", "fish"])
        probabilities = np.array([0.5, 0.4, 0.1])

        result = string_json_label_extractor_v1_impl(
            [json_strings, candidate_labels, probabilities],
            default_label="default",
            extract_key="label",
            size=1,
        )

        expected = np.array([["default"]], dtype=np.str_)
        np.testing.assert_array_equal(result, expected)

    def test_non_string_labels(self):
        json_strings = np.array([['{"label": 42}'], ['{"label": true}']])
        candidate_labels = np.array(["42", "True", "dog"])
        probabilities = np.array([0.4, 0.3, 0.3])

        result = string_json_label_extractor_v1_impl(
            [json_strings, candidate_labels, probabilities],
            default_label=_NONE,
            extract_key="label",
            size=1,
        )

        expected = np.array([["42"], ["True"]], dtype=np.str_)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
