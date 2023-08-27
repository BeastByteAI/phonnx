import unittest
from phonnx.utils.onnx_parsing import (
    split_node_name,
    determine_column_type,
    is_valid_node_name,
    get_node_type,
    get_layer_id,
    retrieve_inputs,
    get_output_names,
    ColumnTypes,
)


class TestParsingUtils(unittest.TestCase):
    def test_split_node_name(self):
        self.assertEqual(split_node_name("abc_def-ghi"), ["abc", "def", "ghi"])

    def test_determine_column_type(self):
        self.assertEqual(
            determine_column_type("node_pl_0"), ColumnTypes.NUMERIC_REGULAR
        )
        self.assertEqual(determine_column_type("node_pl_1"), ColumnTypes.CAT_LOW_CARD)
        self.assertEqual(
            determine_column_type("falcon_pl_2"), ColumnTypes.CAT_HIGH_CARD
        )
        self.assertEqual(determine_column_type("node_pl_3"), ColumnTypes.TEXT_UTF8)
        self.assertEqual(
            determine_column_type("falcon_pl_100"), ColumnTypes.DATE_YMD_ISO8601
        )
        self.assertEqual(
            determine_column_type("node_pl_101"), ColumnTypes.DATETIME_YMDHMS_ISO8601
        )

    def test_is_valid_node_name(self):
        self.assertTrue(is_valid_node_name("abc-pl-0/foo"))
        self.assertTrue(is_valid_node_name("abc_pl_1/foo"))
        self.assertTrue(is_valid_node_name("abc_pl_0/"))
        self.assertFalse(is_valid_node_name("abc_pl_0"))
        self.assertFalse(is_valid_node_name("abc_pl_a/foo"))
        self.assertFalse(is_valid_node_name("abc_pl_afoo"))

    def test_get_node_type(self):
        self.assertEqual(get_node_type("node_pl_0"), "pl")
        self.assertEqual(get_node_type("node-pl-0"), "pl")
        self.assertEqual(get_node_type("falcon-input-0"), "input")
        self.assertEqual(get_node_type("falcon_input-0"), "input")

    def test_get_layer_id(self):
        self.assertEqual(get_layer_id("node_pl_0"), 0)

    def test_retrieve_inputs(self):
        ort_inputs = [
            "falcon_input_1_abc_0",
            "falcon-input-1-bce-0",
            "falcon-dynattr-4",
        ]
        self.assertEqual(
            retrieve_inputs(ort_inputs),
            (["falcon_input_1_abc_0", "falcon-input-1-bce-0"], ["falcon-dynattr-4"]),
        )

    def test_get_output_names(self):
        self.assertEqual(
            get_output_names(["node_pl_0/a", "node-pl-1/b"], "final"), ["node-pl-1/b"]
        )
        self.assertEqual(
            get_output_names(["node_pl_0/a", "node-pl-1/b"], "all"),
            ["node_pl_0/a", "node-pl-1/b"],
        )

    def test_get_output_names_invalid(self):
        with self.assertRaises(ValueError):
            get_output_names(["node_0"], "invalid")

    def test_get_output_names_invalid_name(self):
        with self.assertRaises(RuntimeError):
            get_output_names(["invalid_name", "another_invalid_name"], "final")


if __name__ == "__main__":
    unittest.main()
