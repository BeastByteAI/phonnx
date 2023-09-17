import unittest
import numpy as np
from phonnx.cuops.strings.string_formatter import string_template_formatter_v1_impl


class TestStringTemplateFormatterV1(unittest.TestCase):
    def test_basic_functionality(self):
        input_data = np.array([["John", "25"], ["Emily", "30"]], dtype=np.str_)
        template_str = "Name: $name, Age: $age"
        variables = "name,age"

        expected_output = np.array(
            [["Name: John, Age: 25"], ["Name: Emily, Age: 30"]], dtype=np.str_
        )
        output = string_template_formatter_v1_impl(input_data, template_str, variables)

        np.testing.assert_array_equal(output, expected_output)

    def test_empty_input(self):
        input_data = np.array([["", ""], ["", ""]], dtype=np.str_)
        template_str = "Name: $name, Age: $age"
        variables = "name,age"

        expected_output = np.array(
            [["Name: , Age: "], ["Name: , Age: "]], dtype=np.str_
        )
        output = string_template_formatter_v1_impl(input_data, template_str, variables)

        np.testing.assert_array_equal(output, expected_output)

    def test_mismatch_shape(self):
        input_data = np.array(
            [["John", "25", "USA"], ["Emily", "30", "Canada"]], dtype=np.str_
        )
        template_str = "Name: $name, Age: $age"
        variables = "name,age"

        with self.assertRaises(AssertionError):
            string_template_formatter_v1_impl(input_data, template_str, variables)

    def test_single_row(self):
        input_data = np.array([["John", "25"]], dtype=np.str_)
        template_str = "Name: $name, Age: $age"
        variables = "name,age"

        expected_output = np.array([["Name: John, Age: 25"]], dtype=np.str_)
        output = string_template_formatter_v1_impl(input_data, template_str, variables)

        np.testing.assert_array_equal(output, expected_output)


if __name__ == "__main__":
    unittest.main()
