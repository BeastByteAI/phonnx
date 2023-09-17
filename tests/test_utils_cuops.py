import unittest
from os import environ
from phonnx.utils.cuops import (
    extract_from_env,
    extract_highest_priority,
    create_nested_dict,
    create_batched_nested_dict,
    extract_value_from_nested_dict,
    batch_generator,
    prepare_request_info,
    extract_json_key_from_string,
)
import numpy as np
import json


class TestUtilsCuops(unittest.TestCase):
    def test_extract_from_env(self):
        environ["TEST_ENV"] = "123"
        self.assertEqual(extract_from_env("env://TEST_ENV"), "123")
        self.assertIsNone(extract_from_env("env://NOT_EXIST"))

    def test_extract_highest_priority(self):
        environ["TEST_ENV"] = "123"
        self.assertEqual(extract_highest_priority("", "env://TEST_ENV"), "123")
        self.assertEqual(extract_highest_priority("456", "env://TEST_ENV"), "456")

    def test_create_nested_dict(self):
        self.assertEqual(create_nested_dict(["a"], "value"), {"a": "value"})
        self.assertEqual(create_nested_dict(["a", "b"], "value"), {"a": {"b": "value"}})
        self.assertEqual(
            create_nested_dict(["a[]", "b"], "value"), {"a": [{"b": "value"}]}
        )
        self.assertEqual(
            create_nested_dict(["a", "b[]"], "value"), {"a": {"b": ["value"]}}
        )

    def test_create_batched_nested_dict(self):
        self.assertEqual(create_batched_nested_dict(["<<batch_dim>>"], [1, 2]), [1, 2])
        self.assertEqual(
            create_batched_nested_dict(["a", "<<batch_dim>>"], [1, 2]), {"a": [1, 2]}
        )
        self.assertEqual(
            create_batched_nested_dict(["a", "<<batch_dim>>", "b"], [1, 2]),
            {"a": [{"b": 1}, {"b": 2}]},
        )

        self.assertEqual(
            create_batched_nested_dict(["a", "<<batch_dim>>", "b[]"], [1, 2]),
            {"a": [{"b": [1]}, {"b": [2]}]},
        )

    def test_extract_value_from_nested_dict(self):
        d = {"a": {"b": {"c": "value"}}}
        self.assertEqual(extract_value_from_nested_dict(d, ["a", "b", "c"]), "value")
        self.assertEqual(extract_value_from_nested_dict(d, ["a", "b", "d"]), "")

        d = {"a": {"b": [{"c": "value"}]}}
        self.assertEqual(
            extract_value_from_nested_dict(d, ["a", "b", "0", "c"]), "value"
        )

    def test_batch_generator(self):
        batches = list(batch_generator([1, 2, 3, 4, 5], 2))
        self.assertEqual(batches, [[1, 2], [3, 4], [5]])

        batches = list(batch_generator([1, 2, 3, 4, 5], 0))
        self.assertEqual(batches, [1, 2, 3, 4, 5])

        batches = list(batch_generator([1, 2, 3, 4, 5], 1))
        self.assertEqual(batches, [[1], [2], [3], [4], [5]])

        batches = list(batch_generator([1, 2, 3, 4, 5], 5))
        self.assertEqual(batches, [[1, 2, 3, 4, 5]])

        batches = list(batch_generator([1, 2, 3, 4, 5], 999))
        self.assertEqual(batches, [[1, 2, 3, 4, 5]])

    def test_prepare_request_info(self):
        environ["URL_ATTR"] = "url_env"
        environ["PHONNX_TEST_KEY"] = "key_value"
        environ["PHONNX_TEST_ORG"] = "org_value"

        headers = prepare_request_info(
            np.array("url_in"),
            np.array("key_in"),
            np.array("org_in"),
            "env://URL_ATTR",
            "default_url",
            "env://PHONNX_TEST_KEY",
            "env://PHONNX_TEST_ORG",
            "header_key_auth",
            "header_key_org",
        )
        expected_headers = {
            "final_url": "url_in",
            "headers": {
                "Content-Type": "application/json",
                "header_key_auth": "key_in",
                "header_key_org": "org_in",
            },
        }
        self.assertEqual(headers, expected_headers)

        headers = prepare_request_info(
            np.array(""),
            np.array(""),
            np.array(""),
            "env://URL_ATTR",
            "default_url",
            "env://PHONNX_TEST_KEY",
            "env://PHONNX_TEST_ORG",
            "header_key_auth",
            "header_key_org",
        )
        expected_headers = {
            "final_url": "url_env",
            "headers": {
                "Content-Type": "application/json",
                "header_key_auth": "key_value",
                "header_key_org": "org_value",
            },
        }
        self.assertEqual(headers, expected_headers)

        headers = prepare_request_info(
            np.array(""),
            np.array(""),
            np.array(""),
            "env://URL_ATTR",
            "default_url",
            "env://PHONNX_TEST_KEY",
            "env://PHONNX_TEST_ORG",
            "header_key_auth",
            "header_key_org",
            '{"Content-Type": "image/jpeg"}',
        )
        expected_headers = {
            "final_url": "url_env",
            "headers": {
                "Content-Type": "image/jpeg",
                "header_key_auth": "key_value",
                "header_key_org": "org_value",
            },
        }
        self.assertEqual(headers, expected_headers)


class TestExtractJSONKeyFromString(unittest.TestCase):
    def test_valid_json(self):
        self.assertEqual(
            extract_json_key_from_string('Some text {"key": "value"} here', "key"),
            "value",
        )

    def test_invalid_json(self):
        self.assertIsNone(
            extract_json_key_from_string('Some text {key: "value"} here', "key")
        )

    def test_no_json(self):
        self.assertIsNone(extract_json_key_from_string("Some text here", "key"))

    def test_key_not_found(self):
        self.assertIsNone(
            extract_json_key_from_string(
                'Some text {"another_key": "value"} here', "key"
            )
        )

    def test_single_quotes(self):
        self.assertEqual(
            extract_json_key_from_string("Some text {'key': 'value'} here", "key"),
            "value",
        )

    def test_nested_json(self):
        d = {"outer": {"inner": "v"}}
        self.assertEqual(
            extract_json_key_from_string(json.dumps(d), "outer"),
            None,
        )

    def test_empty_string(self):
        self.assertIsNone(extract_json_key_from_string("", "key"))


if __name__ == "__main__":
    unittest.main()
