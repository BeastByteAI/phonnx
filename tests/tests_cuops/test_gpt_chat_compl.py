import unittest
from unittest.mock import patch
import numpy as np
import json
from phonnx.cuops.llm.gpt_chat_completion import (
    _prepare_messages,
    gpt_chat_completion_v1_impl,
)


class TestGPTChatCompletion(unittest.TestCase):
    def test_prepare_messages(self):
        messages = np.array(["Hello", "How are you?", "Goodbye"])
        roles = ["user", "assistant", "user"]
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "How are you?"},
            {"role": "user", "content": "Goodbye"},
        ]
        result = _prepare_messages(messages, roles)
        self.assertEqual(result, expected)

    @patch("phonnx.utils.cuops.extract_highest_priority")
    @patch("phonnx.cuops.llm.gpt_chat_completion.RESTClient.make_request")
    def test_gpt_chat_completion_v1_impl_base_case(
        self, mock_make_request, mock_extract_priority
    ):
        s = [
            np.array([["Hello", "Hi"], ["Goodbye", "Bye"]]),
            np.array([""]),
            np.array([""]),
            np.array([""]),
        ]
        mock_extract_priority.return_value = "http://test.url"
        mock_make_request.return_value = {
            "choices": [{"message": {"content": "Test Response"}}]
        }
        expected_output = np.array(
            [["Test Response"], ["Test Response"]], dtype=np.str_
        ).reshape(-1, 1)

        output = gpt_chat_completion_v1_impl(
            s,
            temperature=0.7,
            extra_params_json="",
            model="",
            key="",
            org="",
            url="",
            default_url="",
            roles="user,assistant",
        )
        self.assertTrue(np.array_equal(output, expected_output))

    @patch("phonnx.utils.cuops.extract_highest_priority")
    @patch("phonnx.cuops.llm.gpt_chat_completion.RESTClient.make_request")
    def test_gpt_chat_completion_v1_impl_with_extra_params(
        self, mock_make_request, mock_extract_priority
    ):
        s = [
            np.array([["Hello", "Hi"]]),
            np.array([""]),
            np.array([""]),
            np.array([""]),
        ]
        mock_extract_priority.return_value = "http://test.url"
        mock_make_request.return_value = {
            "choices": [{"message": {"content": "Test Response"}}]
        }
        expected_output = np.array([["Test Response"]], dtype=np.str_).reshape(-1, 1)

        output = gpt_chat_completion_v1_impl(
            s,
            temperature=0.9,
            extra_params_json=json.dumps({"custom_param": "value"}),
            model="gpt-4",
            key="",
            org="",
            url="",
            default_url="",
            roles="user,assistant",
        )
        self.assertTrue(np.array_equal(output, expected_output))

    @patch("phonnx.utils.cuops.extract_highest_priority")
    @patch("phonnx.cuops.llm.gpt_chat_completion.RESTClient.make_request")
    def test_gpt_chat_completion_v1_impl_with_invalid_response(
        self, mock_make_request, mock_extract_priority
    ):
        s = [
            np.array([["Hello", "Hi"]]),
            np.array([""]),
            np.array([""]),
            np.array([""]),
        ]
        mock_extract_priority.return_value = "http://test.url"
        mock_make_request.return_value = {}
        expected_output = np.array([[""]], dtype=np.str_).reshape(-1, 1)

        output = gpt_chat_completion_v1_impl(
            s,
            temperature=0.7,
            extra_params_json="",
            model="",
            key="",
            org="",
            url="",
            default_url="",
            roles="user,assistant",
        )
        self.assertTrue(np.array_equal(output, expected_output))


if __name__ == "__main__":
    unittest.main()
