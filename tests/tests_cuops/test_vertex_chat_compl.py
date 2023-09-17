import unittest
from unittest.mock import patch
import numpy as np
from phonnx.cuops.llm.vertex_chat_completion import (
    vertex_chat_completion_v1_impl,
    _prepare_messages,
)


class TestVertexChatCompletionV1(unittest.TestCase):
    @patch("phonnx.cuops.llm.vertex_chat_completion.RESTClient.make_request")
    def test_valid_single_message(self, mock_make_request):
        mock_make_request.return_value = {
            "predictions": [{"candidates": [{"content": "Hello, world!"}]}]
        }
        messages = np.array([["Hi"]], dtype=np.str_).reshape(-1, 1)
        roles = "user"
        temperature = 0.7
        maxOutputTokens = 4096
        topP = 0.95
        topK = 40
        key = "test_key"
        url = "test_url"
        default_url = "default_test_url"
        expected_result = np.array(["Hello, world!"], dtype=np.str_).reshape(-1, 1)

        result = vertex_chat_completion_v1_impl(
            [messages, np.array([url]), np.array([key])],
            temperature,
            maxOutputTokens,
            topP,
            topK,
            key,
            url,
            default_url,
            roles,
        )
        np.testing.assert_equal(result, expected_result)

    @patch("phonnx.cuops.llm.vertex_chat_completion.RESTClient.make_request")
    def test_empty_response(self, mock_make_request):
        mock_make_request.return_value = {}
        messages = np.array([["Hi"]], dtype=np.str_).reshape(-1, 1)
        roles = "user"
        temperature = 0.7
        maxOutputTokens = 4096
        topP = 0.95
        topK = 40
        key = "test_key"
        url = "test_url"
        default_url = "default_test_url"
        expected_result = np.array([""], dtype=np.str_).reshape(-1, 1)

        result = vertex_chat_completion_v1_impl(
            [messages, np.array([url]), np.array([key])],
            temperature,
            maxOutputTokens,
            topP,
            topK,
            key,
            url,
            default_url,
            roles,
        )
        np.testing.assert_equal(result, expected_result)

    def test_single_message(self):
        messages = np.array(["Hello"], dtype=np.str_)
        roles = ["user"]
        expected_result = {
            "instances": [{"messages": [{"author": "user", "content": "Hello"}]}]
        }
        result = _prepare_messages(messages, roles)
        self.assertEqual(result, expected_result)

    def test_multiple_messages(self):
        messages = np.array(["Hello", "Hi"], dtype=np.str_)
        roles = ["user", "bot"]
        expected_result = {
            "instances": [
                {
                    "messages": [
                        {"author": "user", "content": "Hello"},
                        {"author": "bot", "content": "Hi"},
                    ]
                }
            ]
        }
        result = _prepare_messages(messages, roles)
        self.assertEqual(result, expected_result)

    def test_with_context(self):
        messages = np.array(["Hello", "Hi"], dtype=np.str_)
        roles = ["user", "context"]
        expected_result = {
            "instances": [
                {"messages": [{"author": "user", "content": "Hello"}], "context": "Hi"}
            ]
        }
        result = _prepare_messages(messages, roles)
        self.assertEqual(result, expected_result)

    def test_with_examples(self):
        messages = np.array(["Hello", "Hi", "Hey", "Hola"], dtype=np.str_)
        roles = ["user", "example_in", "bot", "example_out"]
        expected_result = {
            "instances": [
                {
                    "messages": [
                        {"author": "user", "content": "Hello"},
                        {"author": "bot", "content": "Hey"},
                    ],
                    "examples": [
                        {"input": {"content": "Hi"}, "output": {"content": "Hola"}}
                    ],
                }
            ]
        }
        result = _prepare_messages(messages, roles)
        self.assertEqual(result, expected_result)

    def test_invalid_role(self):
        messages = np.array(["Hello"], dtype=np.str_)
        roles = ["invalid"]
        with self.assertRaises(ValueError):
            _prepare_messages(messages, roles)


if __name__ == "__main__":
    unittest.main()
