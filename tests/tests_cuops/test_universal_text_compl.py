import unittest
from unittest.mock import patch
import numpy as np
from phonnx.cuops.llm.universal_text_completion import universal_completion_v1_impl


class TestUniversalCompletionV1(unittest.TestCase):
    @patch("phonnx.cuops.llm.universal_text_completion.RESTClient.make_request")
    def test_valid_single_prompt(self, mock_make_request):
        mock_make_request.return_value = {"0": "Hello, world!"}
        prompts = np.array([["What is"]], dtype=np.str_).reshape(-1, 1)
        extra_params_json = "{}"
        key = "test_key"
        org = "test_org"
        url = "test_url"
        default_url = "default_test_url"
        header_key_auth = "Authorization"
        header_key_org = "Org"
        prompt_location = "prompt"
        completion_location = "0"
        custom_header = ""
        expected_result = np.array(["Hello, world!"], dtype=np.str_).reshape(-1, 1)
        result = universal_completion_v1_impl(
            [prompts, np.array([url]), np.array([key]), np.array([org])],
            extra_params_json,
            key,
            org,
            url,
            default_url,
            header_key_auth,
            header_key_org,
            prompt_location,
            completion_location,
            custom_header,
        )
        np.testing.assert_equal(result, expected_result)

    @patch("phonnx.cuops.llm.universal_text_completion.RESTClient.make_request")
    def test_valid_multiple_prompts(self, mock_make_request):
        mock_make_request.return_value = {"0": "Hello, world!"}
        prompts = np.array([["What is"], ["How are"]], dtype=np.str_).reshape(-1, 1)
        extra_params_json = "{}"
        key = "test_key"
        org = "test_org"
        url = "test_url"
        default_url = "default_test_url"
        header_key_auth = "Authorization"
        header_key_org = "Org"
        prompt_location = "prompt"
        completion_location = "0"
        custom_header = ""
        expected_result = np.array(
            ["Hello, world!", "Hello, world!"], dtype=np.str_
        ).reshape(-1, 1)
        result = universal_completion_v1_impl(
            [prompts, np.array([url]), np.array([key]), np.array([org])],
            extra_params_json,
            key,
            org,
            url,
            default_url,
            header_key_auth,
            header_key_org,
            prompt_location,
            completion_location,
            custom_header,
        )
        np.testing.assert_equal(result, expected_result)

    @patch("phonnx.cuops.llm.universal_text_completion.RESTClient.make_request")
    def test_empty_response(self, mock_make_request):
        mock_make_request.return_value = {}
        prompts = np.array([["What is"]], dtype=np.str_).reshape(-1, 1)
        extra_params_json = "{}"
        key = "test_key"
        org = "test_org"
        url = "test_url"
        default_url = "default_test_url"
        header_key_auth = "Authorization"
        header_key_org = "Org"
        prompt_location = "prompt"
        completion_location = "0"
        custom_header = ""
        expected_result = np.array([""], dtype=np.str_).reshape(-1, 1)
        result = universal_completion_v1_impl(
            [prompts, np.array([url]), np.array([key]), np.array([org])],
            extra_params_json,
            key,
            org,
            url,
            default_url,
            header_key_auth,
            header_key_org,
            prompt_location,
            completion_location,
            custom_header,
        )
        np.testing.assert_equal(result, expected_result)

    @patch("phonnx.cuops.llm.universal_text_completion.RESTClient.make_request")
    def test_custom_header(self, mock_make_request):
        mock_make_request.return_value = {"0": "Hello, world!"}
        prompts = np.array([["What is"]], dtype=np.str_).reshape(-1, 1)
        extra_params_json = "{}"
        key = "test_key"
        org = "test_org"
        url = "test_url"
        default_url = "default_test_url"
        header_key_auth = "Authorization"
        header_key_org = "Org"
        prompt_location = "prompt"
        completion_location = "0"
        custom_header = '{"Test-Header": "Test-Value"}'
        expected_result = np.array(["Hello, world!"], dtype=np.str_).reshape(-1, 1)
        result = universal_completion_v1_impl(
            [prompts, np.array([url]), np.array([key]), np.array([org])],
            extra_params_json,
            key,
            org,
            url,
            default_url,
            header_key_auth,
            header_key_org,
            prompt_location,
            completion_location,
            custom_header,
        )
        np.testing.assert_equal(result, expected_result)
        mock_make_request.assert_called_with(
            url,
            {"prompt": "What is"},
            {
                "Authorization": "test_key",
                "Org": "test_org",
                "Test-Header": "Test-Value",
            },
        )
