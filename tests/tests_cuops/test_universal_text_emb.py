import unittest
import numpy as np
from unittest.mock import patch
from phonnx.cuops.llm.universal_text_embedding import universal_embedding_v1_impl


class TestUniversalEmbedding(unittest.TestCase):
    @patch("phonnx.cuops.llm.universal_text_embedding.RESTClient.make_request")
    def test_single_prompt(self, mock_make_request):
        mock_make_request.return_value = {"some": {"nested": {"key": [0.2, 0.4]}}}

        s = [
            np.array([["Hello world"]]),  # prompt
            np.array(["http://example.com"]),  # url
            np.array(["key"]),
            np.array(["org"]),
        ]

        extra_params_json = ""
        key = "key"
        org = "org"
        url = "http://example.com"
        default_url = "http://default.com"
        header_key_auth = "Authorization"
        header_key_org = "Organization"
        prompt_location = "inputs,text"
        embedding_location = "some,nested,key"
        custom_header = ""
        batch_size = 0

        output = universal_embedding_v1_impl(
            s,
            extra_params_json,
            key,
            org,
            url,
            default_url,
            header_key_auth,
            header_key_org,
            prompt_location,
            embedding_location,
            custom_header,
            batch_size,
        )

        expected_output = np.array([[0.2, 0.4]], dtype=np.float32)

        np.testing.assert_array_almost_equal(output, expected_output, decimal=5)

    @patch("phonnx.cuops.llm.universal_text_embedding.RESTClient.make_request")
    def test_batched_prompts(self, mock_make_request):
        mock_make_request.return_value = {
            "embeddings": [{"key": [0.2, 0.4]}, {"key": [0.3, 0.5]}]
        }

        s = [
            np.array([["Hello"], ["World"]]),  # batched prompts
            np.array(["http://example.com"]),  # url
            np.array(["key"]),
            np.array(["org"]),
        ]

        extra_params_json = ""
        batch_size = 2
        prompt_location = "<<batch_dim>>,inputs,text"
        embedding_location = "embeddings,<<batch_dim>>,key"

        output = universal_embedding_v1_impl(
            s,
            extra_params_json,
            "key",
            "org",
            "http://example.com",
            "http://default.com",
            "Authorization",
            "Organization",
            prompt_location,
            embedding_location,
            "",
            batch_size,
        )

        expected_output = np.array([[0.2, 0.4], [0.3, 0.5]], dtype=np.float32)

        np.testing.assert_array_almost_equal(output, expected_output, decimal=5)

    @patch("phonnx.cuops.llm.universal_text_embedding.RESTClient.make_request")
    def test_empty_response(self, mock_make_request):
        mock_make_request.return_value = {}
        s = [
            np.array([["Hello world"]]),
            np.array(["http://example.com"]),
            np.array(["key"]),
            np.array(["org"]),
        ]
        extra_params_json = ""
        with self.assertRaises(ValueError):
            universal_embedding_v1_impl(
                s,
                extra_params_json,
                "key",
                "org",
                "http://example.com",
                "http://default.com",
                "Authorization",
                "Organization",
                "inputs,text",
                "some,nested,key",
                "",
                0,
            )


if __name__ == "__main__":
    unittest.main()
