import unittest
from unittest.mock import patch, Mock
import numpy as np
from phonnx.runtime import Runtime


class TestRuntime(unittest.TestCase):
    @patch("phonnx.runtime.ort.InferenceSession")
    def setUp(self, mock_ort_session):
        mock_ort_session.return_value = Mock()
        input1 = Mock()
        input1.name = "phonnx_input_1"
        input1.type = "tensor(float)"
        input2 = Mock()
        input2.name = "phonnx-dynattr-1"
        input2.type = "tensor(string)"
        output1 = Mock()
        output1.name = "phonnx_pl_1/"
        output1.type = "tensor(float)"
        mock_ort_session.return_value.get_inputs.return_value = [input1, input2]
        mock_ort_session.return_value.get_outputs.return_value = [output1]
        mock_ort_session.run.return_value = [np.array([1, 2, 3])]
        self.model_path = "fake_model.onnx"
        self.runtime = Runtime(self.model_path)
        self.ort_session = mock_ort_session.return_value
        self.ort_session.run.return_value = [np.array([1, 2, 3])]

    def test_run(self):
        X = [np.array([1, 2, 3])]
        dynattrs = {"phonnx-dynattr-1": np.array(["some_str"])}

        result = self.runtime.run(X, dynattrs, "final")

        self.assertTrue((result[0] == np.array([1, 2, 3])).all())

    def test__finalize_inputs(self):
        inputs = {
            "phonnx_input_1": np.array([1, 2, 3], dtype=np.float32),
            "phonnx-dynattr-1": np.array(["some str"], dtype=np.str_),
        }

        self.runtime._finalize_inputs(inputs)
        self.assertTrue(np.issubdtype(inputs["phonnx_input_1"].dtype, np.float32))

    def test__assert_single_element(self):
        inputs = {"input_1": np.array([1])}
        self.runtime._assert_single_element(inputs)

        inputs = {"input_1": np.array([1, 2])}
        with self.assertRaises(ValueError):
            self.runtime._assert_single_element(inputs)

    def test__run_type_preprocessing(self):
        # default case
        inputs = {"node_pl_0": np.array([1, 2, 3])}
        self.runtime._run_type_preprocessing(inputs)
        self.assertEqual(inputs["node_pl_0"].shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
