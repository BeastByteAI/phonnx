from typing import Any, Union, Dict, List, Optional
import onnxruntime as ort
import numpy as np

from phonnx.utils.onnx_parsing import (
    get_output_names,
    retrieve_inputs,
    determine_column_type,
)
from phonnx.utils.types import onnx_to_numpy_type
from phonnx.col_preprocessors import MAP


class Runtime:
    """
    Wrapper around Microsoft ONNXRuntime.
    The runtime can only be used for models that follow phonnx naming conventions.
    """

    def __init__(self, model: Union[bytes, str]):
        self.model_path = None
        if isinstance(model, str) and not model.endswith(".onnx"):
            self.model_path = model
        self.ort_session = ort.InferenceSession(model)

    def run(
        self,
        X: Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray],
        dynattrs: Optional[Dict[str, np.ndarray]] = None,
        outputs_to_return: str = "final",
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """
        Runs the model.

        Parameters
        ----------
        X : List[np.ndarray]
            Model inputs as list of numpy arrays
        dynattrs : Dict[str, np.ndarray], optional
            Dynamic attributes, by default None
        outputs_to_return : str, optional
            when set to "all", all onnx output nodes will be returned; when "final" only the last layer outputs are returned, by default "final"

        Returns
        -------
        List[np.ndarray]
            model predictions
        """
        if outputs_to_return not in ["all", "final"]:
            raise ValueError(
                f"Expected `outputs_to_return` to be one of [all, final], got `{outputs_to_return}`."
            )
        if not isinstance(X, (dict, list, np.ndarray)):
            raise TypeError(
                f"Expected `X` to be of type [dict, list], got `{type(X)}`."
            )
        if dynattrs is None:
            dynattrs = {}
        elif not isinstance(dynattrs, dict):
            raise TypeError(
                f"Expected `dynattrs` to be of type [dict, None], got `{type(dynattrs)}`."
            )

        # get the input/output names from the onnx model
        ort_input_names = [inp.name for inp in self.ort_session.get_inputs()]
        ort_output_names = [out.name for out in self.ort_session.get_outputs()]

        # split the input names into inputs and dynamic attributes
        input_names, dynattr_names = retrieve_inputs(ort_input_names)

        # set missing dynamic attributes to empty strings (default value)
        for name in dynattr_names:
            if name not in dynattrs.keys():
                dynattrs[name] = np.array([""], dtype=np.str_)

        # if X is a list, we need to match the inputs to the input names
        if isinstance(X, np.ndarray):
            # split into list across last dimension
            X = np.split(X, X.shape[-1], axis=-1)
        if isinstance(X, list):
            if len(X) != len(input_names):
                raise ValueError(
                    f"Expected `X` to have {len(input_names)} elements, got {len(X)}."
                )
            inputs = dict(zip(input_names, X))

        # if X is a dictionary, we use it as is and add the dynamic attributes
        elif isinstance(X, dict):
            inputs = X

        # preprocess the columns by type
        self._run_type_preprocessing(inputs)

        # verify that dynamic attributes consist of a single element
        self._assert_single_element(dynattrs)

        # add the dynamic attributes to the inputs
        inputs = inputs | dynattrs

        # finalize the inputs
        self._finalize_inputs(inputs)

        # get the final (possibly reduced) list of output names
        output_names = get_output_names(ort_output_names, outputs_to_return)

        # check if all inputs are present
        if len(inputs.keys()) != len(ort_input_names):
            raise ValueError(
                f"Expected {len(ort_input_names)} inputs, got {len(inputs.keys())}."
            )

        return self.ort_session.run(output_names, inputs)

    def _finalize_inputs(self, inputs: Dict[str, np.ndarray]) -> None:
        ort_input_types = {inp.name: inp.type for inp in self.ort_session.get_inputs()}
        for i, t in ort_input_types.items():
            if i not in inputs.keys():
                raise ValueError(f"Input {i} is missing.")
            inputs[i] = inputs[i].astype(onnx_to_numpy_type(t))

    def _assert_single_element(self, inputs: Dict[str, np.ndarray]) -> None:
        for i, v in inputs.items():
            if v.size != 1:
                raise ValueError(f"{i} must have a single element, got {v.shape[0]}.")

    def _run_type_preprocessing(self, inputs: Dict[str, np.ndarray]) -> None:
        for k, v in inputs.items():
            type_ = determine_column_type(k)
            fn = MAP[type_]
            inputs[k] = fn(v)
