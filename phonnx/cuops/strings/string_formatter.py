from onnxruntime_extensions import onnx_op, PyCustomOpDef
from string import Template
from typing import List
import numpy as np
from phonnx.constants import AI_BEASTBYTE_PYOPS

OP_NAME = "StringTemplateFormatter"


@onnx_op(
    op_type=f"{AI_BEASTBYTE_PYOPS}::{OP_NAME}V1",
    inputs=[
        PyCustomOpDef.dt_string,  # input_data
    ],
    attrs={
        "template_str": PyCustomOpDef.dt_string,  # Template string for formatting
        "variables": PyCustomOpDef.dt_string,  # Comma separated variable names
    },
    outputs=[PyCustomOpDef.dt_string],
)
def string_template_formatter_v1(*s, **kwargs):
    template_str = kwargs.get("template_str", "")
    variables = kwargs.get("variables", "")

    return string_template_formatter_v1_impl(s[0], template_str, variables)


def string_template_formatter_v1_impl(
    input_data: np.ndarray, template_str: str, variables: str
):
    variable_list = variables.split(",")
    assert input_data.ndim == 2, "input_data must be a 2D tensor"
    assert input_data.shape[1] == len(
        variable_list
    ), "Mismatch in number of variables and input_data shape"

    t = Template(template_str)
    formatted_strings = []

    for row in input_data:
        var_dict = {var: val for var, val in zip(variable_list, row)}
        formatted_str = t.substitute(var_dict)
        formatted_strings.append(formatted_str)

    return np.array(formatted_strings, dtype=np.str_).reshape(-1, 1)
