from onnxruntime_extensions import onnx_op, PyCustomOpDef
from phonnx.constants import AI_BEASTBYTE_PYOPS
from phonnx.utils.cuops import extract_json_key_from_string
from typing import List
import numpy as np

OP_NAME = "StringJSONLabelExtractor"

_RANDOM = "<<RANDOM>>"


@onnx_op(
    op_type=f"{AI_BEASTBYTE_PYOPS}::{OP_NAME}V1",
    inputs=[
        PyCustomOpDef.dt_string,  # json_strings
        PyCustomOpDef.dt_string,  # candidate labels
        PyCustomOpDef.dt_float,  # probabilities
    ],
    attrs={
        # <<RANDOM>> is a special values
        "default_label": PyCustomOpDef.dt_string,
        "extract_key": PyCustomOpDef.dt_string,
        "size": PyCustomOpDef.dt_string,
    },
    outputs=[PyCustomOpDef.dt_string],
)
def string_json_label_extractor_v1(*s, **kwargs):
    default_label = kwargs.get("default_label", _RANDOM)
    extract_key = kwargs.get("extract_key", "label")
    size = kwargs.get("size", 1)
    assert size > 0, "The number of elements must be a positive int"
    return string_json_label_extractor_v1_impl(s, default_label, extract_key, size)


def string_json_label_extractor_v1_impl(
    s: List, default_label: str, extract_key: str, size: int
):
    assert s[0].ndim == 2, "json_strings must be a 2D tensor"
    assert s[1].ndim == 1, "candidate_labels must be a flat tensor"
    assert s[2].ndim == 1, "probabilities must be a flat tensor"
    assert (
        s[1].size == s[2].size
    ), "candidate_labels and probabilities must have the same size"
    json_strings = s[0]
    candidate_labels = s[1]
    probabilities = s[2]
    final_labels = []
    for json_string in json_strings:
        label = extract_json_key_from_string(json_string[0], extract_key)
        local_labels = []
        if not isinstance(label, list):
            label = [label]
        for sublabel in label:
            if str(sublabel) in candidate_labels:
                if str(sublabel) not in local_labels:
                    local_labels.append(str(sublabel))
            elif (
                adjusted_sublabel := str(sublabel).replace("'", "").replace('"', "")
                in candidate_labels
            ):
                if adjusted_sublabel not in local_labels:
                    local_labels.append(adjusted_sublabel)
            else:
                if default_label == _RANDOM:
                    randomized = str(
                        np.random.choice(candidate_labels, p=probabilities)
                    )
                    if randomized not in local_labels:
                        local_labels.append(randomized)
                else:
                    local_labels.append(default_label)
        local_labels = (local_labels + [""] * size)[:size]
        final_labels.append(local_labels)
    return np.array(final_labels, dtype=np.str_).reshape(-1, size)
