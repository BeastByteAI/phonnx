from typing import List, Tuple
import re
from phonnx.constants import INPUT_IDENTIFIER, DYNAMIC_ATTRIBUTE_IDENTIFIER
from phonnx.constants import ColumnTypes


def split_node_name(node: str, pattern="[_-]") -> List[str]:
    return re.split(pattern, node)


def determine_column_type(node: str) -> ColumnTypes:
    # unknown types are treated as numeric (float32)
    type_ = split_node_name(node)[2]
    try:
        return ColumnTypes(int(type_))
    except ValueError:
        return ColumnTypes.NUMERIC_REGULAR


def is_valid_node_name(node: str) -> bool:
    pattern = r"^.+[-_]pl[-_]\d+/.*$"
    return bool(re.match(pattern, node))


def get_node_type(node: str) -> str:
    return split_node_name(node)[1]


def get_layer_id(node: str) -> int:
    return int(split_node_name(node, pattern="[/_-]")[2])


def retrieve_inputs(ort_inputs: List[str]) -> Tuple[List[str], List[str]]:
    inputs, dynattrs = [], []
    for input in ort_inputs:
        if get_node_type(input) == INPUT_IDENTIFIER:
            inputs.append(input)
        elif get_node_type(input) == DYNAMIC_ATTRIBUTE_IDENTIFIER:
            dynattrs.append(input)
    return inputs, dynattrs


def get_output_names(output_names: List[str], outputs: str = "final") -> List[str]:
    if outputs not in ["all", "final"]:
        raise ValueError(
            f"Expected `outputs` to be one of [all, final], got `{outputs}`."
        )
    if outputs == "final" and len(output_names) > 1:
        idx_ = []
        for name in output_names:
            if not is_valid_node_name(name):
                raise RuntimeError("One of the output nodes has an invalid name.")
            idx = get_layer_id(name)
            idx_.append(idx)
        max_idx = max(idx_)
        output_names = [n for n in output_names if get_layer_id(n) == max_idx]
    return output_names
