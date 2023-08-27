from typing import Type, Dict
import numpy as np

_DTYPE_MAP: Dict[str, Type] = {
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(string)": np.str_,
    "tensor(bool)": np.bool_,
    "tensor(complex64)": np.complex64,
    "tensor(complex128)": np.complex128,
}


def onnx_to_numpy_type(onnx_type: str) -> Type:
    if onnx_type not in _DTYPE_MAP.keys():
        raise ValueError(f"Uns ONNX type {onnx_type}")
    return _DTYPE_MAP[onnx_type]
