import numpy as np
from typing import Callable, Dict
from phonnx.constants import ColumnTypes


def ensure_min_2d(X: np.ndarray) -> np.ndarray:
    """
    Ensures that the input array has at least 2 dimensions.
    """
    if X.ndim < 2:
        return X.reshape(1, -1)
    return X


MAP: Dict[ColumnTypes, Callable] = {
    ColumnTypes.NUMERIC_REGULAR: ensure_min_2d,
    ColumnTypes.CAT_LOW_CARD: ensure_min_2d,
    ColumnTypes.CAT_HIGH_CARD: ensure_min_2d,
    ColumnTypes.TEXT_UTF8: ensure_min_2d,
    ColumnTypes.DATE_YMD_ISO8601: ensure_min_2d,
    ColumnTypes.DATETIME_YMDHMS_ISO8601: ensure_min_2d,
}
