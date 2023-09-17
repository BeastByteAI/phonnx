from enum import Enum

INPUT_IDENTIFIER = "input"
DYNAMIC_ATTRIBUTE_IDENTIFIER = "dynattr"
STEP_IDENTIFIER = "pl"

AI_BEASTBYTE_PYOPS = "ai.beastbyte.pyops"


class ColumnTypes(Enum):
    NUMERIC_REGULAR = 0
    CAT_LOW_CARD = 1
    CAT_HIGH_CARD = 2
    TEXT_UTF8 = 3
    DATE_YMD_ISO8601 = 100  # %Y-%m-%d i.e. '2023-02-21'
    DATETIME_YMDHMS_ISO8601 = 101  # %Y-%m-%dT%H:%M:%SZ i.e. '2023-02-21T17:24:22Z' OR %Y-%m-%d %H:%M:%S i.e. '2023-02-21 17:24:22' # noqa: E501
