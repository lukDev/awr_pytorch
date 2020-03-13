from enum import Enum, auto


class DebugType(Enum):
    NONE = auto()  # no debug output
    EVAL = auto()  # debug output only in evaluation, i. e. when computing the average return
    FULL = auto()  # full debug output
