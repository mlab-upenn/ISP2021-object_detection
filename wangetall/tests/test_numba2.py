from numba import jit
import numpy as np
from numba.typed import Dict
from numba import types

d = Dict.empty(
    key_type=types.int64,
    value_type=types.float64[:],
)

    compat_boundaries = Dict.empty(
                        key_type=types.int64,
                        value_type=types.float64[:],
    )


@jit(nopython=True)
def foo():
    # The typed-dict can be used from the interpreter.
    d[0] = np.array([1, 0, np.nan])


foo()