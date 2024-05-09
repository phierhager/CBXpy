from __future__ import annotations
from typing import List, Callable, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..dynamics import ParticleDynamic


class DefaultPostProcess:
    """
    Default post processing.

    This function performs a clipping operations on the particles after the inner step.

    Parameters:
        None

    Return:
        None
    """
    def __init__(self, max_thresh: float = 1e8) -> None:
        self.max_thresh = max_thresh
    
    def __call__(self, dyn: ParticleDynamic) -> None:
        np.nan_to_num(dyn.x, copy=False, nan=self.max_thresh)
        dyn.x = np.clip(dyn.x, -self.max_thresh, self.max_thresh)



class CompositePostProcess:
    """
    Composite Post Processing.

    This function sequentially passes the dynamic particle object through a given list of operations.
    """
    def __init__(self, operations: List[Callable]) -> None:
        self.operations = operations
    
    def __call__(self, dyn: ParticleDynamic) -> None:
        for operation in self.operations:
            operation(dyn)



RESHUFFLE_INTERVAL: int = 50
class ReshuffleAgentBatches:
    def __init__(self, reshuffle_interval: int = RESHUFFLE_INTERVAL) -> None:
        self.reshuffle_interval = reshuffle_interval

    def _reshuffle(self, dyn: ParticleDynamic) -> None:
        if (dyn.it + 1) % self.reshuffle_interval == 0:
            x = dyn.x
            M, N, d = x.shape

            x = x.reshape(-1, d)
            x = np.random.permutation(x)
            x = x.reshape(M, N, d)

            dyn.x = x

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._reshuffle(*args, **kwds)
