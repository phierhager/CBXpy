from . import CBO
from ..utils.particle_init import init_particles
from ..noise import noise, isotropic_noise, quantum_noise


from typing import Callable
from numpy.typing import ArrayLike
import numpy as np
import torch


class QCBO(CBO):
    r"""QCBO class"""

    def __init__(self, f, **kwargs) -> None:
        c = kwargs.pop("c")
        super().__init__(f, **kwargs)
        print("QCBO")
        self.set_noise(quantum_noise(framework="numpy", c=c))
