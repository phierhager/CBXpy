from . import CBO
from ..utils.particle_init import init_particles
from ..noise import noise, isotropic_noise, anisotropic_noise


from typing import Callable
from numpy.typing import ArrayLike
import numpy as np
import torch


class QCBO(CBO):
    r"""QCBO class"""

    def __init__(self, f, **kwargs) -> None:
        super().__init__(f, **kwargs)
        print("QCBO")
        self.set_noise(
            anisotropic_noise(
                norm=torch.linalg.norm,
                sampler=lambda _, scale, size: torch.distributions.Exponential(
                    scale
                ).sample(size),
            )
        )
