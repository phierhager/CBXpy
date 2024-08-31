from . import CBO
from ..utils.particle_init import init_particles
from ..noise import noise, isotropic_noise, anisotropic_noise


from typing import Callable
from numpy.typing import ArrayLike
import numpy as np


# maybe needed later, but expo can be easily implemented with the sampler in noise
class exponential_noise(noise):
    def __init__(self, norm: Callable = None, sampler: Callable = None):
        """_summary_

        Args:
            norm (Callable, optional): If None, np.linalg.norm. Defaults to None.
            sampler (Callable, optional): If None, normal sampler. Defaults to None.
        """
        super().__init__(norm=norm, sampler=sampler)

    def __call__(self, dyn) -> ArrayLike:
        factor = np.sqrt((1 / dyn.lamda) * (1 - np.exp(-dyn.dt) ** 2))[
            (...,) + (None,) * (dyn.x.ndim - 2)
        ]
        return factor * self.sample(dyn.drift, dyn.Cov_sqrt)

    def sample(self, drift: ArrayLike, Cov_sqrt: ArrayLike) -> ArrayLike:
        z = self.sampler(0, 1, size=drift.shape)
        return self.apply_cov_sqrt(Cov_sqrt, z)


class QCBO(CBO):
    r"""QCBO class"""

    def __init__(self, f, **kwargs) -> None:
        super().__init__(f, **kwargs)
        self.set_noise(
            anisotropic_noise(
                norm=np.linalg.norm,
                sampler=lambda _, scale, size: np.random.exponential(
                    scale=scale, size=size
                ),
            )
        )
