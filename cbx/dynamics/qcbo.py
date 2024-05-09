from . import CBO
from ..utils.particle_init import init_particles
from ..noise import noise, isotropic_noise


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
        self.set_correction("no_correction")
        self.set_noise(
            isotropic_noise(
                norm=np.linalg.norm,
                sampler=lambda _, scale, size: np.random.exponential(
                    scale=scale, size=size
                ),
            )
            # for testing purposes we want to have different noise
            if kwargs.get("noise") is None
            else kwargs.get("noise")
        )

    # overwrites init_x in CBXDynamic
    # for now init with gaussian
    def init_x(self, x, M, N, d, x_min, x_max):
        # copy paste from CBO
        # TODO: Change init to exponential init
        """
        Initialize the particle system with the given parameters.

        Parameters:
            x: the initial particle system. If x is None, the dimension d must be specified and the particle system is initialized randomly.
               If x is specified, it is broadcasted to the correct shape (M,N,d).
            M: the number of particles in the first dimension
            N: the number of particles in the second dimension
            d: the dimension of the particle system
            x_min: the minimum value for x
            x_max: the maximum value for x

        Returns:
            None
        """
        if x is None:
            if d is None:
                raise RuntimeError(
                    "If the inital partical system is not given, the dimension d must be specified!"
                )
            x = init_particles(
                shape=(M, N, d),
                x_min=x_min,
                x_max=x_max,
                # mode = exponential TODO
            )
        else:  # if x is given correct shape
            if len(x.shape) == 1:
                x = x[None, None, :]
            elif len(x.shape) == 2:
                x = x[None, :]

        self.M = x.shape[0]
        self.N = x.shape[1]
        self.d = x.shape[2:]
        self.ddims = tuple(i for i in range(2, x.ndim))
        self.x = self.copy(x)

    def inner_step(
        self,
    ) -> None:
        r"""Performs one step of the QCBO algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # update, consensus point, drift and energy
        self.consensus, energy = self.compute_consensus()
        self.energy[self.consensus_idx] = energy
        self.drift = self.x[self.particle_idx] - self.consensus

        # compute noise
        self.s = self.sigma * self.noise()

        #  update particle positions
        self.x[self.particle_idx] = (
            self.x[self.particle_idx]
            - self.correction(self.lamda * self.dt * self.drift)
            + self.s
        )
