"""Global Cartesian dynamics: state = [px, py, pz, vx, vy, vz]."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConstantVelocity:
    """dp/dt = v, dv/dt = 0. Units: metres and seconds."""

    name = "cv"

    def rhs(self, time_s, state):
        # Autonomous ODE: the derivative does not explicitly depend on time.
        return np.concatenate((state[3:6], np.zeros(3)))


@dataclass(frozen=True)
class ConstantAcceleration:
    """dp/dt = v, dv/dt = a; acceleration is a fixed external parameter."""

    acceleration_mps2: tuple
    name = "ca"

    def __post_init__(self):
        acceleration = np.asarray(self.acceleration_mps2, dtype=np.float64)
        if acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
            raise ValueError("acceleration_mps2 must contain three finite numbers")
        object.__setattr__(self, "acceleration_mps2", tuple(acceleration.tolist()))

    def rhs(self, time_s, state):
        return np.concatenate((state[3:6], self.acceleration_mps2))
