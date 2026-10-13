"""Pure timestamp query using numerical ODE integration, without track updates."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import ConstantAcceleration, ConstantVelocity


def _timestamp_us(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("{} must be an integer timestamp in microseconds".format(name))
    return int(value)


def _positive_float(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be finite and positive".format(name))
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("{} must be finite and positive".format(name))
    return value


@dataclass(frozen=True)
class Prediction:
    model: str
    state: tuple
    state_timestamp_us: int
    query_timestamp_us: int
    delta_t_s: float
    solver_evaluations: int
    solver_message: str

    def to_dict(self):
        velocity = np.asarray(self.state[3:6])
        return {
            "model": self.model,
            "state_timestamp_us": self.state_timestamp_us,
            "query_timestamp_us": self.query_timestamp_us,
            "delta_t_s": self.delta_t_s,
            "state": list(self.state),
            "position_m": list(self.state[:3]),
            "velocity_mps": list(self.state[3:6]),
            "speed_xy_mps": float(np.linalg.norm(velocity[:2])),
            "speed_3d_mps": float(np.linalg.norm(velocity)),
            "solver": {
                "method": "RK45", "success": True,
                "nfev": self.solver_evaluations, "message": self.solver_message,
            },
        }


class ODEPredictor:
    """Query a state at t >= t0; input state and the predictor are not advanced.

    Both supported dynamics are autonomous. The solver's time origin is t0,
    so rhs receives elapsed seconds, never a floating-point Unix timestamp.
    Only a successful, finite integration returns a Prediction.
    """

    def __init__(self, dynamics, rtol=1e-9, atol=1e-11, max_step_s=0.1):
        if not isinstance(dynamics, (ConstantVelocity, ConstantAcceleration)):
            raise TypeError("dynamics must be ConstantVelocity or ConstantAcceleration")
        self.dynamics = dynamics
        self.rtol = _positive_float(rtol, "rtol")
        self.atol = _positive_float(atol, "atol")
        self.max_step_s = _positive_float(max_step_s, "max_step_s")

    def predict(self, state, state_timestamp_us, query_timestamp_us):
        initial = np.array(state, dtype=np.float64, copy=True)
        if initial.shape != (6,) or not np.all(np.isfinite(initial)):
            raise ValueError("state must be six finite numbers: px py pz vx vy vz")
        t0 = _timestamp_us(state_timestamp_us, "state_timestamp_us")
        t = _timestamp_us(query_timestamp_us, "query_timestamp_us")
        if t < t0:
            raise ValueError("query_timestamp_us must be >= state_timestamp_us")
        # Subtract integers FIRST: preserve small intervals at large epochs.
        dt = (t - t0) / 1_000_000.0
        if not np.isfinite(dt):
            raise ValueError("timestamp interval must be finite")
        if dt == 0:
            final, nfev, message = initial, 0, "Same timestamp; no integration required."
        else:
            solution = solve_ivp(
                self.dynamics.rhs, (0.0, dt), initial, method="RK45",
                t_eval=[dt], rtol=self.rtol, atol=self.atol,
                max_step=self.max_step_s,
            )
            if not solution.success:
                raise RuntimeError("ODE integration failed: {}".format(solution.message))
            if solution.y.shape != (6, 1) or not np.all(np.isfinite(solution.y)):
                raise RuntimeError("ODE integration returned an invalid state")
            final, nfev, message = solution.y[:, -1], solution.nfev, solution.message
        return Prediction(
            self.dynamics.name, tuple(final.tolist()), t0, t, dt, int(nfev), str(message)
        )


def predictor_from_config(config):
    """Strict YAML-compatible configuration; reject misspelled/unused settings."""
    if not isinstance(config, dict) or set(config) - {"dynamics", "solver"}:
        raise ValueError("config must contain only dynamics and solver mappings")
    dynamics = config.get("dynamics")
    solver = config.get("solver", {})
    if not isinstance(dynamics, dict) or not isinstance(solver, dict):
        raise ValueError("dynamics and solver must be mappings")
    name = dynamics.get("model")
    if name == "cv":
        if set(dynamics) != {"model"}:
            raise ValueError("cv dynamics accepts only model; acceleration is always zero")
        model = ConstantVelocity()
    elif name == "ca":
        if set(dynamics) != {"model", "acceleration_mps2"}:
            raise ValueError("ca dynamics requires model and acceleration_mps2")
        model = ConstantAcceleration(dynamics["acceleration_mps2"])
    else:
        raise ValueError("dynamics.model must be cv or ca")
    if set(solver) - {"rtol", "atol", "max_step_s"}:
        raise ValueError("solver accepts only rtol, atol and max_step_s (method is RK45)")
    return ODEPredictor(model, **solver)
