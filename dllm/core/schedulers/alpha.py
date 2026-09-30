from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import torch

from .base import BaseScheduler, Number


# ---------------- Registry-enabled Base ---------------- #
@dataclasses.dataclass
class BaseAlphaScheduler(BaseScheduler):
    """
    Base class for alpha schedulers in diffusion language models.

    Alpha schedulers define the masking rate α(t) as a function of diffusion time t ∈ [0,1].
    Subclasses are automatically registered and can be instantiated by name.

    To implement a custom scheduler, inherit from this class and implement:
    - _value(t): Compute α(t) for a tensor of timesteps
    - _derivative(t): Compute dα/dt for a tensor of timesteps

    Example:
        @dataclasses.dataclass
        class CustomScheduler(BaseAlphaScheduler):
            def _value(self, t):
                return 1 - t**2
            def _derivative(self, t):
                return -2 * t
    """

    __registry__: ClassVar[dict[str, type[BaseAlphaScheduler]]] = {}
    _argument_name: ClassVar[str] = "i"

    # ---- common API ----
    def alpha(self, i: Number) -> Number:
        return self.value(i)

    def alpha_derivative(self, i: Number) -> Number:
        return self.derivative(i)

    def reverse_mask_prob(self, s: Number, t: Number) -> Number:
        s_tensor = self._as_tensor(s)
        t_tensor = self._as_tensor(t)
        valid = (
            (0.0 <= s_tensor)
            & (s_tensor < 1.0)
            & (0.0 < t_tensor)
            & (t_tensor <= 1.0)
            & (s_tensor < t_tensor)
        )
        if not torch.all(valid):
            raise ValueError(f"Require 0 <= s < t <= 1, but got (t={t}, s={s})")

        output = (1 - self._value(s_tensor)) / (1 - self._value(t_tensor))
        return self._convert_output(output, s, t)

    def weight(self, i: Number) -> Number:
        # w(t) = - α'(t) / (1 - α(t))
        alpha, alpha_derivative = self.value_and_derivative(i)
        return -alpha_derivative / (1 - alpha + 1e-6)


# ---------------- Implementations ---------------- #


@dataclasses.dataclass
class LinearAlphaScheduler(BaseAlphaScheduler):
    def _value(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def _derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


@dataclasses.dataclass
class CosineAlphaScheduler(BaseAlphaScheduler):
    def _value(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - torch.cos((math.pi / 2) * (1 - t))

    def _derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -(math.pi / 2) * torch.sin((math.pi / 2) * (1 - t))


# ---------------- Example usage ---------------- #

if __name__ == "__main__":
    lin_sched = LinearAlphaScheduler()
    print("Linear α(0.5):", lin_sched.alpha(0.5))
    print("Linear w(0.5):", lin_sched.weight(0.5))
    print("Linear α([.25,.5,.75]):", lin_sched.alpha(torch.tensor([0.25, 0.5, 0.75])))
    print("Linear w([.25,.5,.75]):", lin_sched.weight(torch.tensor([0.25, 0.5, 0.75])))
    print("==========================================")
    cos_sched = CosineAlphaScheduler()
    print("Cosine α(0.5):", cos_sched.alpha(0.5))
    print("Cosine w(0.5):", cos_sched.weight(0.5))
    print("Cosine α([.25,.5,.75]):", cos_sched.alpha(torch.tensor([0.25, 0.5, 0.75])))
    print("Cosine w([.25,.5,.75]):", cos_sched.weight(torch.tensor([0.25, 0.5, 0.75])))
