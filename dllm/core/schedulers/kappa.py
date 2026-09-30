from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import torch

from .base import BaseScheduler, Number


# ---------------- Registry-enabled Base ---------------- #
@dataclasses.dataclass
class BaseKappaScheduler(BaseScheduler):
    """
    Base class for kappa schedulers in diffusion language models.

    Kappa schedulers define the noise schedule κ(t) as a function of diffusion time t ∈ [0,1].
    Unlike alpha schedulers (which control masking rates), kappa controls the interpolation
    between source and target in edit flow models. Subclasses are automatically registered.

    To implement a custom scheduler, inherit from this class and implement:
    - _value(t): Compute κ(t) for a tensor of timesteps
    - _derivative(t): Compute dκ/dt for a tensor of timesteps

    Example:
        @dataclasses.dataclass
        class CustomKappaScheduler(BaseKappaScheduler):
            def _value(self, t):
                return t**3
            def _derivative(self, t):
                return 3 * t**2
    """

    __registry__: ClassVar[dict[str, type[BaseKappaScheduler]]] = {}

    # ---- common API ----
    def kappa(self, t: Number) -> Number:
        return self.value(t)

    def kappa_derivative(self, t: Number) -> Number:
        return self.derivative(t)

    def weight(self, t: Number) -> Number:
        # w(t) = κ'(t) / (1 - κ(t))
        kappa, kappa_derivative = self.value_and_derivative(t)
        return kappa_derivative / (1 - kappa + 1e-6)


# ---------------- Implementations ---------------- #


@dataclasses.dataclass
class CubicKappaScheduler(BaseKappaScheduler):
    a: float = 1.0
    b: float = 1.0

    def _value(self, t: torch.Tensor) -> torch.Tensor:
        # κ(t) = (a+1) t^3 - (a+b+1) t^2 + (b+1) t
        return (self.a + 1) * (t**3) - (self.a + self.b + 1) * (t**2) + (self.b + 1) * t

    def _derivative(self, t: torch.Tensor) -> torch.Tensor:
        # κ'(t) = 3(a+1) t^2 - 2(a+b+1) t + (b+1)
        return 3 * (self.a + 1) * (t**2) - 2 * (self.a + self.b + 1) * t + (self.b + 1)


@dataclasses.dataclass
class LinearKappaScheduler(CubicKappaScheduler):
    # Special case: κ(t) = t corresponds to a=-1, b=0
    a: float = -1.0
    b: float = 0.0


@dataclasses.dataclass
class CosineKappaScheduler(BaseKappaScheduler):
    def _value(self, t: torch.Tensor) -> torch.Tensor:
        # κ(t) = 1 - cos((π/2) * t)
        return 1.0 - torch.cos(0.5 * math.pi * t)

    def _derivative(self, t: torch.Tensor) -> torch.Tensor:
        # κ'(t) = (π/2) * sin((π/2) * t)
        return 0.5 * math.pi * torch.sin(0.5 * math.pi * t)


# ---------------- Example usage ---------------- #

if __name__ == "__main__":
    lin_sched = LinearKappaScheduler()
    print("Linear κ(0.5):", lin_sched.kappa(0.5))
    print("Linear w(0.5):", lin_sched.weight(0.5))
    print("Linear κ([.25,.5,.75]):", lin_sched.kappa(torch.tensor([0.25, 0.5, 0.75])))
    print("Linear w([.25,.5,.75]):", lin_sched.weight(torch.tensor([0.25, 0.5, 0.75])))
    print("==========================================")
    cos_sched = CosineKappaScheduler()
    print("Cosine κ(0.5):", cos_sched.kappa(0.5))
    print("Cosine w(0.5):", cos_sched.weight(0.5))
    print("Cosine κ([.25,.5,.75]):", cos_sched.kappa(torch.tensor([0.25, 0.5, 0.75])))
    print("Cosine w([.25,.5,.75]):", cos_sched.weight(torch.tensor([0.25, 0.5, 0.75])))
