from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, ClassVar, Union

import torch

Number = Union[float, torch.Tensor]


@dataclasses.dataclass
class BaseScheduler(abc.ABC):
    """Shared implementation for scalar-valued diffusion schedulers."""

    __registry__: ClassVar[dict[str, type[BaseScheduler]]] = {}
    _argument_name: ClassVar[str] = "t"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__registry__" in cls.__dict__:
            return

        cls.__registry__[cls.__name__] = cls
        cls.__registry__[cls.__name__.lower()] = cls

    # Make instances callable (sched(t) -> value(t))
    def __call__(self, t: Number) -> Number:
        return self.value(t)

    # ---- common API ----
    def value(self, t: Number) -> Number:
        return self._evaluate(t, self._value)

    def derivative(self, t: Number) -> Number:
        return self._evaluate(t, self._derivative)

    def value_and_derivative(self, t: Number) -> tuple[Number, Number]:
        t_tensor = self._validated_tensor(t)
        return (
            self._convert_output(self._value(t_tensor), t),
            self._convert_output(self._derivative(t_tensor), t),
        )

    def _evaluate(
        self,
        t: Number,
        operation: Callable[[torch.Tensor], torch.Tensor],
    ) -> Number:
        t_tensor = self._validated_tensor(t)
        return self._convert_output(operation(t_tensor), t)

    def _validated_tensor(self, t: Number) -> torch.Tensor:
        t_tensor = self._as_tensor(t)
        if torch.all((0.0 <= t_tensor) & (t_tensor <= 1.0)):
            return t_tensor

        raise ValueError(f"{self._argument_name}={t} not in [0,1]")

    @staticmethod
    def _as_tensor(t: Number) -> torch.Tensor:
        device = t.device if isinstance(t, torch.Tensor) else None
        return torch.as_tensor(t, dtype=torch.float32, device=device)

    @staticmethod
    def _convert_output(output: torch.Tensor, *originals: Number) -> Number:
        if all(isinstance(original, float) for original in originals):
            return output.item()
        return output

    # ---- hooks implemented by subclasses ----
    @abc.abstractmethod
    def _value(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _derivative(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
