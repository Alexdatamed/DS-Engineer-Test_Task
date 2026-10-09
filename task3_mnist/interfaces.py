"""Common contract for digit classification strategies."""
from abc import ABC, abstractmethod
import numpy as np


class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """Digit for the backend-specific image representation."""

    def train(self, *args, **kwargs):
        raise NotImplementedError("Training is outside this inference interface")
