"""random_model digit classification backend."""
import numpy as np
from task3_mnist.interfaces import DigitClassificationInterface


class RandomDigitModel(DigitClassificationInterface):
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def predict(self, image):
        if image.shape != (10, 10):
            raise ValueError("Random model expects a 10x10 center crop")
        return int(self.rng.integers(0, 10))
