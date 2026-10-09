"""Random_Forest digit classification backend."""
import numpy as np
from task3_mnist.interfaces import DigitClassificationInterface


class RandomForestDigitModel(DigitClassificationInterface):
    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, image):
        if image.shape != (784,):
            raise ValueError("RF expects a flat vector of length 784")
        return int(self.estimator.predict(image.reshape(1, -1))[0])
