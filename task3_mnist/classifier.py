"""Unified digit classification facade and input adapters."""
import numpy as np
from task3_mnist.models.cnn import CNNDigitModel
from task3_mnist.models.random_forest import RandomForestDigitModel
from task3_mnist.models.random_model import RandomDigitModel


class DigitClassifier:
    def __init__(self, algorithm, *, backend=None, seed=None, device="cpu"):
        if algorithm == "rand":
            self.model = RandomDigitModel(seed)
        elif algorithm == "rf":
            if backend is None:
                raise ValueError("rf requires a fitted estimator")
            self.model = RandomForestDigitModel(backend)
        elif algorithm == "cnn":
            if backend is None:
                raise ValueError("cnn requires a model with trained weights")
            self.model = CNNDigitModel(backend, device)
        else:
            raise ValueError("algorithm must be cnn, rf or rand")
        self.algorithm = algorithm

    def train(self, *args, **kwargs):
        raise NotImplementedError("Training is outside the task scope")

    def predict(self, image):
        """Accept pixels in [0,255], normalize to [0,1]"""
        image = np.asarray(image)
        if image.shape != (28, 28, 1):
            raise ValueError("Expected shape (28, 28, 1)")
        if image.dtype.kind not in "uif" or not np.isfinite(image).all():
            raise ValueError("Expected finite real pixels")
        if image.min() < 0 or image.max() > 255:
            raise ValueError("Pixels must be in [0,255]")
        image = image.astype(np.float32)/255.0
        if self.algorithm == "rf":
            adapted = image.reshape(784)
        elif self.algorithm == "rand":
            adapted = image[9:19, 9:19, 0]
        else:
            adapted = image
        digit = self.model.predict(adapted)
        if type(digit) is not int or not 0 <= digit <= 9:
            raise ValueError("Backend must return an integer digit in [0,9]")
        return digit
