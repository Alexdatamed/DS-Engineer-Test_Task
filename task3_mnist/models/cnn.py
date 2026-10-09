"""CNN digit classification backend."""
import numpy as np
from task3_mnist.interfaces import DigitClassificationInterface


class CNNDigitModel(DigitClassificationInterface):
    def __init__(self, model, device="cpu"):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def predict(self, image):
        if image.shape != (28, 28, 1):
            raise ValueError("CNN expects a 28x28x1 image")
        tensor = self.torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            scores = self.model(tensor)
        if tuple(scores.shape) != (1, 10) or not self.torch.isfinite(scores).all():
            raise ValueError("CNN must return finite logits of shape (1, 10)")
        return int(scores.argmax(dim=1).item())


def build_cnn():
    """Small architecture; caller must load trained weights before inference."""
    from torch import nn
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1),
                         nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                         nn.Linear(32*7*7, 10))
