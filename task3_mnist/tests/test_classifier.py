import unittest
import importlib.util
import numpy as np
from task3_mnist.classifier import DigitClassifier
from task3_mnist.models.cnn import build_cnn


class DigitTests(unittest.TestCase):
    def test_random_and_training(self):
        first, second = DigitClassifier("rand",seed=4), DigitClassifier("rand",seed=4)
        image = np.zeros((28,28,1),dtype=np.uint8)
        self.assertEqual([first.predict(image) for _ in range(20)], [second.predict(image) for _ in range(20)])
        with self.assertRaises(NotImplementedError):
            first.train()

    def test_rf_adapter(self):
        class FakeRF:
            def predict(self, x):
                np.testing.assert_equal(x.shape,(1,784))
                np.testing.assert_allclose(x,1)
                return np.array([8])
        self.assertEqual(DigitClassifier("rf",backend=FakeRF()).predict(np.full((28,28,1),255)),8)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Install requirements.txt")
    def test_fitted_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        x = np.vstack([np.zeros(784), np.ones(784)])
        backend = RandomForestClassifier(n_estimators=3, random_state=42).fit(x, [0, 1])
        classifier = DigitClassifier("rf", backend=backend)
        self.assertEqual(classifier.predict(np.zeros((28,28,1))), 0)
        self.assertEqual(classifier.predict(np.full((28,28,1),255)), 1)

    def test_crop(self):
        classifier = DigitClassifier("rand")
        class CropSpy:
            def predict(self, crop):
                np.testing.assert_allclose(crop,1)
                self_shape = crop.shape
                assert self_shape == (10,10)
                return 3
        classifier.model = CropSpy()
        image = np.zeros((28,28,1)); image[9:19,9:19] = 255
        self.assertEqual(classifier.predict(image),3)

    def test_invalid_inputs(self):
        classifier = DigitClassifier("rand")
        for image in (np.zeros((28,28)),np.full((28,28,1),np.nan),np.full((28,28,1),256)):
            with self.assertRaises(ValueError):
                classifier.predict(image)
        for algorithm in ("unknown","cnn","rf"):
            with self.assertRaises(ValueError):
                DigitClassifier(algorithm)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Install requirements.txt")
    def test_cnn(self):
        import torch
        model = build_cnn()
        classifier = DigitClassifier("cnn",backend=model)
        digit = classifier.predict(np.zeros((28,28,1)))
        self.assertIs(type(digit),int)
        self.assertFalse(model.training)
        self.assertTrue(0 <= digit <= 9)

if __name__ == "__main__":
    unittest.main()
