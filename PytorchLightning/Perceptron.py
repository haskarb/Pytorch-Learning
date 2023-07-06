# Implement a perceptron model

import numpy as np
from typing import Union, List


# create classification dataset
def create_dataset(
    n_samples: int, n_features: int, n_classes: int
) -> Union[np.ndarray, np.ndarray]:
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


class Perceptron:
    def __init__(self, n_features) -> None:
        self.n_features = n_features
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0

    def forward(self, x: np.ndarray) -> int:
        # x1*w1 + x2*w2 + ... + xn*wn + b
        weighted_sum = self.bias
        for i, _ in enumerate(self.weights):
            weighted_sum += np.dot(x, self.weights)
        return np.where(weighted_sum > 0, 1, 0)

    def update(self, x, y_true) -> float:
        """
        Update weights and bias for one sample.
        """
        y_pred = self.forward(x)
        error = y_true - y_pred

        # update weights and bias
        for i, _ in enumerate(self.weights):
            self.weights[i] += error * x[i]

        return error

def evaluate(model, X: np.ndarray, y: np.ndarray) -> float:
    """
    Evaluate the model on the test set.
    """
    n_samples = len(X)
    n_correct = 0
    for x, y in zip(X, y):
        y_pred = model.forward(x)
        if y == y_pred:
            n_correct += 1
    acc = n_correct / n_samples
    return acc


def train(model, X: np.ndarray, y: np.array, n_epochs: int):
    for epoch in range(n_epochs):
        errors = 0
        # feed data sample by sample
        for x, y in zip(X, Y):
            error = model.update(x, y)
            errors += abs(error)
        print(f"epoch: {epoch}, errors: {errors}")


# test
X, Y = create_dataset(10, 2, 2)
# X= [-1.2, 2.4]
# Y = 0
ppn = Perceptron(2)
# print(ppn.update(X, Y))
# train(ppn, X, Y, 10)
print(evaluate(ppn, X, Y))
