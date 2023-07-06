import torch
import numpy as np
from typing import Union


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
        self.weights = torch.zeros(self.n_features)
        self.bias = torch.tensor(0.0)

    def forward(self, x) -> torch.Tensor:
        # x1*w1 + x2*w2 + ... + xn*wn + b
        weighted_sum = torch.dot(x, self.weights) + self.bias

        return torch.where(weighted_sum > 0, 1, 0)

    def update(self, x, y_true) -> float:
        """
        Update weights and bias for one sample.
        """
        y_pred = self.forward(x)
        error = y_true - y_pred

        # update weights and bias
        self.weights += error * x
        self.bias += error
        # print(error.dtype)
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
            # break
            errors += abs(error)
        print(f"epoch: {epoch}, errors: {errors}")


# test
import time
X, Y = create_dataset(10000, 2, 2)
# Convert to torch tensors 
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)
# convert to float32
X = X.to(dtype=torch.float32)
Y = Y.to(dtype=torch.float32)
ppn = Perceptron(2)
# compute the time
start = time.time()
# print(evaluate(ppn, X, Y))
train(ppn, X, Y, 10)
end = time.time()
print(end - start)