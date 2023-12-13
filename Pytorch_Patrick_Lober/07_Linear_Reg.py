import torch
import numpy as np
from sklearn import datasets

# Prepare regression dataset
X_numpy, y_numpy = datasets.make_regression(
    n_samples=1000, n_features=100, noise=20, random_state=42
)

# conver to tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# create mode
model = torch.nn.Linear(in_features=n_features, out_features=1)

# loss and optimiser
n_iter = 1000
lr = 0.1
loss = torch.nn.MSELoss()
optimiser = torch.optim.SGD(params=model.parameters(), lr=lr)


# training loop
for epoch in range(n_iter):
    # forward pass
    y_pred = model(X)

    # calculate loss
    l = loss(y, y_pred)

    # backprop
    l.backward()

    # update weight
    optimiser.step()

    # zero grad
    optimiser.zero_grad()

    if epoch % 10 == 0:
        print(f"epoch: {epoch} loss: {l.item():.3f}")
