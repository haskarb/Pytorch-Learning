import numpy as np

# Linear Regression
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

# print(X.shape)
w = 0.0


# forward pass
def forward(x):
    return w * x


# loss calculation (MSE)
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


# backwards
## gradient
# MSE = 1/N * (w*x-y)**2
# dJ/dw = 1/N  2(w*x-y) x


def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()


print(f"Prediction before training:f(5)= {forward(5):.3f}")

# Training
n_iter = 20
learning_rate = 0.01
verbose = 2

for epoch in range(n_iter):
    # forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradient
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % verbose == 0:
        print(f"epoch {epoch + 1}: w = {w:.3f}, loss = {l:.2f}")


print(f"Prediction after training:f(5)= {forward(5):.3f}")
