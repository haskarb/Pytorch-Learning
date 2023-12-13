# 1. Design mode (input, output size, forward pass)
# 2. Construct loss and optimiser
# 3. Training loop
#       -forward pass: Compute prediction
#       -backward pass: gradients
#       - update weights


import torch
import torch.nn as nn


X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape

# Design Model
model = nn.Linear(in_features=n_features, out_features=n_features)

print(f"Prediction before training:f(5)= {model(X_test).item():.3f}")


# Contruct loss and optim
lr = 0.01
n_iter = 100

loss = nn.MSELoss()
optim = torch.optim.SGD(params=model.parameters(), lr=lr)

#Training loop
for epoch in range(n_iter):
    y_pred = model(X)

    # calculate loss
    l = loss(Y, y_pred)

    # backwar pass
    l.backward()

    #update weights
    optim.step()

    #zero grads
    optim.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack parameters
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)



print(f"Prediction before training:f(5)= {model(X_test).item():.3f}")
