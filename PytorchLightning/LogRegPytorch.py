import torch

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



class LogReg(torch.nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits)

        return probs

# # set seed for reproducibility
# torch.manual_seed(42)

# model = LogReg(n_features=2)

# x = torch.tensor([1.0, 2.0])
# with torch.inference_mode():    
#     print(model.forward(x))


# Define dataloader
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

def compute_accuracy(
    logits: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    preds = (logits > 0.5).long()
    correct_preds = (preds == y).sum().float()
    acc = correct_preds / y.shape[0]


X, y = create_dataset(n_samples=100, n_features=2, n_classes=2)
train_ds = MyDataset(X, y)

train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)


# train model
import torch.nn.functional as F

model2 = LogReg(n_features=2)
optimizer = torch.optim.SGD(model2.parameters(), lr=0.01)

num_epochs = 10

for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        model2 = model2.train()
        # forward
        logits = model2(features)
        cost = F.binary_cross_entropy(logits, targets.view(-1, 1))

        # backward
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # logging
        # if not batch_idx % 50:
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d} | "
                    f"Batch {batch_idx:03d}/{len(train_loader):03d} | "
                    f"Cost: {cost:.4f}")


