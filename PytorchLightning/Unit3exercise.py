import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

import pandas as pd

# set seed
torch.manual_seed(42)

class MyDataLoader(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def standardize(x):
    return (x - x.mean()) / x.std()

#-----------------#
df = pd.read_csv(
    "/home/bhaskar/mtsc/Pytorch-Learning/PytorchLightning/data_banknote_authentication.txt"
)
# print(df.head())

model = LogisticRegression(input_dim=4, output_dim=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

x, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
train_loader = DataLoader(MyDataLoader(x, y), batch_size=16, shuffle=True)

num_epochs = 10
# batch_size = 16

model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # 0. Standardize the data
        data = standardize(data)

        # 1. Generate predictions
        pred = model(data)

        # 2. Calculate loss
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(pred, target.view(-1, 1))

        # 3. Calculate gradients
        loss.backward()

        # 4. Update parameters using gradients
        optimizer.step()

        # 5. Log accuracy
        if batch_idx % 10 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len 
                (train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(),
            ))

model.eval()
#-----------------#
# accuracy
with torch.no_grad():
    pred = model(torch.tensor(x, dtype=torch.float32))
    pred = torch.sigmoid(pred)
    pred = torch.round(pred)
    print((pred == torch.tensor(y).view(-1, 1)).sum().item() / len(y))

preds = DataLoader




