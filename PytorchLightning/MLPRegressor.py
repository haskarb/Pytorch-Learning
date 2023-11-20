#Create a MLPRegressor using Pytorch

# Importing the libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#Create the MLPRegressor class
class MLPRegressor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.all_layers = nn.Sequential(
            nn.Linear (num_features, 50, dtype = torch.float64),
            nn.ReLU(),

            # nn.Linear(50, 25),
            # nn.ReLU(),

            nn.Linear(50, 1),
        )
        
    def forward(self, x):
        logits =  self.all_layers(x).flatten()
        return logits


#Create DataLoaders
class MLPRegDataLoader(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.features = X
        self.targets = y

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        
        return x, y
    
    def __len__(self):
        return self.targets.shape[0]



# read dataset
import numpy as np

X = np.random.rand(10000, 100)
y = np.random.rand(10000, 10)

# convert to torch format
X = torch.from_numpy(X)
y = torch.from_numpy(y) 


# train mode
model = MLPRegressor(num_features = X.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)


# create object
train = MLPRegDataLoader(X, y)

train_loader =  DataLoader(
    dataset= train,
    batch_size = 16,
    shuffle=True
)

num_epochs  = 1
for epoch in range(num_epochs):

    for batch_id, (x, y) in enumerate(train_loader):
        print(x.dtype)
        logits = model(x)
        loss = nn.functional.mse_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(batch_id)
    # pass
