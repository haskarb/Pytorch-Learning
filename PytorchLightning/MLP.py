import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

df = pd.read_csv("PytorchLightning/data/xor.csv")
print(df.head())
X, y = df[["x1", "x2"]].values, df["class label"].values
# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# create validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1, stratify=y_train
)
print(X_train.shape, X_val.shape, X_test.shape)


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            torch.nn.Linear(num_features, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, num_classes),
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits


class MyDataLoader(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

# computer accuracy
def compute_accuracy(model, data_loader):

    model = model.eval()

    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
            with torch.inference_mode():
                logits = model(features)
            predicted_labels = torch.argmax(logits, 1)

            compare = predicted_labels == targets

            correct_pred += torch.sum(compare)
            num_examples += predicted_labels.size(0)
    return correct_pred.float()/num_examples * 100



train_ds = MyDataLoader(X_train, y_train)
val_ds = MyDataLoader(X_val, y_val)
test_ds = MyDataLoader(X_test, y_test)

# create dataloaders
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)


# train model
torch.manual_seed(1)
model = MLP(num_features=2, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 10
train_losses = []
val_losses = []
for epoch in range(num_epochs):

    # training
    model.train()
    batch_losses = []
    
    for i, (X_batch, y_bat) in enumerate(train_dl):

        # 1. forward propagation
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_bat)

        # 2. backward propagation
        optimizer.zero_grad()
        loss.backward()

        # 3. weight update
        optimizer.step()


        #logging
        if not i % 10:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.4f}")
        batch_losses.append(loss.item())

print("Accuracy: ", compute_accuracy(model, test_dl))


from matplotlib.colors import ListedColormap
import numpy as np


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('D', '^', 'x', 's', 'v')
    colors = ('C0', 'C1', 'C2', 'C3', 'C4')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    tensor = torch.tensor(np.array([xx1.ravel(), xx2.ravel()]).T).float()
    logits = classifier.forward(tensor)
    Z = np.argmax(logits.detach().numpy(), axis=1)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, color=cmap(idx),
                    #edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

plot_decision_regions(X_train, y_train, classifier=model)