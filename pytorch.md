# PyTorch Fundamentals

Updated 2025-10-03

## Tensors

## Vectorized Operations

## Data Loading and Preprocessing

## Automatic Differentiation

## Neural Networks

## Training Loop

## GPU Acceleration

## Model Saving and Loading

## Training a Classifier Example

### Softmax
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# assume:
# X is a pandas DataFrame (n_samples, n_features)
# y is a 1D array/Series with integer labels in range 0..9

# convert pandas data to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class Model(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)  # returns raw logits

# specify dimensions
input_dim = X_tensor.shape[1]
hidden1 = 128
hidden2 = 64
num_classes = 10

model = Model(input_dim, hidden1, hidden2, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training loop
for epoch in range(50):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

# inference
with torch.no_grad():
    logits = model(X_tensor)
    preds = torch.argmax(logits, dim=1)

print("Predicted class counts:", torch.bincount(preds))

# to get probabilities:
probs = torch.softmax(logits, dim=1)
print("Probability shape:", probs.shape)
```

### Hidden Layers, Dropout, and LR Scheduling
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class Model(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes, dropout_prob):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden2, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

input_dim = X_tensor.shape[1]
hidden1 = 128
hidden2 = 64
num_classes = 10
dropout_prob = 0.5

model = Model(input_dim, hidden1, hidden2, num_classes, dropout_prob)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Scheduler: multiply lr by gamma every step_size epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(50):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
    # step the scheduler once per epoch
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, lr = {current_lr:.5f}")

with torch.no_grad():
    logits = model(X_tensor)
    preds = torch.argmax(logits, dim=1)

print("Predicted class counts:", torch.bincount(preds))
probs = torch.softmax(logits, dim=1)
print("Probability shape:", probs.shape)
```

### Train/Validation Split

A **proper train / validation / test split**, and logging of **loss on both training and validation sets** during training. 

* Splits your data into **train / validation / test**
* Trains a model with **two hidden layers + dropout**
* Uses a **learning rate scheduler**
* Computes **train and validation loss** each epoch
* Reports **test accuracy** at the end

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Convert DataFrame / labels to tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# === train / val / test splits ===
n = len(dataset)
n_test = int(0.2 * n)
n_val  = int(0.1 * n)  # e.g. 10% val, 20% test
n_train = n - n_val - n_test

train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], 
                                         generator=torch.Generator().manual_seed(42))

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# === model ===
class NeuralNetWithDropout(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes, dropout_prob):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden2, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

input_dim    = X_tensor.shape[1]
hidden1      = 128
hidden2      = 64
num_classes  = 10
dropout_prob = 0.5

model = NeuralNetWithDropout(input_dim, hidden1, hidden2, num_classes, dropout_prob)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# === training loop with val loss ===
num_epochs = 50

for epoch in range(num_epochs):
    # ---- train ----
    model.train()
    total_train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * xb.size(0)

    scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    # ---- validation ----
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total_val_loss += loss.item() * xb.size(0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)

    lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1:02d} | lr={lr:.5f} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

# === test accuracy after training ===
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

print(f"\nTest Accuracy: {correct/total:.4f}")
```

**Train/Val/Test**: 70/10/20

**Loss tracking**: We accumulate batch loss weighted by `batch_size` so that final averages reflect true dataset averages.

**Scheduler**: Called once per epoch, after training.

**Evaluation mode**: `model.eval()` disables dropout during validation and testing for fair loss/accuracy computation.
