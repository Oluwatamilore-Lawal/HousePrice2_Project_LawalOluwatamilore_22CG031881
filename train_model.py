import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Use only MedInc (feature 0) and AveRooms (feature 5) for this example
# Scaling y to reasonable house price values (assuming in hundred thousands)
X = X[:, [0, 5]]
y = y * 100  # Scale target to reasonable values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# Define PyTorch model
class HousePriceModel(nn.Module):
    def __init__(self):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = HousePriceModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate model
with torch.no_grad():
    train_outputs = model(X_train)
    train_loss = criterion(train_outputs, y_train)
    
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    
    print(f"\nTraining Loss (MSE): {train_loss:.4f}")
    print(f"Testing Loss (MSE): {test_loss:.4f}")

# Save model and scaler
torch.save(model.state_dict(), "model.pth")
import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel saved as 'model.pth'")
print("Scaler saved as 'scaler.pkl'")
