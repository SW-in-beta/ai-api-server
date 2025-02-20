import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from operation import BinaryOperation, UnaryOperation

datasets = {
    "unary": torch.tensor(np.array([[0], [1]], dtype=np.float32)),
    "binary": torch.tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)),
}

labels = {
    "AND": torch.tensor(np.array([[0], [0], [0], [1]], dtype=np.float32)),
    "OR": torch.tensor(np.array([[0], [1], [1], [1]], dtype=np.float32)),
    "NAND": torch.tensor(np.array([[1], [1], [1], [0]], dtype=np.float32)),
    "NOR": torch.tensor(np.array([[1], [0], [0], [0]], dtype=np.float32)),
    "XOR": torch.tensor(np.array([[0], [1], [1], [0]], dtype=np.float32)),
    "NOT": torch.tensor(np.array([[1], [0]], dtype=np.float32)),
}

class Model(nn.Module):
    def __init__(self, operation):
        super(Model, self).__init__()
        self.operation = operation.upper()
        self.layer1 = nn.Linear(2 if self.operation in BinaryOperation else 1, 2)
        self.layer2 = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.x = datasets["binary" if self.operation in BinaryOperation else "unary"]
        self.y = labels[self.operation]

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x
      
    def reset(self):
        self.layer1 = nn.Linear(2 if self.operation in BinaryOperation else 1, 2)
        self.layer2 = nn.Linear(2, 1)
        
    def save(self):
        torch.save(self.state_dict(), f'models/{self.operation}_state_dict.pth')


def train(model: Model, *, epochs=2000, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(model.x)
        loss = criterion(outputs, model.y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    if evaluate(model) < 0.9:
        print("Retraining...")
        model.reset()
        train(model, epochs=epochs, lr=lr)
        
    model.save()
      

def evaluate(model: Model):
    criterion = nn.BCELoss()

    model.eval()
    with torch.no_grad():
        outputs = model(model.x)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == model.y).float().mean()
        loss = criterion(outputs, model.y)
        print(f"Loss: {loss.item()}, Accuracy: {accuracy.item()}")
    return accuracy.item()


def predict(model: Model, input_data):
    tensor = torch.tensor(np.array(input_data, dtype=np.float32))
    with torch.no_grad():
        predictions = model(tensor).item()
        return round(predictions)