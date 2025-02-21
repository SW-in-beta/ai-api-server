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
        print(f"Predictions: {predictions} ({round(predictions)})")


# class Model:
#     def __init__(self, operation='AND'):
#         self.weights = np.random.rand(2) if operation in BinaryOperation else np.random.rand(1)
#         self.bias = np.random.rand(1)
#         self.operation  = operation.upper()

#     @property
#     def dataset(self):
#         return datasets['binary' if self.operation in BinaryOperation else 'unary']

#     @property
#     def labels(self):
#         return labels[self.operation]

#     def train(self):
#         learning_rate = 0.1
#         epochs = 20
#         for _ in range(epochs):
#             for i in range(len(self.dataset)):
#                 # 총 입력 계산
#                 total_input = np.dot(self.dataset[i], self.weights) + self.bias
#                 # 예측 출력 계산
#                 prediction = self.step_function(total_input)
#                 # 오차 계산
#                 error = self.labels[i] - prediction
#                 print(f'self.dataset[i] : {self.dataset[i]}')
#                 print(f'weights : {self.weights}')
#                 print(f'bias before update: {self.bias}')
#                 print(f'prediction: {prediction}')
#                 print(f'error: {error}')
#                 # 가중치와 편향 업데이트
#                 self.weights += learning_rate * error * self.dataset[i]
#                 self.bias += learning_rate * error
#                 print('====')

#     def step_function(self, x):
#         return 1 if x >= 0 else 0

#     def reset(self):
#         self.weights = np.random.rand(2)
#         self.bias = np.random.rand(1)

#     def predict(self, input_data):
#         total_input = np.dot(input_data, self.weights) + self.bias
#         return self.step_function(total_input)
