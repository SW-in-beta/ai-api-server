import numpy as np
from operation import BinaryOperation, UnaryOperation

datasets = {
    'unary': np.array([[0], [1]]),
    'binary': np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
}

labels = {
    'AND': np.array([0, 0, 0, 1]),
    'OR': np.array([0, 1, 1, 1]),
    'NOT': np.array([1, 0]),
    'NAND': np.array([1, 1, 1, 0]),
    'NOR': np.array([1, 0, 0, 0]),
}

class Model:
    def __init__(self, operation='AND'):
        self.weights = np.random.rand(2) if operation in BinaryOperation else np.random.rand(1)
        self.bias = np.random.rand(1)
        self.operation  = operation.upper()
    
    @property
    def dataset(self):
        return datasets['binary' if self.operation in BinaryOperation else 'unary']
    
    @property    
    def labels(self):
        return labels[self.operation]
        
    def train(self):
        learning_rate = 0.1
        epochs = 20
        for _ in range(epochs):
            for i in range(len(self.dataset)):
                # 총 입력 계산
                total_input = np.dot(self.dataset[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = self.labels[i] - prediction
                print(f'self.dataset[i] : {self.dataset[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * self.dataset[i]
                self.bias += learning_rate * error
                print('====')        

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def reset(self):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)