import torch
from model import Model
from operation import BinaryOperation, UnaryOperation
from os.path import exists

def get_models():
    binary_models = dict(
		[(operation.value, load_model(operation.value)) for operation in BinaryOperation]
	)
    unary_models = dict(
		[(operation.value, load_model(operation.value)) for operation in UnaryOperation]
	)
    return binary_models, unary_models

def load_model(operation: str):
    operation = operation.upper()
    model = Model(operation)
    if exists(f'models/{operation}_state_dict.pth'):
        print(f'Yes! {operation} Loading!')
        model.load_state_dict(torch.load(f'models/{operation}_state_dict.pth'))
    return model