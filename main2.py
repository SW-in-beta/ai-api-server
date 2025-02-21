from typing import Union
from fastapi import APIRouter, FastAPI
from model import Model
from operation import BinaryOperation, UnaryOperation

binary_models = dict(
    [(operation.value, Model(operation.value)) for operation in BinaryOperation]
)
unary_models = dict(
    [(operation.value, Model(operation.value)) for operation in UnaryOperation]
)

app = FastAPI()
predict_router = APIRouter(prefix="/predict")


def get_model(operation: str):
    operation = operation.upper()
    return (
        binary_models[operation]
        if operation in BinaryOperation
        else unary_models[operation]
    )


@app.get("/")
def read_root():
    return {"Hello": "World"}

@predict_router.get("/")
def predict():
    return {"result": 'predict!'}

@predict_router.get("/{operation}/left/{left}/right/{right}")
def predict_binary(operation: BinaryOperation, left: int, right: int):
    model = get_model(operation.value)
    result = model.predict([left, right])
    return {"result": result}


@predict_router.get("/{operation}/{value}")
def predict_unary(operation: UnaryOperation, value: int):
    model = get_model(operation)
    result = model.predict([value])
    return {"result": result}


@app.get("/train/{operation}")
def train(operation: str):
    model = get_model(operation)
    model.train()
    return {"result": f"{operation} train success"}


@app.get("/reset/{operation}")
def reset(operation: str):
    model = get_model(operation)
    model.reset()
    return {"result": f"{operation} Model Reset"}


app.include_router(predict_router)