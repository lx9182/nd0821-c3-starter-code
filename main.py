from fastapi import FastAPI
from pydantic import BaseModel
from constants import cat_features
from starter.train_model import online_inference


class Value(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str


app = FastAPI()


@app.get("/")
def greeting():
    return "Greeting!"


@app.post('/inference')
async def predict_income(input: Value):
    model_path = 'model/model.pkl'
    input = {key.replace('_', '-'): value for key, value
             in input.dict().items()}
    return {"salary": online_inference(model_path, input, cat_features)}
