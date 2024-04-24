from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from predict_jersey import predict_jersey
from pydantic import BaseModel
import json

class model_input(BaseModel):
  video : str
  x : int
  y : int

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict_jersey_number(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    tap_coords = (input_dictionary['x'], input_dictionary['y'])
    number = predict_jersey(input_dictionary['video'], tap_coords)
    return {"jersey_number": number}