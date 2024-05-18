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
  timestamp : int

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
    timestamp = input_dictionary['timestamp']
    ending_frame = round((timestamp / 1000) * 30)
    print("Ending frame: ", ending_frame)
    starting_frame = ending_frame - 50
    if starting_frame < 0:
      starting_frame = 0
    print("Starting frame: ", starting_frame)
    number = predict_jersey(input_dictionary['video'], tap_coords, starting_frame, ending_frame)
    return {"jersey_number": number}