from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from predict_jersey import predict_jersey
from pydantic import BaseModel
import json
from pyngrok import ngrok, conf
import nest_asyncio

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

if __name__ == "__main__":
  print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/get-started/your-authtoken")
  conf.get_default().auth_token = "2f12n34CoKYGCV5XWBXCgwZ6Frx_4vpCZthbo8DF4eHgkpVoA"

  ngrok_tunnel = ngrok.connect(8000)
  print('Public URL:', ngrok_tunnel.public_url)
  nest_asyncio.apply()
  uvicorn.run(app, port=8000)