from fastapi import FastAPI
import uvicorn
from predict_jersey import predict_jersey

app = FastAPI()

@app.post("/predict/")
async def predict_number(video:str, x:int, y:int):
    tap_coords = (x, y)
    number = predict_jersey(video, tap_coords)
    return {"jersey_number": number}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)