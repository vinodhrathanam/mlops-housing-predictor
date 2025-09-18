import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc, os

# os.environ['MLFLOW_TRACKING_URI'] = "file:///C:/Users/vrlpr/mlruns"
# Load the latest production model from MLflow
mlflow.set_tracking_uri("http://localhost:5000")
logged_model = 'runs:/b794bc4e8a124333bd71c8b1b9199d8c/housing_model'  # Replace with your actual Run ID
model = mlflow.pyfunc.load_model(logged_model)

app = FastAPI()

class HousingData(BaseModel):
    area: float
    bedrooms: float
    bathrooms: float

@app.post("/predict")
def predict(data: HousingData):
    input_data = [[data.area, data.bedrooms, data.bathrooms]]
    prediction = model.predict(input_data)[0]
    return {"predicted_price": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)