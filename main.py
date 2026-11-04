from pathlib import Path
from flask import app
import joblib
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import numpy as np
import os
from contextlib import asynccontextmanager
from typing import Callable

#MLFLOW Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI","http://127.0.0.1:7001"
)
#model:/iris-classifier@champion

MODEL_URI = "models:/iris-classifier@champion"

def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    return model

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

# Load the championmodel at application startup

#MODEL_PATH = Path("models/iris_random_forest_model.joblib")
#model = joblib.load(MODEL_PATH)
def create_app(model_loader:Callable=load_mlflow_model,version_loader:Callable=None):
    if(version_loader is None):
        def version_loader():
            client = mlflow.MlflowClient(
                tracking_uri=MLFLOW_TRACKING_URI
            )
            champion_version = client.get_model_version_by_alias(name="iris-classifier",alias="champion")
            return champion_version.version
        
    @asynccontextmanager
    async def lifespan(app:FastAPI):
        try:
            app.state.model=model_loader()
        except Exception as ex:
            raise RuntimeError(f"Failed to load model:{ex}")
        app.state.model_version=version_loader()
        yield
        #print(f"Loaded model: iris-classifier version: {model_version}")
    app = FastAPI(title="Iris Random Forest API",
                  description="Iris predition API powered by MLFLOW",
                  version="3.0.0",
                  lifespan=lifespan)
    @app.get("/")
    def ready_to_serve():
        return {"status": "Ready to serve"}

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "model_name": "iris-classification", "model_version": app.state.model_version}

    # Endpoint for making predictions with the Iris Random Forest model
    @app.post("/predict")
    def predict(data: IrisInput,request:Request):
        model = request.app.state.model
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]
        prediction = int(model.predict(features)[0])
        print("Prediction is",prediction)
        return {
            "prediction": prediction,
            "class_name": ["setosa", "versicolor", "virginica"][prediction]
        }
    return app
#Procutin application instance
app=create_app()
# try:
#     model = mlflow.sklearn.load_model(MODEL_URI)
#     client = mlflow.MlflowClient()
#     champion_version = client.get_model_version_by_alias(
#         name="iris-classifier",alias="champion"
#     )
#     model_version = champion_version.version
#     print(f"Loaded model: iris-classifier version: {model_version}")
# except Exception as e:
#     raise RuntimeError(f"Could not load champion model from MLflow: {e}")
