from fastapi import FastAPI
from pydantic import BaseModel 
import pickle 
from core.config import settings
from catboost import CatBoostClassifier
import numpy as np

# Создаем приложение FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION
    )

# Загружаем предварительно обученную модель
with open(r'C:\Users\v_v_z\Classification\models\final_model.sav', 'rb') as model_file: 
    final_model = pickle.load(model_file)

@app.get('/')
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict/")
def predict(data: dict):
       X1 = np.array(data['X1']).reshape(1, -1)
       prediction = final_model.predict(X1)
       class_name = data.target_names[prediction][0]
       return {"class": class_name}
