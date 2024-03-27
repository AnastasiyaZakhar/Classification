from fastapi import FastAPI
from pydantic import BaseModel 
import pickle 
from core.config import settings
from catboost import CatBoostClassifier

# Создаем приложение FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION
    )

# Загружаем предварительно обученную модель
with open(r'C:\Users\v_v_z\VENV\models\final_model.sav', 'rb') as model_file: 
    final_model = pickle.load(model_file)


@app.get('/')
def get_hello():
    return {"msg":"Привет FastAPI"}


