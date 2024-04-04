from fastapi import FastAPI, UploadFile
import numpy as np
from PIL import Image
from core.config import settings
import pickle 
import uvicorn
from catboost import CatBoostClassifier

# Создаем приложение FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION
    )
# 'C:/Users/v_v_z/Classification/models/final_model.sav'
# Загружаем предварительно обученную модель
with open(r'./models/final_model.sav', 'rb') as model_file: 
    final_model = pickle.load(model_file)

@app.get('/')
def read_root():
    return {"message": "Добро пожаловать в приложение для распознавания Цветов и Рыб"}


@app.post("/predict")
def predict(file: UploadFile):  # Changed the function name
    # Read Image
    img = Image.open(file.file)
    img_array = np.array(img)

    # Normalize Features
    X1 = img_array/255
    X1 = X1.reshape(3072)

    # Run Inference
    prediction = final_model.predict([X1])

    # Get Raw Scores (note that tolist is needed because of FastAPI/json)
    prob = final_model.predict_proba([X1]).tolist()

    # Return Info
    # return {'pred' : prediction.tolist(),
    #         'prob' : prob,
    #         'feats': X1.tolist()}
    return 'Fish' if prediction == 0 else 'Flowers', prob

# #if __name__ == "__main__":
#     #uvicorn.run(app, port=8000, host='0.0.0.0', debug=True)
