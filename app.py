import modules.my_module
import requests 
import numpy as np
from matplotlib import pyplot as plt
import PIL
import cv2
import os

# Определяем URL адрес
url = 'http://127.0.0.1:8000/flowers_or_fish'
  
# Ввод картинки 

imageFileName = raw_input("Введите имя файла с картинкой: ")

path = os.path.join(image_folder, "gray.jpg")

image = cv2.imread(path)
  
# Делаем запрос  
response = requests.post(url, data=image) 
  
# Ответ модели 
print(response.text)

print(var_test)
