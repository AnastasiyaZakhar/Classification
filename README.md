# Название проекта
Распознавание изображений двух групп: рыбы и цветы
![alt text](https://github.com/AnastasiyaZakhar/Classification/blob/main/img/Main_picture.png)

## Оглавление
1. [Цель проекта](#цельпроекта)
2. [Методы](#методы)
3. [Структура](#структура)
4. [Запуск](#запуск)

## Цель проекта <a name="цельпроекта"></a>
Данный проект предназначен для создания модели машинного обучения, которая могла бы распознавать картинки двух групп: те, на которых изображены **рыбы** и те, на которых изображены **цветы**. Другими словами требуется реализовать алгоритм ***компьютерного зрения***. 

## Методы <a name="методы"></a>
1. Для создания подобного классификатора использовалась модель машинного обучения **CatCatBoost** - это контролируемый метод машинного обучения, который используется инструментом Train Using AutoML и использует деревья решений для классификации и регрессии.
2. Также использовалась библиотека **OpenCV** (открытая библиотека для работы с алгоритмами компьютерного зрения, машинным обучением и обработкой изображений) для выявления дополнительных признаков исходных изображений. 

## Структура <a name="структура"></a>
Проект включает в себя следующие основные этапы:  
1. Загрузка исходных данных из исходного датасета CIFAR 100.
2. EDA исследование.
3. Отбор данных только для двух необходимых групп.
4. Feature Engineering (нормализация, LabelEncoding).
5. Создание прототипа модели.  
    5.1 Инициализация исходных моделей.  
    5.2 Обучение моделей.  
    5.3 Оценка и выбор наилучшей модели по показателям точности (accuracy, precision, recall, f1 score).  
6. Дополнительный Feature Engineering.  
    6.1 Добавление к признакам канала R.  
    6.2 Добавление к признакам канала G.  
    6.3 Добавление к признакам канала B.  
    6.4 Добавление к признакам черно-белых характеристик.  
7. Обучение моделей с новыми признаками.  
8. Применение нейронной сети.
9. Настройка модели и валидация.   
    9.1 Кросс-валидация.  
    9.2 Настройка гиперпараметров.  
    9.3 Проверка модели на валидационных данных.  
10. Окончательная версия модели.  
    10.1 Проверка на тестовом наборе данных.  
    10.2 Запись финальной модели в файл.  

## Запуск <a name="запуск"></a>
1. Запуск модели осуществляется через веб-сервис FastAPI.
2. Для запуска модели машинного обучения необходимо загрузить картинку на веб-сервисе в формате 32х32 пикселей.

