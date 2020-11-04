# YouGate
## Проект распознавания автомобильных номеров
Данный проект нацелен на детекцию и распознавание автомобильных номеров. Для наглядности мы обернули нашу технологию в виде Telegram бота.

<img src="https://user-images.githubusercontent.com/36695287/98160838-9b3f2580-1f00-11eb-8bf1-986b792cb878.jpg" alt="1" width="250"/> <img src="https://user-images.githubusercontent.com/36695287/98160842-9d08e900-1f00-11eb-9909-80e27d7bae65.jpg" alt="2" width="250"/> <img src="https://user-images.githubusercontent.com/36695287/98160844-9da17f80-1f00-11eb-9ba9-f562653b89ea.jpg" alt="3" width="250"/>

## Запуск
1. Необходимо скачать модель детекции по ссылке [здесь](https://drive.google.com/file/d/13qNb-OfK8DmZVhSORy--Jho9tNVRWXgx/view?usp=sharing)
2. Добавить модель детекции в папку `models`
3. Установить зависимости: `pip install -r requirements.txt`
4. Запустить скрипт телеграм бота: `python bot.py`

## Датасеты
* Детекция: более **8000** размеченных автомобильных номеров
* OCR: более **10000** размеченных автомобильных номеров

## Метрики:
* Детекция: **mAP@IoU:0.5 = 0.96-0.98**
* OCR: **Accuracy = 97%**

### TODO:
* Super resolution
* Улучшение точности OCR
