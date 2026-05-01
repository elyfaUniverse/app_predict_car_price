# 🚗 Отчёт по проекту: прогнозирование цены автомобиля

### 👉 Приложение на Streamlit  
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://appredictcarprice.streamlit.app/)

### Видео работы сервисов
- FastAPI (CSV): [Смотреть](https://cloud.mail.ru/public/ZKAp/ZWH2LjzD9)  
- FastAPI (JSON): [Смотреть](https://cloud.mail.ru/public/2C8j/q12xNSTAN)  
- Streamlit-приложение: [Смотреть](https://cloud.mail.ru/public/SG9g/MEEhhq562)

---

## 1. Что я сделала

### 1.1. Анализ данных
Я загрузила тренировочный (`cars_train.csv`, 6999 объектов) и тестовый (`cars_test.csv`, 1000 объектов) датасеты. Посмотрела на данные с помощью обычных описательных статистик и быстрого профиля через `ydata-profiling`.  
**Главные наблюдения:**
- Половина машин — 2014 года или новее.
- Цены от 30 тысяч до 10 миллионов, но большинство стоит около 450 тысяч. Дорогие авто задирают среднюю до 640 тысяч.
- Средний пробег примерно 70 000 км, встречались машины с пробегом больше 2 млн км.
- Топливо почти пополам: дизель 54%, бензин 44%, газ редко.
- Продавцы в основном частные (83%), коробка передач на 87% ручная.
- Пропуски были в `mileage`, `engine`, `max_power`, `torque`, `seats`, но их немного (2–3%). Полностью одинаковых строк (дубликатов) не нашлось.
- Числовые признаки `mileage`, `engine`, `max_power`, `torque` записаны как текст с единицами измерения.

Я построила корреляционные матрицы (Пирсон, Спирмен, phik).  
- **Сильнее всего с ценой связаны:** `max_power` (0.69), `torque` (0.61), `engine` (0.45), `year` (0.43).  
- Пробег и расход топлива почти не влияют на цену.  
- Между годом выпуска и пробегом корреляция –0.37 — старые машины действительно проехали больше.  
- `max_power` и `torque` сильно связаны друг с другом (0.74), но для бустингов это не проблема.

### 1.2. Предобработка данных
В колонках с единицами измерения я вытащила числа регулярными выражениями и привела к `float`.  
Крутящий момент (`torque`) разбила на две части: значение момента и обороты (`max_torque_rpm`).

**Пропуски** заполнила медианой, посчитанной **только на тренировочных данных**. Медиана не боится выбросов (в отличие от среднего), и из-за малого количества пропусков распределения совсем не сдвинулись.

Категориальные признаки (`fuel`, `seller_type`, `transmission`, `owner`, `name`) я закодировала OneHot-кодированием. Для `name` оставила первые два слова и топ-10 самых частых комбинаций, остальное заменила на «other». Получилось 35 признаков, но качество модели не улучшилось, поэтому в финале я оставила только 8 исходных числовых переменных.

Пробовала удалять машины с аномально высокой ценой (верхний 1%) — R² резко упал до 0.49, значит, выбросы несли полезную информацию. Логарифмирование цены сделало распределение красивее, но точность модели не поднялась.

### 1.3. Моделирование
Я обучила несколько моделей на 8 числовых признаках:  
- Линейная регрессия (LinearRegression)  
- Lasso  
- Ridge  
- ElasticNet  
- L0-регуляризация (отбор признаков)

Гиперпараметры подбирала с помощью `GridSearchCV` и кросс-валидации на 10 фолдов. Основные метрики — R², MSE и доля прогнозов с ошибкой не более 10% (обычная и «строгая» — когда заниженная цена штрафуется сильнее).

---

### 1.4. Сервис на FastAPI (код в api.py)

Сервис реализован в файле `api.py` с помощью **FastAPI** и **Pydantic**. В нём:

- Описан класс `CarData` (наследуется от `BaseModel`) — это модель для входных данных одного автомобиля.
- Есть функция `preprocess_item`, которая парсит текстовые поля, выделяет числа, добавляет недостающие признаки и масштабирует всё через заранее сохранённый `scaler`.
- Загружаются обученная модель `model.pkl`, список нужных признаков `feature_cols.pkl` и скейлер `scaler.pkl`.

**Эндпоинты:**

| Маршрут | Метод | Что принимает | Что возвращает |
|--------|------|---------------|----------------|
| `/predict` | POST | Один объект `CarData` (JSON) | `{"predicted_price": число}` |
| `/predict_batch` | POST | Список объектов `List[CarData]` (JSON) | `{"predictions": [числа]}` |
| `/predict_csv` | POST | CSV-файл (`UploadFile`) | CSV-файл с новым столбцом `predicted_price` |

Код самого сервиса (api.py):

```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import io

app = FastAPI(title="Car Price Prediction API")

# Загружаем модель и scaler
model = joblib.load('model.pkl')
feature_cols = joblib.load('feature_cols.pkl')
scaler = joblib.load('scaler.pkl')

class CarData(BaseModel):
    year: int
    km_driven: int
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: int

def preprocess_item(data_dict: dict):
    df = pd.DataFrame([data_dict])
    df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['engine'] = df['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)
    torque_numbers = df['torque'].str.findall(r'(\d+\.?\d*)')
    df['torque'] = torque_numbers.apply(lambda x: float(x[0]) if x else 0)
    df['max_torque_rpm'] = torque_numbers.apply(lambda x: float(x[-1]) if len(x) > 1 else 0)
    
    X = df[feature_cols].fillna(0)
    
    # Масштабируем
    X_scaled = scaler.transform(X)
    return X_scaled

@app.post("/predict")
def predict_single(item: CarData):
    X = preprocess_item(item.dict())
    price = model.predict(X)[0]
    return {"predicted_price": float(price)}

@app.post("/predict_batch")
def predict_batch(items: List[CarData]):
    results = []
    for item in items:
        X = preprocess_item(item.dict())
        price = model.predict(X)[0]
        results.append(float(price))
    return {"predictions": results}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    predictions = []
    for _, row in df.iterrows():
        data = {
            'year': int(row['year']),
            'km_driven': int(row['km_driven']),
            'mileage': str(row['mileage']),
            'engine': str(row['engine']),
            'max_power': str(row['max_power']),
            'torque': str(row['torque']),
            'seats': int(row['seats'])
        }
        X = preprocess_item(data)
        price = model.predict(X)[0]
        predictions.append(price)
    
    df['predicted_price'] = predictions
    return df.to_csv(index=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Этот код выполняет всё, что требовалось в задании: принимает один объект и выдаёт предсказание, принимает CSV-файл и возвращает его с предсказаниями, а бонусом может обработать список объектов через `/predict_batch`.

---

### 1.5. Streamlit-приложение

Я сделала отдельное веб-приложение на **Streamlit**, чтобы пользователь мог легко попробовать модель без программирования.  
Оно загружает модель, скейлер и список признаков (те же `model.pkl`, `feature_cols.pkl`, `scaler.pkl`) и предлагает:

- **Вкладка «Одно предсказание»** — форма с полями (год, пробег, расход, двигатель, мощность, крутящий момент, кол‑во мест). После нажатия кнопки показывается предсказанная цена.
- **Вкладка «Загрузить CSV»** — загрузка CSV-файла с колонками как в тестовых данных и скачивание результата с дополнительным столбцом `predicted_price`.

В основе используется ровно та же функция `preprocess_item`, что и в FastAPI, поэтому предсказания полностью совпадают.

Приложение лежит в файле `app.py` (или `streamlit_app.py`). Зависимости: `streamlit`, `pandas`, `joblib`, `scikit-learn`.

#### Как запустить локально

1. Убедиться, что в папке проекта есть файлы:
   - `model.pkl`
   - `feature_cols.pkl`
   - `scaler.pkl`
   - `app.py` с кодом Streamlit-приложения

2. В терминале (с активным виртуальным окружением) выполнить:
   ```powershell
   streamlit run app.py
   ```
   После этого приложение откроется в браузере по адресу `http://localhost:8501`.

#### Как развернуть через GitHub (Streamlit Cloud)

1. Загрузить все необходимые файлы (`app.py`, `model.pkl`, `feature_cols.pkl`, `scaler.pkl`, `requirements.txt`) в **публичный репозиторий на GitHub**.
2. Файл `requirements.txt` должен содержать:
   ```
   streamlit
   pandas
   joblib
   scikit-learn
   ```
3. Перейти на [share.streamlit.io](https://share.streamlit.io) и авторизоваться с GitHub-аккаунтом.
4. Нажать «New app», выбрать репозиторий, ветку и главный файл (`app.py`).
5. Нажать «Deploy» — через пару минут приложение станет доступно по ссылке, которую можно дать кому угодно.

Моё развёрнутое приложение живёт по кнопке в самом верху отчёта.

---

## 2. Результаты

### Лучшая модель
Я сравнила несколько вариантов, и для деплоя выбрала **Ridge (α=1000)**, потому что она дала самую высокую долю прогнозов с ошибкой ≤10% — это важнее для бизнеса, чем просто R².

| Модель | R² train | R² test | Точных прогнозов (≤10%) | С учётом строгости* |
|--------|----------|---------|-------------------------|---------------------|
| LinearRegression | 0.6020 | **0.6028** | 0.2460 | 0.1870 |
| Lasso (α=1000)   | 0.577  | ~0.57   | 0.2460 | 0.1870 |
| **Ridge (α=1000)** | 0.570  | 0.5703  | **0.2610** | **0.1960** |
| ElasticNet (α=0.5, l1=0.7) | 0.583 | ~0.58 | 0.2580 | 0.1840 |
| L0 (5 признаков) | 0.6005 | 0.5983  | – | – |

*\*Строгая метрика: если модель недооценила машину (ошибка >5% вниз), такой прогноз считается плохим.*

### Важность признаков (линейная регрессия, стандартизированные веса)

| Признак | Вес (по модулю) |
|--------|----------------|
| `max_power` | 287 835 |
| `year` | 161 379 |
| `torque` | 64 762 |
| `km_driven` | 47 351 (отрицательный) |
| `max_torque_rpm` | 36 678 |
| `seats` | 30 767 |
| `engine` | 22 470 |
| `mileage` | 6 764 |

Мощность двигателя — самый главный фактор цены, за ней идут год выпуска и крутящий момент. Расход топлива и объём двигателя почти не играют роли.

---

## 3. Что дало наибольший прирост качества

- **Преобразование текстовых полей с единицами в числа** — без этого модели просто не завелись бы.
- **Сохранение дорогих машин-выбросов** — их удаление сильно ухудшило результат.
- **Восемь исходных числовых признаков** — любые дополнительные признаки (категории, квадраты) не помогли.
- **Отказ от логарифмирования цены** — не дало плюса.
- **Ridge с небольшим α** — чуть снизила R², зато подняла долю точных прогнозов, что важнее в реальной жизни.

---

## 🚀 Как запустить сервис FastAPI с нуля

### 1. Сохраняем модель и скейлер из Colab
После обучения лучшей модели выполняем в ноутбуке:

```python
from sklearn.linear_model import Ridge
import joblib
from google.colab import files

best_ridge = Ridge(alpha=1000, random_state=42)
best_ridge.fit(X_train_scaled, y_train)

joblib.dump(best_ridge, 'model.pkl')
feature_cols = X_train_scaled.columns.tolist()
joblib.dump(feature_cols, 'feature_cols.pkl')
joblib.dump(scaler, 'scaler.pkl')

files.download('model.pkl')
files.download('feature_cols.pkl')
files.download('scaler.pkl')
```

Забираем скачанные файлы в папку с проектом (туда же, где будет лежать `api.py`).

### 2. Настраиваем окружение и ставим библиотеки
В терминале (на Windows):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install fastapi uvicorn pandas numpy joblib scikit-learn
```

### 3. Кладём код сервиса в файл `api.py`
Копируем весь код из секции 1.4 в файл с именем `api.py`.

### 4. Запускаем сервер
```powershell
python -m uvicorn api:app --reload
```

После этого сервис будет доступен по адресу `http://localhost:8000`, а документация с возможностью тестирования — на `http://localhost:8000/docs`.

---

## 🔵 Streamlit-приложение: полный цикл

### Что используется
- Файл приложения: `app.py` (Streamlit)
- Те же артефакты: `model.pkl`, `feature_cols.pkl`, `scaler.pkl`

### Запуск локально
```powershell
streamlit run app.py
```

Приложение откроется в браузере на `http://localhost:8501`.

### Деплой через GitHub
1. Создать репозиторий, залить туда `app.py`, артефакты и `requirements.txt`.
2. [Подключить репозиторий к Streamlit Cloud](https://share.streamlit.io/) и нажать «Deploy».
3. Получить публичный URL, которым можно делиться.

---

## Итог

Я построила полный пайплайн: от анализа и очистки данных до двух работающих сервисов — FastAPI (для программного доступа) и Streamlit (для демонстрации и удобного интерфейса). Модель объясняет около 57–60% разброса цен и даёт приемлемую точность для каждого четвёртого автомобиля. Главные факторы стоимости — мощность, год выпуска и крутящий момент.
