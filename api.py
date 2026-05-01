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
scaler = joblib.load('scaler.pkl')  # загружаем scaler

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
    
    # Масштабируем!
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