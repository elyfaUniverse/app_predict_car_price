
import streamlit as st
import pandas as pd
import pickle
import re
import json
import plotly.express as px
import joblib
import numpy as np

# ЗАГРУЗКА МОДЕЛИ 
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

model, FEATURE_COLS = load_model()
st.success(f"Model loaded. Features: {', '.join(FEATURE_COLS)}")

def preprocess_input(year, km_driven, mileage_str, engine_str, max_power_str, torque_str, seats):
    df = pd.DataFrame([{
        'year': year,
        'km_driven': km_driven,
        'mileage': mileage_str,
        'engine': engine_str,
        'max_power': max_power_str,
        'torque': torque_str,
        'seats': seats
    }])
    df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['engine'] = df['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)
    torque_numbers = df['torque'].str.findall(r'(\d+\.?\d*)')
    df['torque'] = torque_numbers.apply(lambda x: float(x[0]) if x else 0)
    df['max_torque_rpm'] = torque_numbers.apply(lambda x: float(x[-1]) if len(x) > 1 else 0)
    return df[FEATURE_COLS].fillna(0)

def predict_price_from_dict(data_dict):
    X = preprocess_input(
        year=data_dict['year'],
        km_driven=data_dict['km_driven'],
        mileage_str=data_dict['mileage'],
        engine_str=data_dict['engine'],
        max_power_str=data_dict['max_power'],
        torque_str=data_dict['torque'],
        seats=data_dict['seats']
    )
    return model.predict(X)[0]

# ВЫБОР РЕЖИМА 
mode = st.radio("Выберите режим ввода",
                        ("Форма", "JSON"))

#  РЕЖИМ ФОРМЫ 
if mode == "Форма":
    st.title("🚗 Предсказание цены автомобиля")
    st.markdown("Введите данные")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year / Год выпуска", min_value=1980, max_value=2025, value=2015, step=1)
        km_driven = st.number_input("Kilometers driven / Пробег (км)", min_value=0, value=55000, step=1000)
        seats = st.number_input("Number of seats / Количество мест", min_value=2, max_value=9, value=5, step=1)
    with col2:
        mileage = st.text_input("Mileage (km/l) / Расход топлива (км/л)", value="18.2", help="Example: 18.2 or 18.2 kmpl")
        engine = st.text_input("Engine (cc) / Объём двигателя (куб. см)", value="1197", help="Example: 1197 or 1197 CC")
        max_power = st.text_input("Max power (bhp) / Мощность (л.с.)", value="82", help="Example: 82 or 82 bhp")
        torque = st.text_input("Torque (Nm) / Крутящий момент (Нм)", value="115Nm@4000rpm", help="Example: 115 or 115Nm@4000rpm")

    if st.button("Предсказать"):
        try:
            price = predict_price_from_dict({
                'year': year,
                'km_driven': km_driven,
                'mileage': mileage,
                'engine': engine,
                'max_power': max_power,
                'torque': torque,
                'seats': seats
            })
            st.success(f"Предсказанная цена: {price:.0f} руб.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    #  ВИЗУАЛИЗАЦИЯ ВЕСОВ МОДЕЛИ 
    st.markdown("---")
    st.subheader("Коэффициенты линейной регрессии")
    
    coefs = pd.DataFrame({
        'Признак': FEATURE_COLS,
        'Коэффициент': model.coef_
    }).sort_values('Коэффициент', ascending=False)
    
    st.dataframe(coefs, use_container_width=True)

    # ГРАФИК ЗАВИСИМОСТИ 
    st.subheader("Зависимость цены от года выпуска")
    years = list(range(1990, 2026))
    
    prices_by_year = []
    for y in years:
        price = predict_price_from_dict({
            'year': y,
            'km_driven': km_driven,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'torque': torque,
            'seats': seats
        })
        prices_by_year.append(price / 1000)
    
    fig1 = px.line(x=years, y=prices_by_year, 
                   labels={'x': 'Год выпуска', 'y': 'Цена (тыс. руб)'},
                   markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Зависимость цены от пробега")
    km_values = list(range(0, 300001, 10000))
    
    prices_by_km = []
    for km in km_values:
        price = predict_price_from_dict({
            'year': year,
            'km_driven': km,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'torque': torque,
            'seats': seats
        })
        prices_by_km.append(price / 1000)
    
    fig2 = px.line(x=km_values, y=prices_by_km,
                   labels={'x': 'Пробег ', 'y': 'Цена '},
                   markers=True)
    st.plotly_chart(fig2, use_container_width=True)
   
    st.subheader("Зависимость цены от мощности")
    power_values = list(range(30, 251, 5))
    
    prices_by_power = []
    for p in power_values:
        price = predict_price_from_dict({
            'year': year,
            'km_driven': km_driven,
            'mileage': mileage,
            'engine': engine,
            'max_power': str(p),
            'torque': torque,
            'seats': seats
        })
        prices_by_power.append(price / 1000)
    
    fig3 = px.line(x=power_values, y=prices_by_power,
                   labels={'x': 'Мощность ', 'y': 'Цена '},
                   markers=True)
    st.plotly_chart(fig3, use_container_width=True)
  
# РЕЖИМ JSON 
else:
    st.title("📄 JSON Input / Ввод JSON")
    st.markdown("Paste a JSON object with car data. All fields are required.")
    example_json = {
        "year": 2015,
        "km_driven": 55000,
        "mileage": "18.2",
        "engine": "1197",
        "max_power": "82",
        "torque": "115Nm@4000rpm",
        "seats": 5
    }
    st.json(example_json)
    
    json_str = st.text_area("JSON string / JSON строка", height=200, value=json.dumps(example_json, indent=2))
    
    if st.button("Предсказать из JSON"):
        try:
            data = json.loads(json_str)
            required_fields = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats']
            missing = [f for f in required_fields if f not in data]
            if missing:
                st.error(f"Missing fields: {missing}")
            else:
                price = predict_price_from_dict(data)
                st.success(f"💵 Predicted price: {price:,.0f} руб.")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except Exception as e:
            st.error(f"Prediction error: {e}")