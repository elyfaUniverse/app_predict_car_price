import streamlit as st
import pandas as pd
import json
import plotly.express as px
import joblib
import numpy as np
import os

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("Предсказание цены автомобиля")
st.markdown("---")

# ЗАГРУЗКА МОДЕЛИ
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, feature_cols, scaler

# ЗАГРУЗКА ДАННЫХ ДЛЯ ГРАФИКОВ
@st.cache_data
def load_plot_data():
    if os.path.exists('data_for_plot.json'):
        with open('data_for_plot.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        st.error("❌ Файл data_for_plot.json не найден!")
        return None

# Загружаем всё
model, FEATURE_COLS, SCALER = load_model()
plot_data = load_plot_data()

if plot_data:
    st.success(f" Модель Ridge загружена. Признаков: {len(FEATURE_COLS)}")

# ФУНКЦИЯ ПРЕДСКАЗАНИЯ 
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
    
    X = df[FEATURE_COLS].fillna(0)
    X_scaled = SCALER.transform(X)
    return X_scaled

def predict_price_from_dict(data_dict):
    X_scaled = preprocess_input(
        year=data_dict['year'],
        km_driven=data_dict['km_driven'],
        mileage_str=data_dict['mileage'],
        engine_str=data_dict['engine'],
        max_power_str=data_dict['max_power'],
        torque_str=data_dict['torque'],
        seats=data_dict['seats']
    )
    return model.predict(X_scaled)[0]

# ФОРМА ВВОДА ДАННЫХ
st.header("Введите данные автомобиля")
    
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Год выпуска", min_value=1980, max_value=2025, value=2015, step=1)
    km_driven = st.number_input("Пробег (км)", min_value=0, value=60000, step=5000)
    seats = st.number_input("Количество мест", min_value=2, max_value=9, value=5, step=1)

with col2:
    mileage = st.text_input("Расход топлива (км/л)", value="18.5")
    engine = st.text_input("Объём двигателя (куб. см)", value="1500")
    max_power = st.text_input("Мощность (л.с.)", value="100")
    torque = st.text_input("Крутящий момент", value="115Nm@4000rpm")

if st.button("Предсказать цену", type="primary"):
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
        st.success(f"Предсказанная цена: **{price:,.0f} ₽**")
        
        # ТАБЛИЦА КОЭФФИЦИЕНТОВ МОДЕЛИ
        if plot_data and 'coefficients' in plot_data:
            st.markdown("---")
            st.subheader("Коэффициенты модели Ridge регрессии")
            
          
            df_coef_table = pd.DataFrame(plot_data['coefficients'])
            df_coef_table = df_coef_table.sort_values('coefficient', ascending=False)
            df_coef_table['abs_coefficient'] = df_coef_table['coefficient'].abs()
            df_coef_table['importance'] = df_coef_table['coefficient'] / df_coef_table['coefficient'].abs().max()
            df_coef_table['coefficient_formatted'] = df_coef_table['coefficient'].apply(lambda x: f"{x:,.2f}")
            
            st.dataframe(
                df_coef_table[['feature', 'coefficient_formatted', 'importance']].rename(columns={
                    'feature': 'Признак',
                    'coefficient_formatted': 'Коэффициент',
                    'importance': 'Относительная важность'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            
            
    except Exception as e:
        st.error(f"Ошибка: {e}")

st.markdown("---")

#ГРАФИКИ 
if plot_data:
    col3, col4 = st.columns(2)
    
    with col3:
        df_year = pd.DataFrame(plot_data['year_dependency'])
        fig1 = px.line(df_year, x='year', y='price_thousands',
                       labels={'year': 'Год выпуска', 'price_thousands': 'Цена (тыс. ₽)'},
                       markers=True,
                       title="Зависимость цены от года выпуска")
        fig1.update_traces(line=dict(color='green', width=3))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col4:
        df_km = pd.DataFrame(plot_data['km_dependency'])
        fig2 = px.line(df_km, x='km_driven', y='price_thousands',
                       labels={'km_driven': 'Пробег (км)', 'price_thousands': 'Цена (тыс. ₽)'},
                       markers=True,
                       title="Зависимость цены от пробега")
        fig2.update_traces(line=dict(color='blue', width=3))
        st.plotly_chart(fig2, use_container_width=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        df_power = pd.DataFrame(plot_data['power_dependency'])
        fig3 = px.line(df_power, x='max_power', y='price_thousands',
                       labels={'max_power': 'Мощность (л.с.)', 'price_thousands': 'Цена (тыс. ₽)'},
                       markers=True,
                       title="Зависимость цены от мощности двигателя")
        fig3.update_traces(line=dict(color='red', width=3))
        st.plotly_chart(fig3, use_container_width=True)
    
    with col6:
        df_coef = pd.DataFrame(plot_data['coefficients'])
        df_coef = df_coef.sort_values('coefficient', ascending=False)
        
        fig4 = px.bar(df_coef, x='coefficient', y='feature', orientation='h',
                      labels={'coefficient': 'Коэффициент', 'feature': 'Признак'},
                      title="Коэффициенты Ridge регрессии")
        st.plotly_chart(fig4, use_container_width=True)
    
    # Информация о модели
    with st.expander(" Информация о модели"):
        st.write(f"**Тип модели:** {plot_data['stats']['model_type']}")
        st.write(f"**Alpha (регуляризация):** {plot_data['stats']['alpha']}")
        st.write(f"**Количество признаков:** {plot_data['stats']['n_features']}")
        st.write(f"**Свободный член (intercept):** {plot_data['stats']['intercept']:,.0f}")
        
        st.write("**Признаки модели:**")
        for i, feat in enumerate(plot_data['stats']['features']):
            st.write(f"  {i+1}. {feat}")
        
        st.write("**Базовые параметры для графиков:**")
        st.json(plot_data['base_params'])

else:
    st.error(" Не удалось загрузить данные для графиков. Убедитесь, что файл data_for_plot.json находится в той же папке.")

st.markdown("---")

#РЕЖИМ JSON
with st.expander("Или введите данные в формате JSON"):
    example_json = {
        "year": 2015,
        "km_driven": 60000,
        "mileage": "18.5",
        "engine": "1500",
        "max_power": "100",
        "torque": "115Nm@4000rpm",
        "seats": 5
    }
    st.json(example_json)
    
    json_str = st.text_area("JSON строка", height=150, value=json.dumps(example_json, indent=2))
    
    col7, col8 = st.columns([1, 5])
    with col7:
        if st.button("Предсказать из JSON"):
            try:
                data = json.loads(json_str)
                required_fields = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats']
                missing = [f for f in required_fields if f not in data]
                if missing:
                    st.error(f"Отсутствуют поля: {missing}")
                else:
                    price = predict_price_from_dict(data)
                    st.success(f" Предсказанная цена: {price:,.0f} ₽")
                    
                    if plot_data and 'coefficients' in plot_data:
                        st.markdown("---")
                        st.subheader("Коэффициенты модели Ridge регрессии")
                        df_coef_table = pd.DataFrame(plot_data['coefficients'])
                        df_coef_table = df_coef_table.sort_values('coefficient', ascending=False)
                        df_coef_table['coefficient_formatted'] = df_coef_table['coefficient'].apply(lambda x: f"{x:,.2f}")
                        st.dataframe(
                            df_coef_table[['feature', 'coefficient_formatted']].rename(columns={
                                'feature': 'Признак',
                                'coefficient_formatted': 'Коэффициент'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
            except json.JSONDecodeError as e:
                st.error(f"Ошибка JSON: {e}")
            except Exception as e:
                st.error(f"Ошибка: {e}")

st.markdown("---")
