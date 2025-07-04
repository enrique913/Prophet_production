import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import timedelta
from prophet.make_holidays import get_country_holidays_class
import openpyxl
st.set_page_config(layout="wide")
st.title("Forecast de llamadas con Prophet")

# Subir archivo
uploaded_file = st.file_uploader("Sube tu archivo Excel con columnas 'date' y 'calls'", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    if "date" not in df.columns or "calls" not in df.columns:
        st.error("El archivo debe contener columnas 'date' y 'calls'")
    else:
        df = df.dropna(subset=["date", "calls"])
        df = df[df["calls"] != 0]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df_prophet = df[["date", "calls"]].rename(columns={"date": "ds", "calls": "y"})

        # Parámetros ajustables
        st.sidebar.header("Hiperparámetros del modelo")
        growth = st.sidebar.selectbox("Tipo de crecimiento", ["linear", "logistic"])
        changepoint_prior_scale = st.sidebar.slider("changepoint_prior_scale", 0.0001, 0.5, 0.05)
        seasonality_prior_scale = st.sidebar.slider("seasonality_prior_scale", 1.0, 50.0, 10.0)
        seasonality_mode = st.sidebar.selectbox("Modo de estacionalidad", ["additive", "multiplicative"])
        forecast_days = st.sidebar.slider("Días a predecir", 30, 180, 90)

        # Capacidad para crecimiento logístico
        if growth == "logistic":
            cap_value = st.sidebar.number_input("Capacidad máxima (cap)", value=float(df_prophet["y"].max() * 1.2))
            df_prophet["cap"] = cap_value

        # Festivos
        start_year = df_prophet["ds"].min().year
        end_year = (df_prophet["ds"].max() + timedelta(days=forecast_days)).year
        years = list(range(start_year, end_year + 1))
        holidays_df = pd.DataFrame()
        for country in ["US", "CA"]:
            country_class = get_country_holidays_class(country)
            holidays = country_class(years=years)
            df_holidays = pd.DataFrame({
                'ds': pd.to_datetime(list(holidays.keys())),
                'holiday': [', '.join(holidays.get_list(date)) for date in holidays]
            })
            holidays_df = pd.concat([holidays_df, df_holidays], ignore_index=True)

        # Crear modelo
        model = Prophet(
            growth=growth,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            holidays=holidays_df
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df_prophet)

        # Crear fechas futuras
        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        if growth == "logistic":
            future["cap"] = cap_value

        forecast = model.predict(future)

        # Mostrar gráficos
        st.subheader("Pronóstico")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("Componentes del modelo")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # Exportar resultados
        result = forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "forecast"})
        calls_map = df_prophet.set_index("ds")["y"].to_dict()
        result["calls"] = result["date"].map(calls_map).fillna(0)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result.to_excel(writer, index=False, sheet_name="Forecast")
        st.download_button("Descargar forecast en Excel", data=output.getvalue(), file_name="forecast.xlsx")
