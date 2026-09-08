import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import torch
import plotly.graph_objects as go
from src.cvae import PatchTST_CVAE

st.set_page_config(page_title="Thesis: Weather CVAE Live Inference", layout="wide")
st.title("☁️ Live CVAE Forecasting (Probabilistic)")

# Load Metadata & Model
@st.cache_resource
def load_resources():
    with open("checkpoints/city_meta.json", "r") as f:
        city_meta = json.load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchTST_CVAE(pred_len=14, d_model=64, z_dim=16)
    model.load_state_dict(torch.load("checkpoints/cvae_z16_14d.pt", map_location=device))
    model.to(device)
    model.eval()
    
    return city_meta, model, device

try:
    city_meta, model, device = load_resources()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# Sidebar Inputs
import datetime

with st.sidebar:
    st.header("Location Settings")
    st.markdown("Current coordinates point to Patna, India.")
    lat = st.number_input("Latitude", value=25.5941)
    lon = st.number_input("Longitude", value=85.1376)
    city_profile = st.selectbox("Apply City Embedding Profile (Closest climate)", list(city_meta.keys()), index=list(city_meta.keys()).index("Lucknow"))
    
    st.header("Time Settings")
    target_date = st.date_input("Target Date (Forecast Origin)", value=datetime.date.today())
    
    st.markdown("---")
    run_inference = st.button("Run Live Inference")

if run_inference:
    with st.spinner("Fetching data from Open-Meteo..."):
        start_date = target_date - datetime.timedelta(days=30)
        end_date = target_date + datetime.timedelta(days=13)
        
        # Open-Meteo historical API is required for dates older than 90 days
        if start_date < datetime.date.today() - datetime.timedelta(days=90):
            url = "https://archive-api.open-meteo.com/v1/archive"
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "timezone": "Asia/Kolkata",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        res = requests.get(url, params=params)
        
        if res.status_code == 200:
            data = res.json()
            dates = data["daily"]["time"]
            temps = data["daily"]["temperature_2m_max"]
            
            df = pd.DataFrame({"date": pd.to_datetime(dates), "temp": temps})
            
            # The API returns the exact range requested (30 days past + 14 days future = 44 days)
            context_df = df.iloc[:30].copy()
            future_df = df.iloc[30:44].copy()
            
            if len(context_df) < 30 or len(future_df) < 14:
                st.error(f"Not enough data fetched from the API for the selected date range. Got {len(df)} days total.")
                st.stop()
            
            # Normalization
            meta = city_meta[city_profile]
            c_id = meta["id"]
            c_mean = meta["mean"]
            c_std = meta["std"]
            
            context_raw = context_df['temp'].values.astype(np.float32)
            context_norm = (context_raw - c_mean) / c_std
            
            # Prepare tensors
            x_t = torch.tensor(context_norm).unsqueeze(0).unsqueeze(-1).to(device) # [1, 30, 1]
            c_t = torch.tensor([c_id], dtype=torch.int64).to(device) # [1]
            
            # Sample from CVAE
            with torch.no_grad():
                samples = model.sample(x_t, c_t, n_samples=100) # [100, 1, 14, 1]
            
            samples_np = samples[:, 0, :, 0] # [100, 14]
            
            # Un-normalize
            samples_real = samples_np * c_std + c_mean
            
            # Calculate Median and 90% CI
            median_pred = np.median(samples_real, axis=0)
            p95 = np.percentile(samples_real, 95, axis=0)
            p05 = np.percentile(samples_real, 5, axis=0)
            
            # PLOTTING
            fig = go.Figure()
            
            # Past Truth
            fig.add_trace(go.Scatter(
                x=context_df['date'], y=context_df['temp'],
                mode='lines+markers', name='Past Actual (30 days)',
                line=dict(color='black')
            ))
            
            # API Future / Forecast (Baseline)
            fig.add_trace(go.Scatter(
                x=future_df['date'], y=future_df['temp'],
                mode='lines+markers', name='Open-Meteo Actual/Forecast (14 days)',
                line=dict(color='gray', dash='dash')
            ))
            
            # CVAE Confidence Interval Shading
            fig.add_trace(go.Scatter(
                x=future_df['date'].tolist() + future_df['date'].tolist()[::-1],
                y=p95.tolist() + p05.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='CVAE 90% Confidence Interval'
            ))
            
            # CVAE Median
            fig.add_trace(go.Scatter(
                x=future_df['date'], y=median_pred,
                mode='lines+markers', name=f'CVAE Median Forecast',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title=f"14-Day Probabilistic Forecast from {target_date} (Lat: {lat}, Lon: {lon}, Profile: {city_profile})",
                xaxis_title="Date",
                yaxis_title="Max Temperature (°C)",
                hovermode="x unified",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw Data Table
            st.subheader("Forecast Data Summary")
            res_df = future_df.copy()
            res_df['Open_Meteo_Actual'] = res_df['temp']
            res_df['CVAE_Median'] = median_pred
            res_df['CVAE_Lower_5%'] = p05
            res_df['CVAE_Upper_95%'] = p95
            res_df = res_df.drop(columns=['temp'])
            st.dataframe(res_df.style.format({col: "{:.2f}" for col in res_df.columns if col != 'date'}))
            
        else:
            st.error(f"Failed to fetch data from Open-Meteo API. Status Code: {res.status_code}")
            st.write(res.text)
