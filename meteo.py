import requests
import pandas as pd

LAT = 25.5941
LON = 85.1376

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": LAT,
    "longitude": LON,
    "daily": "temperature_2m_max",
    "timezone": "Asia/Kolkata",
    "past_days": 92,
    "forecast_days": 0
}

data = requests.get(url, params=params).json()

actual = pd.DataFrame({
    "date": data["daily"]["time"],
    "actual_temp_max": data["daily"]["temperature_2m_max"]
})

print(actual)