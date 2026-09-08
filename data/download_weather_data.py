import requests
import pandas as pd
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CITIES = {
    'Delhi':     {'lat': 28.6139, 'lon': 77.2090},
    'Mumbai':    {'lat': 19.0760, 'lon': 72.8777},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
    'Chennai':   {'lat': 13.0827, 'lon': 80.2707},
    'Kolkata':   {'lat': 22.5726, 'lon': 88.3639},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
    'Pune':      {'lat': 18.5204, 'lon': 73.8567},
    'Jaipur':    {'lat': 26.9124, 'lon': 75.7873},
    'Lucknow':   {'lat': 26.8467, 'lon': 80.9462},
}

API_URL = "https://archive-api.open-meteo.com/v1/archive"

VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "weather_code",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant"
]

CHUNKS = [
    ("2000-01-01", "2004-12-31"),
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]

def get_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def fetch_weather_data():
    session = get_session()
    all_data = []

    for city, coords in CITIES.items():
        print(f"Fetching data for {city}...")
        for start_date, end_date in CHUNKS:
            params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "start_date": start_date,
                "end_date": end_date,
                "daily": ",".join(VARIABLES),
                "timezone": "Asia/Kolkata"
            }
            try:
                response = session.get(API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "daily" in data:
                    df = pd.DataFrame(data["daily"])
                    df["city"] = city
                    all_data.append(df)
                else:
                    print(f"No daily data found for {city} from {start_date} to {end_date}")
            except Exception as e:
                print(f"Error fetching data for {city} from {start_date} to {end_date}: {e}")
            
            # Rate limiting
            time.sleep(1)
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    df = fetch_weather_data()
    
    if not df.empty:
        # Save to csv
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, "indian_cities_weather_2000_2024.csv")
        df.to_csv(out_path, index=False)
        
        # Summary stats
        print(f"Shape: {df.shape}")
        print("\nMissing values:")
        print(df.isnull().sum())
        print(f"\nDate range: {df['time'].min()} to {df['time'].max()}")
        print(f"Data saved to {out_path}")
    else:
        print("No data was fetched.")
