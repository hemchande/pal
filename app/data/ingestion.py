import requests
import pandas as pd
from io import StringIO
from datetime import datetime
from app.data.getCounty import getCounty

# URL for FIRMS VIIRS data (NRT, global, 1-day window as of 2025‑07‑18)
url = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "da956afb6784ae668cd90d05eb50c59f/VIIRS_SNPP_NRT/world/1/2025-07-18"
)

response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Failed to fetch data (status {response.status_code}): {response.text}")

# Parse CSV content into a DataFrame
csv_content = response.text
df = pd.read_csv(StringIO(csv_content))

# Quick preview
print(df.head())

# Load cleaned VIIRS fire data
fire_df = df.copy()

# Convert acquisition datetime
fire_df['datetime'] = pd.to_datetime(
    fire_df['acq_date'] + ' ' + fire_df['acq_time'].astype(str).str.zfill(4),
    format='%Y-%m-%d %H%M'
)

# Filter recent fires (last 48 hours)
fire_df = fire_df[fire_df['datetime'] > fire_df['datetime'].max() - pd.Timedelta("2 days")]

# Enrich with county and state info
counties = []
states = []
for idx, row in fire_df.iterrows():
    lat, lon = row['latitude'], row['longitude']
    county_data = getCounty(lat, lon)
    if county_data:
        counties.append(county_data['county']['name'])
        states.append(county_data['state']['name'])
    else:
        counties.append(None)
        states.append(None)

fire_df['county'] = counties
fire_df['state'] = states

# Save to CSV for downstream processing
fire_df.to_csv("fire_df_enriched.csv", index=False)
print("Enriched fire data with county and state saved to fire_df_enriched.csv") 