import pandas as pd
# from app.data.getCounty import getCounty
import time

import requests

def getCounty(lat, lon):
    """
    Given latitude and longitude, return a dictionary with county and state info using the FCC Census Block API.
    Returns a dict like {'state': {'name': ...}, 'county': {'name': ...}} or None if not found.
    """
    url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat}&longitude={lon}&format=json"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None
        data = response.json()
        state_name = data.get('State', {}).get('name')
        county_name = data.get('County', {}).get('name')
        if state_name and county_name:
            return {'state': {'name': state_name}, 'county': {'name': county_name}}
    except Exception as e:
        print(f"Error in getCounty: {e}")
    return None


# Read the fire data
fire_df = pd.read_csv('fire_df.csv')

# Prepare a list to store county names
counties = []

for idx, row in fire_df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    county_info = getCounty(lat, lon)
    if county_info and 'county' in county_info:
        counties.append(county_info['county']['name'])
    else:
        counties.append(None)
    # To avoid hitting the FCC API rate limit
    time.sleep(0.2)

# Add the county column
fire_df['county'] = counties

# Save the new DataFrame
fire_df.to_csv('fire_df_with_county.csv', index=False)
print('Saved fire_df_with_county.csv with county information.') 