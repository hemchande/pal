import requests
import pandas as pd
from io import StringIO
# from datetime import datetime
import numpy as np
import requests
import pandas as pd
from io import StringIO

from sklearn.cluster import DBSCAN
from geopy.distance import geodesic

# --- Population lookup function ---
def get_population_for_county_or_city(name, state, census_df):
    # Normalize input
    norm_name = name.lower().replace('county', '').replace('parish', '').strip()
    norm_state = state.lower().strip()
    # Filter for the correct state
    df_state = census_df[census_df['STNAME'].str.lower().str.strip() == norm_state]
    # Try to match county/city name
    matches = df_state[df_state['NAME'].str.lower().str.replace('county', '').str.replace('parish', '').str.strip() == norm_name]
    if matches.empty:
        # Try partial match
        matches = df_state[df_state['NAME'].str.lower().str.contains(norm_name)]
    if not matches.empty:
        print(f"DEBUG: Matched row for {name}, {state}:")
        print(matches.iloc[0])
        print("DEBUG: Columns:", matches.columns)
        if 'POPESTIMATE2024' in matches.columns:
            value = matches.iloc[0]['POPESTIMATE2024']
            print(f"DEBUG: Population value for {name}, {state}: {value} (type: {type(value)})")
            if pd.isna(value) or value is None:
                return None
            return int(value)
    return None

# --- FCC API county lookup with retry ---
def getCounty(lat, lon, max_retries=3):
    url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat}&longitude={lon}&format=json"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue
            data = response.json()
            state_name = data.get('State', {}).get('name')
            county_name = data.get('County', {}).get('name')
            if state_name and county_name:
                return {'state': {'name': state_name}, 'county': {'name': county_name}}
        except Exception as e:
            print(f"Error in getCounty (attempt {attempt+1}): {e}")
    return None


def process_fire_data_for_date(date_str: str):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/da956afb6784ae668cd90d05eb50c59f/VIIRS_SNPP_NRT/world/1/{date_str}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data (status {response.status_code}): {response.text}")
    df = pd.read_csv(StringIO(response.text))

    texas_df = df[(df['latitude'] >= 25.8) & (df['latitude'] <= 36.5) & (df['longitude'] >= -106.7) & (df['longitude'] <= -93.5)]

    census_df = pd.read_csv("https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/cities/totals/sub-est2024_48.csv")

    counties, states, populations = [], [], []
    for idx, row in texas_df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        county_data = getCounty(lat, lon)
        if county_data:
            county_name, state_name = county_data['county']['name'], county_data['state']['name']
            pop = get_population_for_county_or_city(county_name, state_name, census_df)
            counties.append(county_name)
            states.append(state_name)
            populations.append(pop)
        else:
            counties.append(None)
            states.append(None)
            populations.append(None)

    texas_df['county'] = counties
    texas_df['state'] = states
    texas_df['population'] = populations

    texas_df['datetime'] = pd.to_datetime(texas_df['acq_date'] + ' ' + texas_df['acq_time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')

    county_fire_counts = texas_df.groupby('county').size().rename('fire_count')
    county_pop = texas_df.groupby('county')['population'].first()
    county_fire_rate = (county_fire_counts / county_pop * 100000).replace([np.inf, -np.inf], np.nan)

    coords = texas_df[['latitude', 'longitude']].values
    if len(coords) > 0:
        db = DBSCAN(eps=0.1, min_samples=5).fit(coords)
        texas_df['cluster'] = db.labels_
    else:
        texas_df['cluster'] = -1

    spread_stats = []
    for cluster_id in texas_df['cluster'].unique():
        if cluster_id == -1:
            continue
        cluster_fires = texas_df[texas_df['cluster'] == cluster_id].sort_values('datetime')
        if len(cluster_fires) < 2:
            continue
        distances = []
        time_deltas = []
        for i in range(1, len(cluster_fires)):
            loc1 = (cluster_fires.iloc[i-1]['latitude'], cluster_fires.iloc[i-1]['longitude'])
            loc2 = (cluster_fires.iloc[i]['latitude'], cluster_fires.iloc[i]['longitude'])
            dist_km = geodesic(loc1, loc2).km
            time_hr = (cluster_fires.iloc[i]['datetime'] - cluster_fires.iloc[i-1]['datetime']).total_seconds() / 3600
            if time_hr > 0:
                distances.append(dist_km)
                time_deltas.append(time_hr)
        if distances and time_deltas:
            avg_speed_kmh = np.sum(distances) / np.sum(time_deltas)
            total_distance = np.sum(distances)
            spread_stats.append({
                'cluster': cluster_id,
                'avg_speed_kmh': avg_speed_kmh,
                'total_distance_km': total_distance,
                'num_fires': len(cluster_fires)
            })

    spread_df = pd.DataFrame(spread_stats)
    county_summary = pd.DataFrame({
        'fire_count': county_fire_counts,
        'population': county_pop,
        'fire_rate_per_100k': county_fire_rate
    }).reset_index()

    texas_df.to_csv(f'texas_fires_{date_str}_with_population.csv', index=False)
    spread_df.to_csv(f'fire_cluster_spread_stats_{date_str}.csv', index=False)
    county_summary.to_csv(f'fire_county_summary_{date_str}.csv', index=False)

    return texas_df, spread_df, county_summary




def process_fire_data_for_date2(date_str: str):
    """
    Download and process fire data for a given date in YYYY-MM-DD format.
    Outputs a CSV with Texas fire incidents enriched with population data.
    """
    # Build URL dynamically from provided date
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"da956afb6784ae668cd90d05eb50c59f/VIIRS_SNPP_NRT/world/1/{date_str}"
    )
    
    # Fetch fire data
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data (status {response.status_code}): {response.text}")
    
    csv_content = response.text
    df = pd.read_csv(StringIO(csv_content))
    print(df.head())

    # Filter for Texas
    texas_df = df[(df['latitude'] >= 25.8) & (df['latitude'] <= 36.5) &
                  (df['longitude'] >= -106.7) & (df['longitude'] <= -93.5)]

    print(f"Number of fires in Texas: {len(texas_df)}")

    # Load census population data
    census_df = pd.read_csv("https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/cities/totals/sub-est2024_48.csv")

    if 'NAME' in census_df.columns and 'STNAME' in census_df.columns:
        unique_places = census_df[['NAME', 'STNAME']].drop_duplicates()
        print("Sample counties/cities in the census dataset:")
        print(unique_places.head())
    else:
        print("Unexpected column names in census data:", census_df.columns)

    # Enrich fire data with county/state/population
    fire_df = texas_df.copy()
    counties, states, populations = [], [], []
    for idx, row in fire_df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        county_data = getCounty(lat, lon)
        if county_data:
            county_name = county_data['county']['name']
            state_name = county_data['state']['name']
            pop = get_population_for_county_or_city(county_name, state_name, census_df)
            counties.append(county_name)
            states.append(state_name)
            populations.append(pop)
            print(f"Row {idx}: lat={lat}, lon={lon}, county={county_name}, state={state_name}, population={pop}")
        else:
            counties.append(None)
            states.append(None)
            populations.append(None)
            print(f"Row {idx}: lat={lat}, lon={lon}, county=None, state=None, population=None")

    # Save enriched dataset
    output_df = fire_df.copy()
    output_df['county'] = counties
    output_df['state'] = states
    output_df['population'] = populations
    output_filename = f"texas_fires_{date_str}_with_population.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}")



# 1. Download/process fire data
# url = (
#     "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
#     "da956afb6784ae668cd90d05eb50c59f/VIIRS_SNPP_NRT/world/1/2025-07-18"
# )
# response = requests.get(url)
# if response.status_code != 200:
#     raise Exception(f"Failed to fetch data (status {response.status_code}): {response.text}")
# csv_content = response.text
# df = pd.read_csv(StringIO(csv_content))
# print(df.head())

# # Filter for Texas: lat 25.8 to 36.5, lon -106.7 to -93.5
# texas_df = df[(df['latitude'] >= 25.8) & (df['latitude'] <= 36.5) &
#               (df['longitude'] >= -106.7) & (df['longitude'] <= -93.5)]

# print(f"Number of fires in Texas: {len(texas_df)}")

# # Download census CSV once
# census_df = pd.read_csv("https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/cities/totals/sub-est2024_48.csv")

# # Print all unique county/city and state names in the census dataset
# print("All counties/cities in the census dataset:")
# if 'NAME' in census_df.columns and 'STNAME' in census_df.columns:
#     unique_places = census_df[['NAME', 'STNAME']].drop_duplicates()
#     for idx, row in unique_places.iterrows():
#         print(f"{row['NAME']}, {row['STNAME']}")
# else:
#     print("Column names not as expected. Columns are:", census_df.columns)

# # Only focus on FCC county lookup and population for now
# fire_df = texas_df.copy()

# counties, states, populations = [], [], []
# for idx, row in fire_df.iterrows():
#     lat, lon = row['latitude'], row['longitude']
#     county_data = getCounty(lat, lon)
#     if county_data:
#         county_name = county_data['county']['name']
#         state_name = county_data['state']['name']
#         pop = get_population_for_county_or_city(county_name, state_name, census_df)
#         counties.append(county_name)
#         states.append(state_name)
#         populations.append(pop)
#         print(f"Row {idx}: lat={lat}, lon={lon}, county={county_name}, state={state_name}, population={pop}")
#     else:
#         counties.append(None)
#         states.append(None)
#         populations.append(None)
#         print(f"Row {idx}: lat={lat}, lon={lon}, county=None, state=None, population=None")

# # Keep all other processing commented out
# # fire_df['datetime'] = pd.to_datetime(
# #     fire_df['acq_date'] + ' ' + fire_df['acq_time'].astype(str).str.zfill(4),
# #     format='%Y-%m-%d %H%M'
# # )
# # fire_df = fire_df[fire_df['datetime'] > fire_df['datetime'].max() - pd.Timedelta("2 days")]

# # 2. Enrich with county/state info and population sequentially
# # counties, states, populations = [], [], []
# # for idx, row in fire_df.iterrows():
# #     lat, lon = row['latitude'], row['longitude']
# #     county_data = getCounty(lat, lon)
# #     if county_data:
# #         county_name = county_data['county']['name']
# #         state_name = county_data['state']['name']
# #         pop = get_population_for_county_or_city(county_name, census_df)
# #         counties.append(county_name)
# #         states.append(state_name)
# #         populations.append(pop)
# #     else:
# #         counties.append(None)
# #         states.append(None)
# #         populations.append(None)
# # fire_df['county'] = counties
# # fire_df['state'] = states
# # fire_df['county_population'] = populations
# # fire_df.to_csv("fire_df_enriched.csv", index=False)

# # 3. Compute additional fire statistics
# # county_fire_counts = fire_df.groupby('county').size().rename('fire_count')
# # county_pop = fire_df.groupby('county')['county_population'].first()
# # county_fire_rate = (county_fire_counts / county_pop * 100000).replace([np.inf, -np.inf], np.nan)
# # coords = fire_df[['latitude', 'longitude']].values
# # if len(coords) > 0:
# #     db = DBSCAN(eps=0.1, min_samples=5).fit(coords)
# #     fire_df['cluster'] = db.labels_
# # else:
# #     fire_df['cluster'] = -1
# # fire_df['datetime'] = pd.to_datetime(fire_df['datetime'])
# # spread_stats = []
# # for cluster_id in fire_df['cluster'].unique():
# #     if cluster_id == -1:
# #         continue
# #     cluster_fires = fire_df[fire_df['cluster'] == cluster_id].sort_values('datetime')
# #     if len(cluster_fires) < 2:
# #         continue
# #     distances = []
# #     time_deltas = []
# #     for i in range(1, len(cluster_fires)):
# #         loc1 = (cluster_fires.iloc[i-1]['latitude'], cluster_fires.iloc[i-1]['longitude'])
# #         loc2 = (cluster_fires.iloc[i]['latitude'], cluster_fires.iloc[i-1]['longitude'])
# #         dist_km = geodesic(loc1, loc2).km
# #         time_hr = (cluster_fires.iloc[i]['datetime'] - cluster_fires.iloc[i-1]['datetime']).total_seconds() / 3600
# #         if time_hr > 0:
# #             distances.append(dist_km)
# #             time_deltas.append(time_hr)
# #     if distances and time_deltas:
# #         avg_speed_kmh = np.sum(distances) / np.sum(time_deltas)
# #         total_distance = np.sum(distances)
# #         spread_stats.append({
# #             'cluster': cluster_id,
# #             'avg_speed_kmh': avg_speed_kmh,
# #             'total_distance_km': total_distance,
# #             'num_fires': len(cluster_fires)
# #         })
# # spread_df = pd.DataFrame(spread_stats)
# # spread_df.to_csv('fire_cluster_spread_stats.csv', index=False)

# # county_summary = pd.DataFrame({
# #     'fire_count': county_fire_counts,
# #     'population': county_pop,
# #     'fire_rate_per_100k': county_fire_rate
# # }).reset_index()
# # county_summary.to_csv('fire_county_summary.csv', index=False)

# # fire_df.to_csv('fire_df_with_clusters.csv', index=False)
# # print('Pipeline complete. Outputs: fire_df_enriched.csv, fire_df_with_clusters.csv, fire_county_summary.csv, fire_cluster_spread_stats.csv')

# # After the main loop
# output_df = fire_df.copy()
# output_df['county'] = counties
# output_df['state'] = states
# output_df['population'] = populations
# output_df.to_csv('texas_fires_with_population.csv', index=False)
# print('Saved results to texas_fires_with_population.csv') 





#datestring 

datestring = "2025-07-26"

process_fire_data_for_date(datestring)

#2025-07-18