import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic

# Load the Texas fires data
df = pd.read_csv('texas_fires_with_population.csv')

# --- 1. Spatial clustering (DBSCAN) ---
coords = df[['latitude', 'longitude']].values
db = DBSCAN(eps=0.1, min_samples=5).fit(coords)
df['cluster'] = db.labels_

# --- 2. Spread statistics for each cluster ---
df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')
spread_stats = []
for cluster_id in df['cluster'].unique():
    if cluster_id == -1:
        continue
    cluster_fires = df[df['cluster'] == cluster_id].sort_values('datetime')
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
spread_df.to_csv('texas_fire_cluster_spread_stats.csv', index=False)
print('Saved texas_fire_cluster_spread_stats.csv')

# --- 3. County-level summary statistics ---
county_stats = df.groupby('county').agg(
    fire_count=('county', 'count'),
    frp_sum=('frp', 'sum'),
    frp_mean=('frp', 'mean'),
    earliest_fire=('datetime', 'min'),
    latest_fire=('datetime', 'max'),
    unique_satellites=('satellite', 'nunique'),
    unique_instruments=('instrument', 'nunique'),
    day_fires=('daynight', lambda x: (x == 'D').sum()),
    night_fires=('daynight', lambda x: (x == 'N').sum()),
    population=('population', 'first')
).reset_index()
county_stats['day_night_ratio'] = county_stats['day_fires'] / county_stats['night_fires'].replace(0, 1)
county_stats = county_stats.sort_values(by='fire_count', ascending=False)
county_stats.to_csv('texas_fire_county_statistics.csv', index=False)
print('Saved texas_fire_county_statistics.csv')

# --- 4. Print key insights ---
print('Top 5 counties by fire count:')
print(county_stats[['county', 'fire_count', 'population']].head(5))
print('\nTop 5 clusters by number of fires:')
print(spread_df.sort_values('num_fires', ascending=False).head(5))
print('\nTop 5 clusters by average spread speed (km/h):')
print(spread_df.sort_values('avg_speed_kmh', ascending=False).head(5)) 