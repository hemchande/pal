import requests
from geopy.distance import geodesic
import math
import pandas as pd
import csv
import json

WIND_DIRECTION_DEGREES = {
    'N': 0,
    'NNE': 22.5,
    'NE': 45,
    'ENE': 67.5,
    'E': 90,
    'ESE': 112.5,
    'SE': 135,
    'SSE': 157.5,
    'S': 180,
    'SSW': 202.5,
    'SW': 225,
    'WSW': 247.5,
    'W': 270,
    'WNW': 292.5,
    'NW': 315,
    'NNW': 337.5
}


def load_counties_data(csv_path):
    counties = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            counties.append({
                "county": row['county'],
                "latitude": row['latitude'],
                "longitude": row['longitude'],
                "population":row['population']
            })
    return counties


NWS_BASE = "https://api.weather.gov"
AIR_QUALITY_URL = "https://airquality.googleapis.com/v1/currentConditions:lookup"
AIR_QUALITY_API_KEY = "AIzaSyCi_a_DkflqMRGcrFYvYKRB05XwNOFMmOg"  # Replace with your actual key

# 1. Get Weather Data + Fire Zone Geometry
def get_weather(lat, lon):
    try:
        headers = {"User-Agent": "fire-risk-pipeline-contact@example.com"}

        point_url = f"{NWS_BASE}/points/{lat},{lon}"
        point_res = requests.get(point_url, headers=headers).json()
        forecast_url = point_res['properties']['forecastHourly']
        fire_zone_url = point_res['properties']['fireWeatherZone']

        forecast_res = requests.get(forecast_url, headers=headers).json()
        first_hour = forecast_res['properties']['periods'][0]

        fire_zone_res = requests.get(fire_zone_url, headers=headers).json()
        fire_geometry = fire_zone_res.get('geometry', {})

        return {
            "temperature": first_hour['temperature'],
            "wind_speed": first_hour['windSpeed'],
            "wind_direction": first_hour['windDirection'],
            "humidity": first_hour.get('relativeHumidity', {}).get('value'),
            "heatIndex": first_hour.get('heatIndex', {}).get('value') or first_hour['temperature'],
            "fire_geometry": fire_geometry
        }
    except Exception as e:
        print(f"[Weather API Error] {e}")
        return {}
    


def parse_wind_direction(direction_str):
    """
    Converts a wind direction string to degrees.
    Handles both cardinal directions (e.g., 'SSE') and numeric degrees (e.g., '190°').
    """
    try:
        # Case 1: Raw numeric string like '190°'
        if direction_str.strip().endswith('°'):
            return float(direction_str.strip().replace('°', ''))
        # Case 2: Cardinal direction like 'SSE'
        direction = direction_str.strip().upper()
        return WIND_DIRECTION_DEGREES.get(direction, None)
    except Exception as e:
        print(f"[Wind Direction Parse Error] {e}")
        return None


# 2. Get Air Quality Data (Google API)
def get_air_quality(lat, lon):
    try:
        params = {"key": AIR_QUALITY_API_KEY}
        body = {"location": {"latitude": lat, "longitude": lon}}
        response = requests.post(AIR_QUALITY_URL, params=params, json=body)
        if response.status_code == 200:
            data = response.json()
            indexes = data.get('indexes', [])
            if indexes:
                aqi = indexes[0].get('aqi')
                category = indexes[0].get('category')
                dominant_pollutant = indexes[0].get('dominantPollutant', 'N/A')
            else:
                aqi = category = dominant_pollutant = None
            pollutants = data.get('pollutants', [])
            return {
                "aqi": aqi,
                "category": category,
                "dominant_pollutant": dominant_pollutant,
                "pollutants": pollutants
            }
        else:
            print(f"Air Quality API error: {response.text}")
            return {}
    except Exception as e:
        print(f"[Air API Error] {e}")
        return {}

# 3. Estimate Fire Spread Based on Wind Vector
def estimate_spread(lat, lon, wind_speed_kph, wind_direction_deg, distance_km=5):
    try:
        print("wind direction degree",wind_direction_deg)
        wind_direction_deg = parse_wind_direction(wind_direction_deg)
        rad = math.radians(wind_direction_deg)
        print("rad",rad)
        dx = distance_km * math.sin(rad)
        dy = distance_km * math.cos(rad)
        new_lat = lat + (dy / 111)
        new_lon = lon + (dx / (111 * math.cos(math.radians(lat))))
        return [{"lat": new_lat, "lon": new_lon}]
    except Exception as e:
        print(f"[Spread Estimation Error] {e}")
        return []

# 4. Get Nearby Counties with Weather + AQI
def get_nearby_counties(lat, lon, counties_data, radius_km=50):
    enriched = []
    for c in counties_data:
        dist = geodesic((lat, lon), (c['latitude'], c['longitude'])).km
        if dist <= radius_km:
            w = get_weather(c['latitude'], c['longitude'])
            aq = get_air_quality(c['latitude'], c['longitude'])
            enriched.append({
                "county": c['county'],
                "lat": c['latitude'],
                "lon": c['longitude'],
                "population": c.get('population'),
                "humidity": w.get('humidity'),
                "wind_speed": w.get('wind_speed'),
                "wind_direction": w.get('wind_direction'),
                "heat_index": w.get('heatIndex'),
                "temperature": w.get('temperature'),
                "air_quality": aq
            })
    return enriched

# 5. Final Enrichment per Fire Record
def enrich_fire_record(fire_row, counties_data):
    print("fire data",fire_row)
    lat, lon = fire_row['latitude'], fire_row['longitude']
    weather = get_weather(lat, lon)
    print("weather",weather)
    aq= get_air_quality(lat, lon)
    print("aq",aq)
    # weather = fire_row.get("weather", {})
    # aq = fire_row.get("aq", {})

    # Temperature
    temperature = weather.get("temperature", None)

    # Wind speed: '9 mph' → 9
    wind_speed = 0
    if "wind_speed" in weather and weather["wind_speed"]:
        try:
            wind_speed = int(weather["wind_speed"].split()[0])
        except (ValueError, IndexError):
            wind_speed = 0

    # Wind direction: 'SSE' or '190°' → handle cardinal or degree
    wind_direction = weather.get("wind_direction", None)

    # Humidity
    humidity = weather.get("humidity", None)

    # Heat Index
    heat_index = weather.get("heatIndex", None)

    # Fire polygon geometry
    fire_geometry = weather.get("fire_geometry", {})

    # AQI (Air Quality Index)
    aqi = aq.get("aqi", None)
    aq_category = aq.get("category", None)
    dominant_pollutant = aq.get("dominant_pollutant", None)

    # wind_speed = int(weather['wind_speed'].split()[0]) if weather.get('wind_speed') else 0
    # wind_direction = int(weather['wind_direction'].split()[0]) if weather.get('wind_direction') else 0

    spread = estimate_spread(lat, lon, wind_speed, wind_direction)
    enriched_nearby = get_nearby_counties(lat, lon, counties_data)

    return {
        "county": fire_row['county'],
        "population": fire_row['population'],
        "location": {"lat": lat, "lon": lon},
        "heat_index": heat_index,
        "humidity": humidity,
        "temperature": weather.get('temperature'),
        "wind": {
            "speed":wind_speed,
            "direction": wind_direction
        },
        "air_quality": aqi,
        "air_category":aq_category,
        "fire_zone_geometry": weather.get('fire_geometry'),
        "spread_prediction": spread,
        "nearby_areas": enriched_nearby
    }



counties_data = load_counties_data('texas_fires_2025-07-26_with_population.csv')
fireDfFinal = pd.read_csv("texas_fires_2025-07-26_with_population.csv")

# Enrich and collect records
enriched_records = []

for index, row in fireDfFinal.iterrows():
    enriched_record = enrich_fire_record(row, counties_data)
    enriched_records.append(enriched_record)

# Save to JSON file
with open("texas_fires_enriched_2025-07-26.json", "w") as f:
    json.dump(enriched_records, f, indent=2)

print("✅ Enriched fire data saved to 'texas_fires_enriched_2025-07-26.json'")
# # Load input data
# counties_data = load_counties_data('texas_fires_2025-07-26_with_population.csv')
# fireDfFinal = pd.read_csv("texas_fires_2025-07-26_with_population.csv")

# # Enrich and collect records
# enriched_records = []

# for index, row in fireDfFinal.iterrows():
#     enriched_record = enrich_fire_record(row, counties_data)
#     enriched_records.append(enriched_record)  # store in list

# # Convert to DataFrame
# enriched_df = pd.DataFrame(enriched_records)

# # Save to CSV
# enriched_df.to_csv("texas_fires_enriched_2025-07-26.csv", index=False)
# print("✅ Enriched fire data saved to 'texas_fires_enriched_2025-07-26.csv'")


# counties_data = load_counties_data('texas_fires_2025-07-26_with_population.csv')


# fireDfFinal = pd.read_csv("texas_fires_2025-07-26_with_population.csv")


# for index, row in fireDfFinal.iterrows():
#     enriched_record = enrich_fire_record(row, counties_data)
#     print("enriched record", enriched_record)

# for row in fireDfFinal.items():

# # Now you can call:
#     enriched_record = enrich_fire_record(row, counties_data)
#     print("enriched record",enriched_record)


# import requests
# from geopy.distance import geodesic
# import math

# NWS_BASE = "https://api.weather.gov"
# IQAIR_BASE = "http://api.airvisual.com/v2"
# IQAIR_KEY = "YOUR_IQAIR_API_KEY"  # Replace with your actual key

# # 1. Get Weather Data
# def get_weather(lat, lon):
#     try:
#         point_url = f"{NWS_BASE}/points/{lat},{lon}"
#         point_res = requests.get(point_url).json()
#         forecast_url = point_res['properties']['forecastHourly']

#         forecast_res = requests.get(forecast_url).json()
#         first_hour = forecast_res['properties']['periods'][0]  # Current hour

#         return {
#             "temperature": first_hour['temperature'],
#             "wind_speed": first_hour['windSpeed'],
#             "wind_direction": first_hour['windDirection'],
#             "humidity": first_hour.get('relativeHumidity', {}).get('value'),
#             "heatIndex": first_hour.get('heatIndex', {}).get('value') or first_hour['temperature']
#         }
#     except Exception as e:
#         print(f"[Weather API Error] {e}")
#         return {}

# # 2. Get Air Quality Data from IQAir
# def get_air_quality(lat, lon):
#     try:
#         url = f"{IQAIR_BASE}/nearest_city?lat={lat}&lon={lon}&key={IQAIR_KEY}"
#         res = requests.get(url).json()
#         data = res.get("data", {})
#         return {
#             "pm2_5": data.get("current", {}).get("pollution", {}).get("pm2_5"),
#             "aqi": data.get("current", {}).get("pollution", {}).get("aqius"),
#             "category": data.get("current", {}).get("pollution", {}).get("mainus", "Unknown")
#         }
#     except Exception as e:
#         print(f"[Air API Error] {e}")
#         return {}

# # 3. Estimate Fire Spread Based on Wind Vector
# def estimate_spread(lat, lon, wind_speed_kph, wind_direction_deg, distance_km=5):
#     rad = math.radians(wind_direction_deg)
#     dx = distance_km * math.sin(rad)
#     dy = distance_km * math.cos(rad)
#     new_lat = lat + (dy / 111)  # Approximate
#     new_lon = lon + (dx / (111 * math.cos(math.radians(lat))))
#     return [{"lat": new_lat, "lon": new_lon}]

# # 4. Get Nearby Counties with Enriched Weather + AQI
# def get_nearby_counties(lat, lon, counties_data, radius_km=50):
#     enriched = []
#     for c in counties_data:
#         dist = geodesic((lat, lon), (c['latitude'], c['longitude'])).km
#         if dist <= radius_km:
#             w = get_weather(c['latitude'], c['longitude'])
#             aq = get_air_quality(c['latitude'], c['longitude'])
#             enriched.append({
#                 "county": c['county'],
#                 "lat": c['latitude'],
#                 "lon": c['longitude'],
#                 "population": c.get('population'),
#                 "humidity": w.get('humidity'),
#                 "wind_speed": w.get('wind_speed'),
#                 "wind_direction": w.get('wind_direction'),
#                 "heat_index": w.get('heatIndex'),
#                 "temperature": w.get('temperature'),
#                 "air_quality": aq
#             })
#     return enriched

# # 5. Combine All into Fire Record JSON
# def enrich_fire_record(fire_row, counties_data):
#     lat, lon = float(fire_row['latitude']), float(fire_row['longitude'])
#     weather = get_weather(lat, lon)
#     air = get_air_quality(lat, lon)

#     wind_speed = int(weather['wind_speed'].split()[0]) if weather.get('wind_speed') else 0
#     wind_direction = int(weather['wind_direction'].split()[0]) if weather.get('wind_direction') else 0

#     spread = estimate_spread(lat, lon, wind_speed, wind_direction)
#     enriched_nearby = get_nearby_counties(lat, lon, counties_data)

#     return {
#         "county": fire_row['county'],
#         "population": fire_row['population'],
#         "location": {"lat": lat, "lon": lon},
#         "heat_index": weather.get('heatIndex'),
#         "humidity": weather.get('humidity'),
#         "wind": {
#             "speed": weather.get('wind_speed'),
#             "direction": weather.get('wind_direction')
#         },
#         "air_quality": air,
#         "spread_prediction": spread,
#         "nearby_areas": enriched_nearby
#     }....can you fill in the api urls and also the county and population data is already in the csv...# utils.py
# import requests
# from geopy.distance import geodesic
# import math

# NWS_BASE = "https://api.weather.gov"
# IQAIR_BASE = "http://api.airvisual.com/v2"
# IQAIR_KEY = "YOUR_IQAIR_API_KEY"  # Replace with your actual key

# # 1. Get Weather Data
# def get_weather(lat, lon):
#     try:
#         point_url = f"{NWS_BASE}/points/{lat},{lon}"
#         point_res = requests.get(point_url).json()
#         forecast_url = point_res['properties']['forecastHourly']

#         forecast_res = requests.get(forecast_url).json()
#         first_hour = forecast_res['properties']['periods'][0]  # Current hour

#         return {
#             "temperature": first_hour['temperature'],
#             "wind_speed": first_hour['windSpeed'],
#             "wind_direction": first_hour['windDirection'],
#             "humidity": first_hour.get('relativeHumidity', {}).get('value'),
#             "heatIndex": first_hour.get('heatIndex', {}).get('value') or first_hour['temperature']
#         }
#     except Exception as e:
#         print(f"[Weather API Error] {e}")
#         return {}

# # 2. Get Air Quality Data from IQAir
# def get_air_quality(lat, lon):
#     try:
#         url = f"{IQAIR_BASE}/nearest_city?lat={lat}&lon={lon}&key={IQAIR_KEY}"
#         res = requests.get(url).json()
#         data = res.get("data", {})
#         return {
#             "pm2_5": data.get("current", {}).get("pollution", {}).get("pm2_5"),
#             "aqi": data.get("current", {}).get("pollution", {}).get("aqius"),
#             "category": data.get("current", {}).get("pollution", {}).get("mainus", "Unknown")
#         }
#     except Exception as e:
#         print(f"[Air API Error] {e}")
#         return {}

# # 3. Estimate Fire Spread Based on Wind Vector
# def estimate_spread(lat, lon, wind_speed_kph, wind_direction_deg, distance_km=5):
#     rad = math.radians(wind_direction_deg)
#     dx = distance_km * math.sin(rad)
#     dy = distance_km * math.cos(rad)
#     new_lat = lat + (dy / 111)  # Approximate
#     new_lon = lon + (dx / (111 * math.cos(math.radians(lat))))
#     return [{"lat": new_lat, "lon": new_lon}]

# # 4. Get Nearby Counties with Enriched Weather + AQI
# def get_nearby_counties(lat, lon, counties_data, radius_km=50):
#     enriched = []
#     for c in counties_data:
#         dist = geodesic((lat, lon), (c['latitude'], c['longitude'])).km
#         if dist <= radius_km:
#             w = get_weather(c['latitude'], c['longitude'])
#             aq = get_air_quality(c['latitude'], c['longitude'])
#             enriched.append({
#                 "county": c['county'],
#                 "lat": c['latitude'],
#                 "lon": c['longitude'],
#                 "population": c.get('population'),
#                 "humidity": w.get('humidity'),
#                 "wind_speed": w.get('wind_speed'),
#                 "wind_direction": w.get('wind_direction'),
#                 "heat_index": w.get('heatIndex'),
#                 "temperature": w.get('temperature'),
#                 "air_quality": aq
#             })
#     return enriched

# # 5. Combine All into Fire Record JSON
# def enrich_fire_record(fire_row, counties_data):
#     lat, lon = float(fire_row['latitude']), float(fire_row['longitude'])
#     weather = get_weather(lat, lon)
#     air = get_air_quality(lat, lon)

#     wind_speed = int(weather['wind_speed'].split()[0]) if weather.get('wind_speed') else 0
#     wind_direction = int(weather['wind_direction'].split()[0]) if weather.get('wind_direction') else 0

#     spread = estimate_spread(lat, lon, wind_speed, wind_direction)
#     enriched_nearby = get_nearby_counties(lat, lon, counties_data)

#     return {
#         "county": fire_row['county'],
#         "population": fire_row['population'],
#         "location": {"lat": lat, "lon": lon},
#         "heat_index": weather.get('heatIndex'),
#         "humidity": weather.get('humidity'),
#         "wind": {
#             "speed": weather.get('wind_speed'),
#             "direction": weather.get('wind_direction')
#         },
#         "air_quality": air,
#         "spread_prediction": spread,
#         "nearby_areas": enriched_nearby
#     }