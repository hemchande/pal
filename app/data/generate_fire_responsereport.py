import pandas as pd
import json
import requests
from openai import OpenAI
import numpy as np


import math

client = OpenAI(api_key="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")  # Set your OpenAI key
GOOGLE_API_KEY = "AIzaSyCi_a_DkflqMRGcrFYvYKRB05XwNOFMmOg"  

def direction_to_degrees(direction):
    """Map wind cardinal direction to angle in degrees."""
    direction_map = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
        "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180,
        "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270,
        "WNW": 292.5, "NW": 315, "NNW": 337.5
    }
    return direction_map.get(direction, None)

def angle_between_points(lat1, lon1, lat2, lon2):
    """Calculate bearing from point A to point B."""
    d_lon = math.radians(lon2 - lon1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(d_lon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360



# Load CSVs
county_df = pd.read_csv("fire_county_summary_2025-07-26.csv")
spread_df = pd.read_csv("fire_cluster_spread_stats_2025-07-26.csv")
fires_df = pd.read_csv("texas_fires_2025-07-26_with_population.csv")


# fires_df = pd.read_csv("texas_fires_2025-07-26_with_population.csv")

# Load enriched fire data with weather information
with open("texas_fires_enriched_2025-07-26.json", "r") as f:
    enriched_fires = json.load(f)

# Prepare markdown summaries
county_md = county_df[['county', 'fire_count', 'fire_rate_per_100k']].sort_values('fire_count', ascending=False).head(5).to_markdown(index=False)
spread_md = spread_df[['cluster', 'avg_speed_kmh', 'total_distance_km', 'num_fires']].head(5).to_markdown(index=False)

# Define helper functions
def get_air_quality(lat, lon):
    url = "https://airquality.googleapis.com/v1/currentConditions:lookup"
    params = {"key": GOOGLE_API_KEY}
    body = {"location": {"latitude": lat, "longitude": lon}}
    r = requests.post(url, params=params, json=body)
    try:
        idx = r.json()["indexes"][0]
        return {"aqi": idx["aqi"], "category": idx["category"], "dominant_pollutant": idx.get("dominantPollutant")}
    except:
        return None

def get_nearby_places(lat, lon, place_type, radius=30000):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "type": place_type,
        "key": GOOGLE_API_KEY
    }
    r = requests.get(url, params=params)
    return r.json().get("results", [])

def get_route(origin_lat, origin_lon, dest_lat, dest_lon):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin_lat},{origin_lon}",
        "destination": f"{dest_lat},{dest_lon}",
        "key": GOOGLE_API_KEY
    }
    r = requests.get(url, params=params).json()
    if r.get("status") == "OK":
        leg = r["routes"][0]["legs"][0]
        return {
            "distance_km": leg["distance"]["value"] / 1000,
            "duration_min": leg["duration"]["value"] / 60,
            "steps": [s["html_instructions"] for s in leg["steps"]]
        }
    return None




# Extract weather factors summary
weather_summary = []
for fire in enriched_fires:
    summary = {
        "lat": fire["location"]["lat"],
        "lon": fire["location"]["lon"],
        "temp": fire.get("temp"),
        "heat_index": fire["heat_index"],

        "humidity": fire.get("humidity"),
        "wind_speed": fire["wind"]["speed"],
        "wind_direction": fire["wind"]["direction"],
        "county": fire.get("county"),
    }
    weather_summary.append(summary)

weather_df = pd.DataFrame(weather_summary)
weather_md = weather_df.groupby("county")[["temp", "humidity", "wind_speed"]].mean().round(2).head(5).to_markdown()

# Build top county summaries
evac_summaries = []
top_counties = county_df.sort_values('fire_count', ascending=False).head(3)['county']

for county in top_counties:
    sub_df = fires_df[fires_df["county"] == county]
    lat, lon = sub_df["latitude"].mean(), sub_df["longitude"].mean()
    aq = get_air_quality(lat, lon)
    evac_summary = {
        "county": county,
        "center_lat": lat,
        "center_lon": lon,
        "aqi": aq.get("aqi") if aq else None,
        "category": aq.get("category") if aq else None,
        "dominant_pollutant": aq.get("dominant_pollutant") if aq else None,
        "evacuation_locations": []
    }

    for place_type in ["lodging", "hospital"]:
        places = get_nearby_places(lat, lon, place_type)
        for p in places[:3]:
            loc = p["geometry"]["location"]
            p_aq = get_air_quality(loc["lat"], loc["lng"])
            route = get_route(lat, lon, loc["lat"], loc["lng"])
            evac_summary["evacuation_locations"].append({
                "name": p["name"],
                "type": place_type,
                "lat": loc["lat"],
                "lon": loc["lng"],
                "address": p.get("vicinity"),
                "aqi": p_aq.get("aqi") if p_aq else None,
                "category": p_aq.get("category") if p_aq else None,
                "route": route
            })

    evac_summaries.append(evac_summary)


# Filter unsafe evacuation points (downwind)
filtered_summaries = []

for summary in evac_summaries:
    county = summary["county"]
    center_lat, center_lon = summary["center_lat"], summary["center_lon"]
    
    wind_row = weather_df[weather_df["county"] == county]
    wind_info = next((fire for fire in enriched_fires if fire.get("county") == county), None)

    if not wind_info or not wind_info["wind"]["direction"]:
        filtered_summaries.append(summary)
        continue

    wind_deg = direction_to_degrees(wind_info["wind"]["direction"])
    if wind_deg is None:
        filtered_summaries.append(summary)
        continue

    safe_locations = []
    for loc in summary["evacuation_locations"]:
        evac_bearing = angle_between_points(center_lat, center_lon, loc["lat"], loc["lon"])
        diff = abs(evac_bearing - wind_deg)
        diff = min(diff, 360 - diff)  # handle circular angle wraparound

        # Only keep if not within 90 degrees of downwind
        if diff > 90:
            safe_locations.append(loc)

    summary["evacuation_locations"] = safe_locations
    filtered_summaries.append(summary)



# Save JSON
with open("evacuation_summary.json", "w") as f:
    json.dump(filtered_summaries, f, indent=2)

# Build LLM prompt
def build_prompt():
    evac_md = json.dumps(filtered_summaries, indent=2)
    return f"""
🔥 Texas Fire Report — 2025-07-26

Top Counties:
{county_md}

Cluster Spread:
{spread_md}

Weather Factors by County (Avg Temp, Humidity, Wind Speed)
{weather_md}

Evacuation Summary:
{evac_md}

You are an emergency response planner. Based on this data, generate:
- A professional emergency situation report
- Highlight top threat zones
- Comment on spread risks and prevailing wind directions
- Describe weather risks like heat and low humidity
- Air quality concern
- Suggest safe evacuation points and explain air quality considerations
- Recommend how to prioritize resource allocation

"""

# Call LLM
def call_llm(prompt):
    r = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3
    )
    return r.choices[0].message.content

if __name__ == "__main__":
    final_prompt = build_prompt()
    result = call_llm(final_prompt)
    print("\n🧠 LLM Response:\n")
    print(result)
    with open("situation_report_2025-07-26.txt", "w") as f:
        f.write(result)

# client = OpenAI(api_key="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")  # Set your OpenAI key
# GOOGLE_API_KEY = "AIzaSyCi_a_DkflqMRGcrFYvYKRB05XwNOFMmOg"           # Set your Google Maps key

# # Load existing summaries
# with open("texas_fires_enriched_2025-07-26.json") as f:
#     enriched_records = json.load(f)

# county_df = pd.read_csv("fire_county_summary_2025-07-26.csv")
# spread_df = pd.read_csv("fire_cluster_spread_stats_2025-07-26.csv")

# top_counties_md = county_df.sort_values("fire_count", ascending=False).head(5).to_markdown(index=False)
# spread_md = spread_df.sort_values("avg_speed_kmh", ascending=False).head(5).to_markdown(index=False)

# # -------------- Helper Functions --------------
# def get_air_quality(lat, lon):
#     url = "https://airquality.googleapis.com/v1/currentConditions:lookup"
#     params = {"key": GOOGLE_API_KEY}
#     body = {"location": {"latitude": lat, "longitude": lon}}
#     resp = requests.post(url, params=params, json=body)
#     if resp.status_code == 200:
#         data = resp.json()
#         idx = data.get("indexes", [{}])[0]
#         return {
#             "aqi": idx.get("aqi"),
#             "category": idx.get("category"),
#             "dominant_pollutant": idx.get("dominantPollutant")
#         }
#     return None

# def get_nearby_places(lat, lon, place_type, radius=30000):
#     url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
#     params = {
#         "location": f"{lat},{lon}",
#         "radius": radius,
#         "type": place_type,
#         "key": GOOGLE_API_KEY
#     }
#     r = requests.get(url, params=params)
#     return r.json().get("results", [])

# def get_route(origin_lat, origin_lon, dest_lat, dest_lon):
#     url = "https://maps.googleapis.com/maps/api/directions/json"
#     params = {
#         "origin": f"{origin_lat},{origin_lon}",
#         "destination": f"{dest_lat},{dest_lon}",
#         "key": GOOGLE_API_KEY
#     }
#     r = requests.get(url, params=params).json()
#     if r.get("status") == "OK":
#         leg = r["routes"][0]["legs"][0]
#         return {
#             "distance_km": leg["distance"]["value"] / 1000,
#             "duration_min": leg["duration"]["value"] / 60,
#             "steps": [s["html_instructions"] for s in leg["steps"]]
#         }
#     return None

# # -------------- Air Quality + Evacuation Summary --------------
# def build_evacuation_summary(top_n=3):
#     df = pd.DataFrame(enriched_records)
#     summary = []
#     for county in county_df.sort_values("fire_count", ascending=False).head(top_n)["county"]:
#         subset = df[df["county"] == county]
#         if subset.empty:
#             continue
#         lat, lon = subset["latitude"].mean(), subset["longitude"].mean()
#         aq = get_air_quality(lat, lon)
#         evac_summary = {
#             "county": county,
#             "center_lat": lat,
#             "center_lon": lon,
#             "aqi": aq.get("aqi") if aq else None,
#             "category": aq.get("category") if aq else None,
#             "dominant_pollutant": aq.get("dominant_pollutant") if aq else None,
#             "evacuation_locations": []
#         }

#         for place_type in ["lodging", "hospital"]:
#             places = get_nearby_places(lat, lon, place_type)
#             for p in places[:3]:
#                 loc = p["geometry"]["location"]
#                 p_aq = get_air_quality(loc["lat"], loc["lng"])
#                 route = get_route(lat, lon, loc["lat"], loc["lng"])
#                 evac_summary["evacuation_locations"].append({
#                     "name": p["name"],
#                     "type": place_type,
#                     "lat": loc["lat"],
#                     "lon": loc["lng"],
#                     "address": p.get("vicinity"),
#                     "aqi": p_aq.get("aqi") if p_aq else None,
#                     "category": p_aq.get("category") if p_aq else None,
#                     "route": route
#                 })

#         summary.append(evac_summary)
#     return summary

# # -------------- Prompt Templates --------------
# def build_situation_report_prompt(evac_json):
#     evac_md = json.dumps(evac_json, indent=2)
#     return f"""
# Texas Wildfire Emergency Situation – July 26, 2025

# 🔥 Top Counties by Fire Count:
# {top_counties_md}

# 💨 Fire Spread by Cluster:
# {spread_md}

# 📍 Evacuation Options and Air Quality:
# {evac_md}

# Please write a detailed situation report for the Emergency Ops Commander, summarizing:
# - Fire hotspots
# - Evacuation accessibility
# - Air quality zones
# - Resource allocation recommendations.
# """

# def call_llm(prompt):
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#         max_tokens=1200
#     )
#     return response.choices[0].message.content

# # -------------- Run Full Pipeline --------------
# if __name__ == "__main__":
#     print("🔄 Gathering evacuation and air quality data...")
#     evac_summary = build_evacuation_summary()
#     with open("evacuation_recommendations_augmented.json", "w") as f:
#         json.dump(evac_summary, f, indent=2)

#     print("✅ Data ready. Sending prompt to LLM...")
#     final_prompt = build_situation_report_prompt(evac_summary)
#     response = call_llm(final_prompt)

#     with open("situation_report_generated.txt", "w") as f:
#         f.write(response)

#     print("🧠 LLM Response:\n", response)
