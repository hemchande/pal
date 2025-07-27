import pandas as pd
from openai import OpenAI
import requests
import os
import json

client = OpenAI(api_key="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")  # Assumes API key is set in environment

def load_and_format_tables():
    # County stats: include fire_count, frp_sum, frp_mean, population
    county_stats = pd.read_csv('fire_county_summary_2025-07-26.csv')[['county', 'fire_count', 'frp_sum', 'frp_mean', 'population']].head(5).to_markdown(index=False)
    # Cluster stats: use spread stats, but you could also aggregate FRP by cluster if available
    try:
        spread_df = pd.read_csv('texas_fire_cluster_spread_stats.csv')
        cluster_stats = spread_df[['cluster', 'num_fires']].sort_values('num_fires', ascending=False).head(5).to_markdown(index=False)
        spread_stats = spread_df.head(5).to_markdown(index=False)
    except Exception:
        cluster_stats = 'No cluster statistics available.'
        spread_stats = 'No spread statistics available.'
    return county_stats, cluster_stats, spread_stats

def build_situation_report_prompt(county_stats, cluster_stats, spread_stats, air_quality_info):
    return f'''
You are an emergency response analyst. Here is the latest Texas wildfire data summary:

Top counties by fire count and intensity (FRP):
{county_stats}

Top clusters by number of fires:
{cluster_stats}

Fire spread statistics (by cluster):
{spread_stats}

Air quality in affected areas:
{air_quality_info}

Write a concise situation report for the incident commander, highlighting the most urgent areas, the most intense fires (by FRP), air quality concerns, and any recommendations for resource allocation or evacuation.
'''

def build_resource_allocation_prompt(county_stats, cluster_stats, spread_stats):
    return f'''
Given the following Texas wildfire statistics and spread data:
{county_stats}

{cluster_stats}

{spread_stats}

Which counties or clusters should receive additional firefighting resources, and why? Provide a ranked list and a brief justification for each.
'''

def build_qa_prompt(county_stats, cluster_stats, spread_stats, question):
    return f'''
Based on the following Texas wildfire data:
{county_stats}

{cluster_stats}

{spread_stats}

Q: {question}
A:
'''

def call_llm(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    return response.choices[0].message.content

def get_nearby_places(lat, lon, place_type, api_key, radius=20000):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": radius,  # in meters
        "type": place_type,
        "key": api_key
    }
    response = requests.get(url, params=params)
    results = response.json().get("results", [])
    return [
        {
            "name": r["name"],
            "address": r.get("vicinity"),
            "location": r["geometry"]["location"],
            "rating": r.get("rating")
        }
        for r in results
    ]

def get_air_quality(lat, lon, api_key):
    url = "https://airquality.googleapis.com/v1/currentConditions:lookup"
    params = {"key": api_key}
    body = {"location": {"latitude": lat, "longitude": lon}}
    print(f"Calling Air Quality API for lat={lat}, lon={lon}")
    print(f"Request URL: {url}")
    print(f"Request params: {params}")
    print(f"Request body: {body}")
    response = requests.post(url, params=params, json=body)
    print(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Response JSON: {data}")
        try:
            # Use the actual structure of the response
            indexes = data.get('indexes', [])
            if indexes:
                aqi = indexes[0]['aqi']
                category = indexes[0]['category']
                dominant_pollutant = indexes[0].get('dominantPollutant', 'N/A')
            else:
                aqi = category = dominant_pollutant = None
            # Pollutants may not be present in this response
            pollutants = data.get('pollutants', [])
            print(f"Parsed AQI: {aqi}, Category: {category}, Dominant Pollutant: {dominant_pollutant}, Pollutants: {pollutants}")
            return {
                "aqi": aqi,
                "category": category,
                "dominant_pollutant": dominant_pollutant,
                "pollutants": pollutants
            }
        except Exception as e:
            print(f"Error parsing air quality data: {e}")
            print(f"Full response: {data}")
            return None
    else:
        print(f"Air Quality API response: {response.text}")
        return None

def get_route(origin_lat, origin_lon, dest_lat, dest_lon, api_key):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin_lat},{origin_lon}",
        "destination": f"{dest_lat},{dest_lon}",
        "key": api_key
    }
    print(f"Calling Directions API: {url}")
    print(f"Params: {params}")
    response = requests.get(url, params=params)
    print(f"Directions API status code: {response.status_code}")
    data = response.json()
    print(f"Directions API response: {json.dumps(data, indent=2)[:1000]}...")  # Print first 1000 chars
    if data.get('status') == 'OK':
        route = data['routes'][0]
        polyline = route['overview_polyline']['points']
        distance = route['legs'][0]['distance']['value'] / 1000  # km
        duration = route['legs'][0]['duration']['value'] / 60    # min
        steps = [step['html_instructions'] for step in route['legs'][0]['steps']]
        print(f"Parsed route: {distance} km, {duration} min, {len(steps)} steps")
        return {
            "polyline": polyline,
            "distance_km": distance,
            "duration_min": duration,
            "steps": steps
        }
    else:
        print(f"Directions API error: {data.get('status')}")
    return None

if __name__ == "__main__":
    county_stats, cluster_stats, spread_stats = load_and_format_tables()
    county_stats_df = pd.read_csv('texas_fire_county_statistics.csv').sort_values('fire_count', ascending=False)
    df = pd.read_csv('texas_fires_with_population.csv')
    GOOGLE_API_KEY = "AIzaSyCi_a_DkflqMRGcrFYvYKRB05XwNOFMmOg"

    recommendations_json = []
    air_quality_info = ""
    for i in range(3):
        county = county_stats_df.iloc[i]['county']
        county_fires = df[df['county'] == county]
        centroid_lat = county_fires['latitude'].mean()
        centroid_lon = county_fires['longitude'].mean()
        air_quality = get_air_quality(centroid_lat, centroid_lon, GOOGLE_API_KEY)
        aq_summary = {
            "name": county, # Changed to county name for consistency
            "address": "Fire affected area", # Placeholder
            "type": "county",
            "lat": centroid_lat,
            "lon": centroid_lon,
            "aqi": None,
            "category": None,
            "dominant_pollutant": None,
            "pollutants": [],
            "evacuation_locations": []
        }
        if air_quality:
            aq_summary["aqi"] = air_quality["aqi"]
            aq_summary["category"] = air_quality["category"]
            aq_summary["dominant_pollutant"] = air_quality.get("dominant_pollutant")
            aq_summary["pollutants"] = air_quality.get("pollutants", [])
        # --- Evacuation Recommendations ---
        places_lodging = get_nearby_places(centroid_lat, centroid_lon, 'lodging', GOOGLE_API_KEY, radius=30000)
        places_hospital = get_nearby_places(centroid_lat, centroid_lon, 'hospital', GOOGLE_API_KEY, radius=30000)
        for p in places_lodging[:3]:
            lat, lon = p['location']['lat'], p['location']['lng']
            aq = get_air_quality(lat, lon, GOOGLE_API_KEY)
            aqi = aq['aqi'] if aq else None
            category = aq['category'] if aq else None
            dominant_pollutant = aq.get('dominant_pollutant') if aq else None
            route_details = get_route(centroid_lat, centroid_lon, lat, lon, GOOGLE_API_KEY)
            aq_summary["evacuation_locations"].append({
                "name": p['name'],
                "address": p['address'],
                "type": "lodging",
                "lat": lat,
                "lon": lon,
                "aqi": aqi,
                "category": category,
                "dominant_pollutant": dominant_pollutant,
                "route": route_details
            })
        for p in places_hospital[:3]:
            lat, lon = p['location']['lat'], p['location']['lng']
            aq = get_air_quality(lat, lon, GOOGLE_API_KEY)
            aqi = aq['aqi'] if aq else None
            category = aq['category'] if aq else None
            dominant_pollutant = aq.get('dominant_pollutant') if aq else None
            route_details = get_route(centroid_lat, centroid_lon, lat, lon, GOOGLE_API_KEY)
            aq_summary["evacuation_locations"].append({
                "name": p['name'],
                "address": p['address'],
                "type": "hospital",
                "lat": lat,
                "lon": lon,
                "aqi": aqi,
                "category": category,
                "dominant_pollutant": dominant_pollutant,
                "route": route_details
            })
        recommendations_json.append(aq_summary)
    # Save recommendations as JSON
    with open('evacuation_recommendations.json', 'w') as f:
        json.dump(recommendations_json, f, indent=2)
    print("Evacuation and air quality recommendations (JSON):")
    print(json.dumps(recommendations_json, indent=2))
    # For LLM and future map UI integration, you can now use this JSON file.
    # Example: Situation report (still uses air_quality_info for now)
    air_quality_info = json.dumps(recommendations_json, indent=2)
    prompt = build_situation_report_prompt(county_stats, cluster_stats, spread_stats, air_quality_info)
    print("Prompt:\n", prompt)
    print("\nLLM Response:\n", call_llm(prompt))
    # Example: Resource allocation
    # prompt = build_resource_allocation_prompt(county_stats, cluster_stats, spread_stats)
    # print(call_llm(prompt))
    # Example: Q&A
    # question = "Which cluster is spreading the fastest?"
    # prompt = build_qa_prompt(county_stats, cluster_stats, spread_stats, question)
    # print(call_llm(prompt)) 