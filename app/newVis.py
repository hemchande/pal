import json
import pandas as pd
import folium
from folium.plugins import HeatMap
import os

# Load enriched fire data
with open("texas_fires_enriched_2025-07-26.json", "r") as f:
    enriched_fires = json.load(f)

# Initialize map
m = folium.Map(location=[31.0, -99.0], zoom_start=6, tiles='cartodbpositron')


# --- Define confidence → color map ---
def confidence_color(conf):
    conf = str(conf).lower()
    if conf == 'h':
        return 'red'
    elif conf == 'n':
        return 'orange'
    elif conf == 'l':
        return 'green'
    else:
        return 'blue'

# --- Add fire points (confidence + metadata) ---
fires_fg = folium.FeatureGroup(name='Fires by Confidence')

for fire in enriched_fires:
    lat = fire["location"]["lat"]
    lon = fire["location"]["lon"]
    if not lat or not lon:
        continue

    popup_text = (
        f"<b>County:</b> {fire.get('county', 'N/A')}<br>"
        f"<b>Population:</b> {fire.get('population', 'N/A')}<br>"
        f"<b>FRP:</b> {fire.get('frp', 'N/A')}<br>"
        f"<b>Humidity:</b> {fire.get('humidity', 'N/A')}<br>"
        f"<b>Air Quality:</b> {fire.get('air_quality', 'N/A')}<br>"
        f"<b>Heat Index:</b> {fire.get('heat_index', 'N/A')}<br>"

    )

    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=confidence_color(fire.get('confidence')),
        fill=True,
        fill_color=confidence_color(fire.get('confidence')),
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(fires_fg)

fires_fg.add_to(m)


# --- Wind HeatMap ---
wind_heat_data = []
for fire in enriched_fires:
    lat = fire["location"]["lat"]
    lon = fire["location"]["lon"]
    wind_speed = fire["wind"]["speed"]
    if lat is not None and lon is not None and isinstance(wind_speed, (int, float)):
        wind_heat_data.append([lat, lon, wind_speed])

if wind_heat_data:
    HeatMap(wind_heat_data, name='Wind Speed Heatmap', radius=20, blur=12, min_opacity=0.4, max_zoom=13).add_to(m)

# --- Add Wind Direction Arrows + Fire Metadata Popups ---
wind_fg = folium.FeatureGroup(name='Wind Vectors & Fire Info')
for fire in enriched_fires:
    lat = fire["location"]["lat"]
    lon = fire["location"]["lon"]
    wind_speed = fire["wind"]["speed"]
    wind_dir = fire["wind"]["direction"]
    if not (lat and lon and isinstance(wind_speed, (int, float)) and isinstance(wind_dir, (int, float))):
        continue

    # Construct detailed popup text
    popup_html = f"""
    <b>County:</b> {fire.get('county', 'N/A')}<br>
    <b>Date:</b> {fire.get('acq_date', 'N/A')}<br>
    <b>Time:</b> {fire.get('acq_time', 'N/A')}<br>
    <b>FRP:</b> {fire.get('frp', 'N/A')}<br>
    <b>Temperature:</b> {fire.get('temp', 'N/A')}°C<br>
    <b>Humidity:</b> {fire.get('humidity', 'N/A')}%<br>
    <b>Heat Index:</b> {fire.get('heat_index', 'N/A')}<br>
    <b>Wind:</b> {wind_speed} km/h @ {wind_dir}°<br>
    <b>AQI:</b> {fire.get('aqi', 'N/A')}<br>
    <b>Pollutant:</b> {fire.get('dominant_pollutant', 'N/A')}<br>
    """
    popup = folium.Popup(popup_html, max_width=300)

    def wind_color(speed):
        if speed >= 30:
            return "darkred"
        elif speed >= 15:
            return "orange"
        else:
            return "green"

    folium.RegularPolygonMarker(
        location=[lat, lon],
        number_of_sides=3,
        radius=8,
        rotation=wind_dir,
        color=wind_color(wind_speed),
        fill=True,
        fill_opacity=0.9,
        popup=popup
    ).add_to(wind_fg)
wind_fg.add_to(m)

# --- Plot fire_zone_geometry if exists ---
for fire in enriched_fires:
    if "fire_zone_geometry" in fire and fire["fire_zone_geometry"]:
        geojson = fire["fire_zone_geometry"]
        folium.GeoJson(geojson, name="Fire Zone").add_to(m)

# --- Plot nearby evacuation areas ---
evac_fg = folium.FeatureGroup(name="Evacuation Locations")
for fire in enriched_fires:
    if "evacuation_locations" in fire:
        for evac in fire["evacuation_locations"]:
            name = evac.get("name")
            lat = evac.get("lat")
            lon = evac.get("lon")
            typ = evac.get("type", "evac")
            if lat and lon:
                popup = folium.Popup(f"""
                <b>{name}</b><br>
                Type: {typ}<br>
                AQI: {evac.get("aqi", "N/A")}<br>
                Address: {evac.get("address", "N/A")}
                """, max_width=250)
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.Icon(color='blue' if typ == "hospital" else "green", icon="plus-sign"),
                    popup=popup
                ).add_to(evac_fg)
evac_fg.add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save the map
map_path = "clientapp/public/texas_fires_wind_geojson_evac_map.html"
m.save(map_path)
print(f"✅ Map saved as {map_path}")
