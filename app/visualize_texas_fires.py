import pandas as pd
import folium
from folium.plugins import HeatMap
import os

# Load the data
csv_path = 'texas_fires_with_population.csv'
df = pd.read_csv(csv_path)

# Center the map on Texas
m = folium.Map(location=[31.0, -99.0], zoom_start=6, tiles='cartodbpositron')

# --- Add county boundaries if available ---
geojson_path = 'texas_counties.geojson'
if os.path.exists(geojson_path):
    folium.GeoJson(geojson_path, name='Texas Counties').add_to(m)

# --- Color by confidence ---
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

# --- Add fire points ---
fires_fg = folium.FeatureGroup(name='Fires by Confidence')
for _, row in df.iterrows():
    popup_text = (
        f"County: {row.get('county', '')}<br>"
        f"Population: {row.get('population', '')}<br>"
        f"FRP: {row.get('frp', '')}<br>"
        f"Confidence: {row.get('confidence', '')}<br>"
        f"Date: {row.get('acq_date', '')}<br>"
        f"Time: {row.get('acq_time', '')}"
    )
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=confidence_color(row['confidence']),
        fill=True,
        fill_color=confidence_color(row['confidence']),
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(fires_fg)
fires_fg.add_to(m)

# --- Add heatmap layer for fire density ---
heat_data = df[['latitude', 'longitude']].dropna().values.tolist()
if heat_data:
    HeatMap(heat_data, name='Fire Density Heatmap', radius=15, blur=10, min_opacity=0.3).add_to(m)

# --- Add legend ---
legend_html = '''
 <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 180px; height: 120px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: 0.85; padding: 10px;">
 <b>Fire Confidence Legend</b><br>
 <i class="fa fa-circle" style="color:red"></i> High<br>
 <i class="fa fa-circle" style="color:orange"></i> Nominal<br>
 <i class="fa fa-circle" style="color:green"></i> Low<br>
 <i class="fa fa-circle" style="color:blue"></i> Unknown<br>
 </div>
 '''
m.get_root().html.add_child(folium.Element(legend_html))

# --- Add layer control ---
folium.LayerControl().add_to(m)

# Save to HTML
m.save('texas_fires_map.html')
print("Map saved as texas_fires_map.html") 