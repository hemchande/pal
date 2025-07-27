import requests

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

# Example usage:
lat, lon = 29.7604, -95.3698  # Houston, TX
result = getCounty(lat, lon)
print(result)

