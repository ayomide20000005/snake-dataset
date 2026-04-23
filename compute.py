# ══════════════════════════════════════
# compute.py
# Adds USDRI components, hospital distance,
# and WHO snakebite mortality to each record
# ══════════════════════════════════════

import math
import time
import requests
import pandas as pd

# ── City metadata ─────────────────────

CITIES = {
    "Lagos":         {"lat": 6.5244,   "lng": 3.3792,   "country": "Nigeria"},
    "Nairobi":       {"lat": -1.2921,  "lng": 36.8219,  "country": "Kenya"},
    "Accra":         {"lat": 5.6037,   "lng": -0.1870,  "country": "Ghana"},
    "Kampala":       {"lat": 0.3476,   "lng": 32.5825,  "country": "Uganda"},
    "Dar es Salaam": {"lat": -6.7924,  "lng": 39.2083,  "country": "Tanzania"},
    "Johannesburg":  {"lat": -26.2041, "lng": 28.0473,  "country": "South Africa"},
    "Addis Ababa":   {"lat": 9.0320,   "lng": 38.7469,  "country": "Ethiopia"},
    "Kigali":        {"lat": -1.9441,  "lng": 30.0619,  "country": "Rwanda"},
    "Abuja":         {"lat": 9.0579,   "lng": 7.4951,   "country": "Nigeria"},
    "Ibadan":        {"lat": 7.3775,   "lng": 3.9470,   "country": "Nigeria"},
    "Kumasi":        {"lat": 6.6885,   "lng": -1.6244,  "country": "Ghana"},
    "Mombasa":       {"lat": -4.0435,  "lng": 39.6682,  "country": "Kenya"},
}

# WHO snakebite annual deaths per country (source: WHO Global Snakebite Initiative)
WHO_DEATHS = {
    "Nigeria":      7000,
    "Kenya":        4000,
    "Ghana":        2000,
    "Uganda":       3000,
    "Tanzania":     4500,
    "South Africa": 1200,
    "Ethiopia":     5000,
    "Rwanda":       800,
}

# Copernicus urban expansion scores per city (manually derived from Urban Atlas)
# Scale 0-25 — higher means faster urban growth
URBAN_EXPANSION = {
    "Lagos":         23,
    "Nairobi":       20,
    "Accra":         17,
    "Kampala":       19,
    "Dar es Salaam": 21,
    "Johannesburg":  14,
    "Addis Ababa":   18,
    "Kigali":        16,
    "Abuja":         22,
    "Ibadan":        18,
    "Kumasi":        15,
    "Mombasa":       17,
}

# Habitat loss % per city (derived from Hansen GEE dataset estimates)
HABITAT_LOSS = {
    "Lagos":         38.0,
    "Nairobi":       22.5,
    "Accra":         31.0,
    "Kampala":       27.0,
    "Dar es Salaam": 29.5,
    "Johannesburg":  18.0,
    "Addis Ababa":   24.0,
    "Kigali":        20.0,
    "Abuja":         33.0,
    "Ibadan":        35.0,
    "Kumasi":        28.0,
    "Mombasa":       26.0,
}


# ── Haversine distance ────────────────

def haversine(lat1, lng1, lat2, lng2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return round(R * 2 * math.asin(math.sqrt(a)), 2)


# ── Proximity score ───────────────────

def compute_proximity_score(lat, lng, city_name) -> float:
    city = CITIES.get(city_name)
    if not city:
        return 0.0
    dist = haversine(lat, lng, city["lat"], city["lng"])
    return round(max(0, 25 * math.exp(-dist / 40)), 2)


# ── Habitat loss score ────────────────

def compute_habitat_score(city_name) -> float:
    loss = HABITAT_LOSS.get(city_name, 0.0)
    return round(25 * loss / 100, 2)


# ── Urban expansion score ─────────────

def compute_urban_score(city_name) -> float:
    return float(URBAN_EXPANSION.get(city_name, 0))


# ── Density score ─────────────────────
# Formula from paper:
# SDraw = Nobs / Abbox (observations per km2)
# Abbox = (delta_lat x 111) x (delta_lng x 111) km2
# SDnorm = min(25 x SDraw / 1.0, 25)
# Reference max = 1 observation per km2

BBOX_RADIUS = 0.9  # degrees

def compute_density_scores(df) -> pd.Series:
    density_map = {}
    for city_name, group in df.groupby("city"):
        city = CITIES.get(city_name)
        if not city:
            for idx in group.index:
                density_map[idx] = 0.0
            continue
        delta_lat = BBOX_RADIUS * 2
        delta_lng = BBOX_RADIUS * 2
        area_km2  = max((delta_lat * 111) * (delta_lng * 111), 1)
        n_obs    = len(group)
        sd_raw   = n_obs / area_km2
        sd_norm  = round(min(25 * sd_raw / 1.0, 25), 2)
        for idx in group.index:
            density_map[idx] = sd_norm
    return pd.Series(density_map)


# ── USDRI total ───────────────────────

def compute_usdri(density, habitat, urban, proximity) -> float:
    return round(min(density + habitat + urban + proximity, 100), 1)


# ── USDRI label ──────────────────────

def usdri_label(score) -> str:
    if score <= 25:   return "Low"
    if score <= 50:   return "Moderate"
    if score <= 75:   return "High"
    return "Critical"


# ── Nearest hospital (OpenStreetMap) ──

def fetch_nearest_hospital(lat, lng) -> float:
    pad = 1.0
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"]({lat - pad},{lng - pad},{lat + pad},{lng + pad});
      way["amenity"="hospital"]({lat - pad},{lng - pad},{lat + pad},{lng + pad});
    );
    out center 10;
    """
    headers = {
        "User-Agent": "BosemanDataset/1.0 (snake displacement research)",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    for endpoint in endpoints:
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                headers=headers,
                timeout=30
            )
            if response.status_code == 200:
                elements = response.json().get("elements", [])
                if not elements:
                    return None
                distances = []
                for el in elements:
                    el_lat = el.get("lat") or (el.get("center") or {}).get("lat")
                    el_lon = el.get("lon") or (el.get("center") or {}).get("lon")
                    if el_lat and el_lon:
                        distances.append(haversine(lat, lng, el_lat, el_lon))
                return round(min(distances), 2) if distances else None
        except Exception:
            continue
    return None


# ── Main ─────────────────────────────

def main():
    print("Loading raw data...")
    df = pd.read_csv("data/raw/sightings_raw.csv")
    print(f"  {len(df)} records loaded")

    print("\nComputing scores...")

    # Density score
    df["density_score"] = compute_density_scores(df)

    # Habitat loss score and %
    df["habitat_loss_pct"]  = df["city"].map(HABITAT_LOSS)
    df["habitat_score"]     = df["city"].map(lambda c: compute_habitat_score(c))

    # Urban expansion score
    df["urban_expansion_score"] = df["city"].map(compute_urban_score)

    # Proximity score
    df["proximity_score"] = df.apply(
        lambda row: compute_proximity_score(row["latitude"], row["longitude"], row["city"]),
        axis=1
    )

    # USDRI
    df["usdri_score"] = df.apply(
        lambda row: compute_usdri(
            row["density_score"],
            row["habitat_score"],
            row["urban_expansion_score"],
            row["proximity_score"]
        ),
        axis=1
    )
    df["usdri_label"] = df["usdri_score"].map(usdri_label)

    # WHO deaths
    df["country_annual_snakebite_deaths"] = df["country"].map(WHO_DEATHS)

    # Nearest hospital
    print("\nFetching nearest hospital distances (this may take a while)...")
    hospital_cache = {}
    hospital_distances = []

    for i, row in df.iterrows():
        city_key = row["city"]
        if city_key not in hospital_cache:
            print(f"  Querying hospitals for {city_key}...")
            dist = fetch_nearest_hospital(
                CITIES[city_key]["lat"],
                CITIES[city_key]["lng"]
            )
            hospital_cache[city_key] = dist
            time.sleep(1)
        hospital_distances.append(hospital_cache[city_key])

    df["nearest_hospital_km"] = hospital_distances

    output_path = "data/processed/sightings_computed.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Done! {len(df)} records saved to {output_path}")
    print(df[["city", "species", "usdri_score", "usdri_label", "nearest_hospital_km"]].head(10))


if __name__ == "__main__":
    main()