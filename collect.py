# ══════════════════════════════════════
# collect.py
# Pulls raw snake sighting data from
# iNaturalist and GBIF for 12 African cities
# ══════════════════════════════════════

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

INATURALIST_API_KEY = os.getenv("INATURALIST_API_KEY")
RECORDS_PER_CITY    = 500

CITIES = [
    {"name": "Lagos",         "country": "Nigeria",       "lat": 6.5244,   "lng": 3.3792},
    {"name": "Nairobi",       "country": "Kenya",         "lat": -1.2921,  "lng": 36.8219},
    {"name": "Accra",         "country": "Ghana",         "lat": 5.6037,   "lng": -0.1870},
    {"name": "Kampala",       "country": "Uganda",        "lat": 0.3476,   "lng": 32.5825},
    {"name": "Dar es Salaam", "country": "Tanzania",      "lat": -6.7924,  "lng": 39.2083},
    {"name": "Johannesburg",  "country": "South Africa",  "lat": -26.2041, "lng": 28.0473},
    {"name": "Addis Ababa",   "country": "Ethiopia",      "lat": 9.0320,   "lng": 38.7469},
    {"name": "Kigali",        "country": "Rwanda",        "lat": -1.9441,  "lng": 30.0619},
    {"name": "Abuja",         "country": "Nigeria",       "lat": 9.0579,   "lng": 7.4951},
    {"name": "Ibadan",        "country": "Nigeria",       "lat": 7.3775,   "lng": 3.9470},
    {"name": "Kumasi",        "country": "Ghana",         "lat": 6.6885,   "lng": -1.6244},
    {"name": "Mombasa",       "country": "Kenya",         "lat": -4.0435,  "lng": 39.6682},
]

# Bounding box radius in degrees (~100km)
BBOX_RADIUS = 0.9

SERPENTES_TAXON_ID_INAT = 85553
SERPENTES_TAXON_KEY_GBIF = 11592253


# ── iNaturalist ──────────────────────

def fetch_inaturalist(city: dict) -> list:
    print(f"  [iNaturalist] Fetching {city['name']}...")
    results = []
    page    = 1
    headers = {"Authorization": f"Bearer {INATURALIST_API_KEY}"}

    while len(results) < RECORDS_PER_CITY:
        params = {
            "taxon_id":  SERPENTES_TAXON_ID_INAT,
            "nelat":     city["lat"] + BBOX_RADIUS,
            "nelng":     city["lng"] + BBOX_RADIUS,
            "swlat":     city["lat"] - BBOX_RADIUS,
            "swlng":     city["lng"] - BBOX_RADIUS,
            "per_page":  200,
            "page":      page,
            "order":     "desc",
            "order_by":  "created_at",
            "has[]":     "geo",
        }

        try:
            response = requests.get(
                "https://api.inaturalist.org/v1/observations",
                headers=headers,
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            obs  = data.get("results", [])
            if not obs:
                break

            for o in obs:
                coords = o.get("location", "")
                lat, lng = None, None
                if coords:
                    parts = coords.split(",")
                    if len(parts) == 2:
                        try:
                            lat = float(parts[0])
                            lng = float(parts[1])
                        except ValueError:
                            pass

                if lat is None or lng is None:
                    continue

                taxon       = o.get("taxon", {})
                species     = taxon.get("name", "Unknown")
                common_name = taxon.get("preferred_common_name", species)
                photos      = o.get("photos", [])
                photo_url   = photos[0].get("url", "").replace("square", "medium") if photos else None

                results.append({
                    "source":       "iNaturalist",
                    "city":         city["name"],
                    "country":      city["country"],
                    "species":      species,
                    "common_name":  common_name,
                    "latitude":     lat,
                    "longitude":    lng,
                    "date":         o.get("observed_on", ""),
                    "observer":     o.get("user", {}).get("login", ""),
                    "photo_url":    photo_url,
                    "quality":      o.get("quality_grade", ""),
                    "record_url":   f"https://www.inaturalist.org/observations/{o.get('id')}",
                })

            page += 1
            if len(obs) < 200:
                break
            time.sleep(0.5)

        except Exception as e:
            print(f"    iNaturalist error for {city['name']}: {e}")
            break

    print(f"    → {len(results)} records")
    return results[:RECORDS_PER_CITY]


# ── GBIF ─────────────────────────────

def fetch_gbif(city: dict) -> list:
    print(f"  [GBIF] Fetching {city['name']}...")
    results = []
    offset  = 0
    limit   = 300

    while len(results) < RECORDS_PER_CITY:
        params = {
            "taxonKey":           SERPENTES_TAXON_KEY_GBIF,
            "decimalLatitude":    f"{city['lat'] - BBOX_RADIUS},{city['lat'] + BBOX_RADIUS}",
            "decimalLongitude":   f"{city['lng'] - BBOX_RADIUS},{city['lng'] + BBOX_RADIUS}",
            "hasCoordinate":      True,
            "hasGeospatialIssue": False,
            "limit":              limit,
            "offset":             offset,
        }

        try:
            response = requests.get(
                "https://api.gbif.org/v1/occurrence/search",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            occs = data.get("results", [])
            if not occs:
                break

            for o in occs:
                lat = o.get("decimalLatitude")
                lng = o.get("decimalLongitude")
                if lat is None or lng is None:
                    continue

                species     = o.get("species") or o.get("scientificName", "Unknown")
                common_name = o.get("vernacularName", species)
                country     = o.get("country", city["country"])
                locality    = o.get("locality", "")
                state       = o.get("stateProvince", "")

                year  = o.get("year", "")
                month = o.get("month", "")
                day   = o.get("day", "")
                date_parts = [str(p) for p in [year, month, day] if p]
                date  = "-".join(date_parts) if date_parts else ""

                media     = o.get("media", [])
                photo_url = media[0].get("identifier") if media else None

                results.append({
                    "source":       "GBIF",
                    "city":         city["name"],
                    "country":      city["country"],
                    "species":      species,
                    "common_name":  common_name,
                    "latitude":     lat,
                    "longitude":    lng,
                    "date":         date,
                    "observer":     o.get("institutionCode", o.get("datasetName", "")),
                    "photo_url":    photo_url,
                    "quality":      o.get("basisOfRecord", ""),
                    "record_url":   f"https://www.gbif.org/occurrence/{o.get('key')}",
                })

            offset += limit
            if data.get("endOfRecords", True):
                break
            time.sleep(0.3)

        except Exception as e:
            print(f"    GBIF error for {city['name']}: {e}")
            break

    print(f"    → {len(results)} records")
    return results[:RECORDS_PER_CITY]


# ── Main ─────────────────────────────

def main():
    all_records = []

    for city in CITIES:
        print(f"\n📍 {city['name']}, {city['country']}")
        inat_records = fetch_inaturalist(city)
        gbif_records = fetch_gbif(city)
        all_records.extend(inat_records)
        all_records.extend(gbif_records)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["species", "latitude", "longitude", "date"])
    df = df.reset_index(drop=True)

    output_path = "data/raw/sightings_raw.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Done! {len(df)} total records saved to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()