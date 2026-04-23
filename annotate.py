# ══════════════════════════════════════
# annotate.py
# Reshapes computed dataset into rich
# prompt/completion format and sends
# to Adaption API for multilingual
# optimization across African languages
# ══════════════════════════════════════

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from adaption import Adaption, DatasetTimeout

load_dotenv()

ADAPTION_API_KEY = os.getenv("ADAPTION_API_KEY")

INPUT_PATH  = "data/processed/sightings_computed.csv"
OUTPUT_PATH = "data/processed/sightings_annotated.csv"
UPLOAD_PATH = "data/processed/sightings_for_adaption.csv"

VENOMOUS_TERMS = [
    'cobra', 'mamba', 'viper', 'adder', 'rattlesnake', 'krait',
    'boomslang', 'puff adder', 'king cobra', 'taipan', 'death adder',
    'bushmaster', 'fer-de-lance', 'water moccasin', 'copperhead',
    'naja', 'bitis', 'dendroaspis'
]

CITY_CONTEXT = {
    'Lagos':         'one of Africa\'s largest and most rapidly expanding megacities, with significant periurban deforestation and a documented high snakebite burden across Lagos State',
    'Nairobi':       'a major East African capital bordered by Nairobi National Park, where urban expansion is actively encroaching on critical wildlife corridors and savanna habitats',
    'Accra':         'the capital of Ghana and a fast-growing coastal city where urban sprawl is fragmenting remaining tropical forest patches along the Gulf of Guinea coastline',
    'Kampala':       'Uganda\'s capital situated across a series of hills and wetlands that are under intense urban development pressure from one of Africa\'s fastest-growing populations',
    'Dar es Salaam': 'Tanzania\'s largest city and one of the fastest-growing urban centres in Africa, with coastal and riverine habitats under severe pressure from informal settlement expansion',
    'Johannesburg':  'South Africa\'s economic hub where suburban expansion into the Highveld grassland ecosystem creates regular and well-documented human-wildlife conflict zones',
    'Addis Ababa':   'Ethiopia\'s capital situated at high altitude in the Ethiopian Highlands, where informal settlement growth and infrastructure development are destroying montane forest habitat',
    'Kigali':        'Rwanda\'s capital known for its rapid planned urbanization across a hilly landscape that retains significant but increasingly fragmented forest cover',
    'Abuja':         'Nigeria\'s federal capital territory experiencing rapid planned and informal urban growth that is actively encroaching on Guinea savanna woodland and riverine habitats',
    'Ibadan':        'one of Nigeria\'s largest cities by geographic area, where a dense and expanding urban-rural fringe creates significant and underreported snake displacement pressure',
    'Kumasi':        'Ghana\'s second largest city in the Ashanti region, surrounded by tropical moist forest that is under intense agricultural and urban development pressure',
    'Mombasa':       'Kenya\'s principal coastal city where mangrove ecosystem destruction and coastal forest clearance is linked to increasing human-reptile encounter frequency',
}

RISK_GUIDANCE = {
    'Low': (
        'This USDRI reading indicates that snake sightings in this location are consistent with the '
        'natural movement range of the species and that urban displacement pressure is currently within '
        'acceptable ecological bounds. Routine monitoring of occurrence patterns is recommended to detect '
        'any emerging changes. Residents should remain observant during outdoor activity but no emergency '
        'precautions are required at this risk level. Sightings are more likely to reflect opportunistic '
        'exploration rather than systematic habitat-driven displacement from destroyed natural areas.'
    ),
    'Moderate': (
        'This USDRI reading indicates emerging and measurable displacement pressure in this location. '
        'Habitat loss and urban expansion are beginning to push snake populations toward residential and '
        'agricultural areas, though the situation has not yet reached crisis levels requiring emergency '
        'response. Increased surveillance of urban-forest boundary zones is recommended. Residents should '
        'adopt preventive behaviours including wearing protective closed footwear outdoors, keeping yards '
        'and compounds clear of debris, woodpiles, and dense low vegetation that provide snake shelter, '
        'and familiarising themselves with the location and operating hours of the nearest hospital with '
        'antivenom treatment capacity.'
    ),
    'High': (
        'This USDRI reading indicates significant and active displacement pressure at this location. '
        'Substantial habitat loss combined with rapid urban expansion is systematically forcing snake '
        'populations out of their natural habitats and into human-occupied residential and periurban '
        'spaces. Targeted conservation and public health intervention is warranted, including structured '
        'community education programmes on snake encounter safety, systematic mapping of high-risk '
        'urban-forest boundary zones for local authority attention, and verification that antivenom '
        'stocks are available and accessible at local health facilities. Residents should exercise '
        'heightened caution in all green spaces and avoid walking barefoot or in open footwear near '
        'vegetation, particularly at dawn and dusk when snake activity peaks.'
    ),
    'Critical': (
        'This USDRI reading indicates an emergency-level displacement situation at this location. '
        'Snake sightings are concentrated in densely populated urban areas and the underlying habitat '
        'loss and urban expansion pressure driving this displacement is severe and accelerating. '
        'Immediate coordinated public health and conservation response is warranted, including activation '
        'of community alert networks, rapid antivenom distribution to local clinics and health posts, '
        'and formal engagement of national conservation and wildlife authorities to assess the full '
        'scale of displacement. Residents should restrict unnecessary outdoor activity in green spaces, '
        'ensure children and livestock are supervised near vegetated areas, and seek immediate hospital '
        'treatment for any suspected snakebite without attempting traditional or home remedies.'
    ),
}


def build_prompt(row) -> str:
    is_venomous  = any(t in (row.get('species', '') + row.get('common_name', '')).lower() for t in VENOMOUS_TERMS)
    venom_str    = 'venomous' if is_venomous else 'non-venomous'
    common       = row.get('common_name', row.get('species', 'Unknown'))
    species      = row.get('species', 'Unknown')
    city         = row['city']
    country      = row['country']
    lat          = float(row['latitude'])
    lng          = float(row['longitude'])
    date         = row.get('date', 'an unrecorded date')
    source       = row.get('source', 'an unknown source')
    city_context = CITY_CONTEXT.get(city, 'a rapidly urbanizing African city')

    return (
        f'A {venom_str} snake species — the {common} ({species}) — was recorded in {city}, {country}, '
        f'{city_context}. The observation was documented at geographic coordinates {lat:.4f} degrees '
        f'latitude and {lng:.4f} degrees longitude on {date}, sourced from {source}. '
        f'Using the Urban Snake Displacement Risk Index (USDRI) framework, which combines sighting '
        f'density, habitat loss, urban expansion pressure, and urban proximity into a composite score '
        f'from 0 to 100, provide a comprehensive assessment of the displacement risk at this specific '
        f'location in {city}. Your response should explain the score, interpret each of the four '
        f'contributing components, describe the public health implications for residents and practitioners '
        f'in {city}, and outline the recommended conservation and emergency response actions appropriate '
        f'to the assessed risk level.'
    )


def build_completion(row) -> str:
    common       = row.get('common_name', row.get('species', 'Unknown'))
    species      = row.get('species', 'Unknown')
    city         = row['city']
    country      = row['country']
    usdri        = row.get('usdri_score', 0)
    label        = row.get('usdri_label', 'Unknown')
    density      = row.get('density_score', 0)
    habitat_s    = row.get('habitat_score', 0)
    habitat_pct  = float(row.get('habitat_loss_pct', 0))
    urban        = row.get('urban_expansion_score', 0)
    proximity    = float(row.get('proximity_score', 0))
    deaths       = row.get('country_annual_snakebite_deaths')
    hospital_km  = row.get('nearest_hospital_km')
    record_url   = row.get('record_url', '')
    source       = row.get('source', 'Unknown')
    guidance     = RISK_GUIDANCE.get(label, RISK_GUIDANCE['Low'])
    is_venomous  = any(t in (species + common).lower() for t in VENOMOUS_TERMS)

    venom_note = (
        f'As a venomous species, the {common} poses a direct envenomation risk to any human who '
        f'encounters it in an urban context. Public health responders in {city} should treat reports '
        f'of this species in residential areas with particular urgency and ensure that antivenom '
        f'appropriate for this species is stocked at nearby medical facilities.'
    ) if is_venomous else (
        f'Although the {common} is a non-venomous species and does not pose a direct envenomation '
        f'risk, its presence in urban {city} is a meaningful ecological indicator. Non-venomous species '
        f'respond to the same displacement drivers as venomous ones, and an increase in non-venomous '
        f'urban sightings frequently precedes or accompanies increases in venomous species encounters '
        f'as the broader snake community is displaced from deteriorating natural habitat.'
    )

    hospital_note = (
        f'The nearest identified hospital facility is approximately {hospital_km:.1f} km from this '
        f'sighting location. '
        f'{"This represents relatively accessible emergency care for snakebite treatment." if pd.notna(hospital_km) and hospital_km and float(hospital_km) < 20 else "This distance may represent a significant barrier to timely antivenom administration in the event of envenomation, which is a critical factor given that treatment delay is the leading cause of snakebite mortality in sub-Saharan Africa."}'
    ) if pd.notna(hospital_km) and hospital_km else (
        f'Hospital proximity data was not available for this specific sighting location in {city}. '
        f'Residents should proactively identify and record the location of the nearest antivenom '
        f'treatment facility as a precautionary measure.'
    )

    deaths_note = (
        f'{country} reports approximately {int(deaths):,} snakebite-related deaths annually according '
        f'to WHO estimates, placing it among the highest-burden countries for snakebite envenomation '
        f'in sub-Saharan Africa. This national mortality context underscores the public health '
        f'significance of displacement risk monitoring in {city} and other major urban centres.'
    ) if pd.notna(deaths) and deaths else (
        f'Country-level snakebite mortality data was not available for {country} in this dataset. '
        f'Absence of data should not be interpreted as absence of burden — snakebite is systematically '
        f'underreported across sub-Saharan Africa.'
    )

    return (
        f'Urban Snake Displacement Risk Assessment — {city}, {country}\n\n'
        f'USDRI Score: {usdri}/100 ({label} Risk)\n\n'
        f'The sighting of the {common} ({species}) at this location in {city} has been assessed using '
        f'the Urban Snake Displacement Risk Index (USDRI), a reproducible composite framework that '
        f'quantifies the pressure forcing snake populations out of natural habitats and into '
        f'human-occupied urban spaces. The total USDRI score of {usdri} out of 100 places this '
        f'location in the {label} risk band.\n\n'
        f'Component Analysis:\n'
        f'Sighting Density ({density}/25): This component reflects the total concentration of '
        f'verified snake occurrence records across the {city} bounding area relative to its geographic '
        f'extent in square kilometres. Higher density scores indicate that snake-human encounters are '
        f'occurring at a rate inconsistent with low-pressure natural movement.\n\n'
        f'Habitat Loss ({habitat_s}/25): Satellite-derived estimates indicate that approximately '
        f'{habitat_pct:.1f}% of natural forest and vegetation cover has been lost in the broader '
        f'{city} region. This component directly captures the destruction of the habitat that snake '
        f'populations depend on for thermoregulation, prey access, and refuge.\n\n'
        f'Urban Expansion ({urban}/25): This component reflects the rate and scale at which urban '
        f'infrastructure is expanding into natural areas surrounding {city}. Rapid urban growth at '
        f'the city periphery is the primary driver of displacement pressure, systematically '
        f'eliminating the buffer zones between human settlement and remaining natural habitat.\n\n'
        f'Urban Proximity ({proximity}/25): Calculated from the Haversine distance between this '
        f'specific sighting location and the {city} urban centre, this component indicates how deeply '
        f'displaced snake populations have penetrated into human-occupied areas. A high proximity '
        f'score means snakes are being recorded close to the city core rather than at the periphery.\n\n'
        f'Risk Interpretation and Recommended Response:\n'
        f'{guidance}\n\n'
        f'Species Assessment:\n'
        f'{venom_note}\n\n'
        f'Public Health Context:\n'
        f'{deaths_note}\n\n'
        f'Healthcare Access:\n'
        f'{hospital_note}\n\n'
        f'Data Provenance:\n'
        f'This occurrence record was sourced from {source}. The full observation record including '
        f'photographs, taxonomic verification, and observer details is available at: {record_url}'
    )


def main():
    print("Loading computed dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df)} records loaded")

    print("\nBuilding rich prompt/completion pairs...")
    df["prompt"]     = df.apply(build_prompt, axis=1)
    df["completion"] = df.apply(build_completion, axis=1)

    adaption_df = df[[
        "prompt", "completion", "city", "country", "species",
        "common_name", "latitude", "longitude", "date", "source",
        "usdri_score", "usdri_label", "density_score", "habitat_score",
        "habitat_loss_pct", "urban_expansion_score", "proximity_score",
        "nearest_hospital_km", "country_annual_snakebite_deaths", "record_url"
    ]]

    adaption_df.to_csv(UPLOAD_PATH, index=False)
    print(f"  Saved to {UPLOAD_PATH}")

    print("\nSample prompt:")
    print(f"  {df['prompt'].iloc[0]}\n")
    print("Sample completion:")
    print(f"  {df['completion'].iloc[0][:500]}...\n")

    # ── Send to Adaption API ──
    print("Connecting to Adaption API...")
    client = Adaption(api_key=ADAPTION_API_KEY)

    print("Uploading dataset to Adaption...")
    result = client.datasets.upload_file(
        UPLOAD_PATH,
        name="african-urban-snake-displacement-dataset"
    )
    dataset_id = result.dataset_id
    print(f"  Dataset ID: {dataset_id}")

    print("Waiting for file processing...")
    while True:
        status = client.datasets.get_status(dataset_id)
        if status.row_count is not None:
            print(f"  Processed — {status.row_count} rows detected")
            break
        time.sleep(3)

    print("\nEstimating cost...")
    estimate = client.datasets.run(
        dataset_id,
        column_mapping={"prompt": "prompt", "completion": "completion"},
        estimate=True,
    )
    print(f"  Estimated time: ~{estimate.estimated_minutes} min")
    print(f"  Estimated credits: {estimate.estimated_credits_consumed}")

    confirm = input("\nProceed with adaptation run? (yes/no): ").strip().lower()
    if confirm != "yes":
        print(f"Aborted. Dataset ID: {dataset_id}")
        return

    print("\nStarting adaptation run...")
    run = client.datasets.run(
        dataset_id,
        column_mapping={"prompt": "prompt", "completion": "completion"},
    )
    print(f"  Run ID: {run.run_id}")

    print("\nWaiting for adaptation to complete...")
    try:
        final = client.datasets.wait_for_completion(dataset_id, timeout=3600)
        print(f"  Status: {final.status}")
    except DatasetTimeout:
        print("  Timed out — check your Adaption dashboard manually")
        return

    try:
        evaluation = client.datasets.get_evaluation(dataset_id)
        if evaluation and evaluation.evaluation_summary:
            print(f"\nQuality: {evaluation.evaluation_summary.grade_before} → {evaluation.evaluation_summary.grade_after}")
    except Exception:
        pass

    print("\nDownloading adapted dataset...")
    url = client.datasets.download(dataset_id)
    adapted = requests.get(url)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(adapted.content)
    print(f"  Saved to {OUTPUT_PATH}")
    print(f"\n✅ Done! Final dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()