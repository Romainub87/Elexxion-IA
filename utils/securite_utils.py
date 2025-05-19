from google.cloud import bigquery
from utils.indicators_utils import getYears

def getSecuriteByYearAndType(cache_file='data/securite_cache.json'):
    import os
    import json

    # Vérifier si le cache existe déjà
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    client = bigquery.Client(project="epsi-454815")
    years = getYears()

    query = """
            SELECT *, type_infraction
            FROM `epsi-454815.election.fait_securite`
            """

    query_job = client.query(query)
    results = list(query_job.result())
    data = [dict(row) for row in results]
    merged_data = {}
    for d in data:
        for y in years:
            if d['id_temps'] == y['id_temps']:
                key = d['type_infraction']
                item = {**d, **y}
                if key not in merged_data:
                    merged_data[key] = []
                merged_data[key].append(item)
    for key in merged_data:
        merged_data[key] = sorted(merged_data[key], key=lambda item: item['annee'])

    # Sauvegarder dans le cache
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    return merged_data

