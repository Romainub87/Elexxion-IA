from google.cloud import bigquery
import numpy as np
from utils import getYears

def getSecuriteByYearAndType():
    client = bigquery.Client(project="epsi-454815")
    years = getYears()

    query = """
            SELECT *, type_infraction
            FROM `epsi-454815.election.fait_securite`
            """

    query_job = client.query(query)
    results = list(query_job.result())
    data = [dict(row) for row in results]
    # Fusionne les données avec les années sur id_temps et groupe par type d'infraction
    merged_data = {}
    for d in data:
        for y in years:
            if d['id_temps'] == y['id_temps']:
                key = d['type_infraction']
                item = {**d, **y}
                if key not in merged_data:
                    merged_data[key] = []
                merged_data[key].append(item)
    # Trie chaque groupe par année
    for key in merged_data:
        merged_data[key] = sorted(merged_data[key], key=lambda item: item['annee'])
    return merged_data

