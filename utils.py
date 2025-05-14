from google.cloud import bigquery
import numpy as np

def getYears():

    client = bigquery.Client()

    query = """
            SELECT * \
            FROM `epsi-454815.election.dimension_temps` \
            """

    # Exécutez la requête
    query_job = client.query(query)
    results = query_job.result()

    # Transformez les résultats en liste de dictionnaires
    data = [dict(row) for row in results]

    # Retournez les résultats en JSON
    return data

def fetchDataByYear(query):
    client = bigquery.Client()
    years = getYears()
    query_job = client.query(query)
    results = list(query_job.result())
    data = [dict(row) for row in results]
    return sorted(
        [{**d, **y} for d in data for y in years if d['id_temps'] == y['id_temps']],
        key=lambda item: item['annee']
    )

def getPollutionByYear():
    query = """
            SELECT *
            FROM `epsi-454815.election.fait_qualite_air`
            """
    return fetchDataByYear(query)

def getChomagePerYear():
    query = """
            SELECT *
            FROM `epsi-454815.election.fait_chomage`
            """
    return fetchDataByYear(query)

def getCAC40perYear():
    query = """
            SELECT *
            FROM `epsi-454815.election.fait_bourse`
            """
    return fetchDataByYear(query)

def getAllDataPerYear():
    data_cac40 = getCAC40perYear()
    data_chomage = getChomagePerYear()
    data_pollution = getPollutionByYear()

    merged_data = [
        {
            'annee': item_cac40['annee'],
            'point_bourse': item_cac40['point_bourse'],
            'taux_chomage': next(
                (item_chomage['taux_chomage'] for item_chomage in data_chomage if item_chomage['annee'] == item_cac40['annee']),
                None
            ),
            'nombre_jour_pic_particules_fines': next(
                (item_pollution['nombre_jour_pic_particules_fines'] for item_pollution in data_pollution if item_pollution['annee'] == item_cac40['annee']),
                None
            )
        }
        for item_cac40 in data_cac40
    ]

    return merged_data