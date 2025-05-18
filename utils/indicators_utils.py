from google.cloud import bigquery
import numpy as np

def getYears():

    client = bigquery.Client(project="epsi-454815")

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
    client = bigquery.Client(project="epsi-454815")
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

def getResultatElection():
    query = """
            SELECT *
            FROM `epsi-454815.election.fait_resultat_election`
            """
    return fetchDataByYear(query)

def getCandidats():
    query = """
            SELECT *
            FROM `epsi-454815.election.dimension_candidat`
            """
    client = bigquery.Client(project="epsi-454815")
    query_job = client.query(query)
    results = list(query_job.result())
    data = [dict(row) for row in results]
    return sorted(data, key=lambda item: item['id_candidat'])

def getTypeElection():
    query = """
            SELECT *
            FROM `epsi-454815.election.dimension_scrutin`
            """
    client = bigquery.Client(project="epsi-454815")
    query_job = client.query(query)
    results = list(query_job.result())
    data = [dict(row) for row in results]
    return sorted(data, key=lambda item: item['id_scrutin'])

def getTourElection():
    query = """
            SELECT *
            FROM `epsi-454815.election.dimension_tour`
            """
    client = bigquery.Client(project="epsi-454815")
    query_job = client.query(query)
    results = list(query_job.result())
    data = [dict(row) for row in results]
    return sorted(data, key=lambda item: item['id_tour'])

def getPopulationByYear():
    query = """
            SELECT *
            FROM `epsi-454815.election.fait_population`
            """
    return fetchDataByYear(query)

def getAllDataPerYear():
    data_cac40 = getCAC40perYear()
    data_chomage = getChomagePerYear()
    data_pollution = getPollutionByYear()
    data_election = getResultatElection()
    data_candidats = getCandidats()
    data_scrutins = getTypeElection()
    data_tours = getTourElection()
    data_population = getPopulationByYear()

    pop_moy = np.mean(
        [item['nombre_habitant'] for item in data_population if item.get('nombre_habitant') is not None])

    type_election_id = next(
        (item['id_scrutin'] for item in data_scrutins if item['type_scrutin'] == 'Présidentielle'),
        None
    )

    tour_2 = next(
        (item['id_tour'] for item in data_tours if item['numero_tour'] == 2),
        None
    )

    particip_moy = np.mean([
        round(
            (sum(e2['nombre_voix'] for e2 in data_election if
                 e2.get('id_tour') == tour_2 and e2.get('annee') == item['annee']) /
             sum(e2['nombre_inscrit'] for e2 in data_election if
                 e2.get('id_tour') == tour_2 and e2.get('annee') == item['annee'])) * 100, 2
        )
        for item in data_election
        if item.get('id_tour') == tour_2 and
           sum(e2['nombre_inscrit'] for e2 in data_election if
               e2.get('id_tour') == tour_2 and e2.get('annee') == item['annee']) > 0
    ])

    merged_data = [
        {
            'annee': item_cac40['annee'],
            'point_bourse': item_cac40['point_bourse'],
            'population': next(
                (item_population['nombre_habitant'] for item_population in data_population if
                 item_population['annee'] == item_cac40['annee']),
                pop_moy
            ),
            'taux_chomage': next(
                (item_chomage['taux_chomage'] for item_chomage in data_chomage if
                 item_chomage['annee'] == item_cac40['annee']),
                None
            ),
            'nombre_jour_pic_particules_fines': next(
                (item_pollution['nombre_jour_pic_particules_fines'] for item_pollution in data_pollution if
                 item_pollution['annee'] == item_cac40['annee']),
                None
            ),
            'gagnant': {
                'id_candidat': gagnant_id,
                'nom_candidat': next(
                    (candidat['nom'] for candidat in data_candidats if candidat['id_candidat'] == gagnant_id),
                    None
                ),
                'nombre_voix': max_voix,
                'orientation_politique': next(
                    (candidat['orientation_politique'] for candidat in data_candidats if
                     candidat['id_candidat'] == gagnant_id),
                    None
                ),
                'participationPourcentage': next(
                    iter(
                        [
                            round(
                                (sum(e2['nombre_voix'] for e2 in data_election if
                                     e2['annee'] == item_cac40['annee'] and e2['id_scrutin'] == type_election_id and e2[
                                         'id_tour'] == tour_2) /
                                 sum(e2['nombre_inscrit'] for e2 in data_election if
                                     e2['annee'] == item_cac40['annee'] and e2['id_scrutin'] == type_election_id and e2[
                                         'id_tour'] == tour_2)) * 100, 2
                            )
                            if sum(e2['nombre_inscrit'] for e2 in data_election if
                                   e2['annee'] == item_cac40['annee'] and e2['id_scrutin'] == type_election_id and e2[
                                       'id_tour'] == tour_2) > 0
                            else particip_moy
                        ]
                    ),
                    particip_moy
                ),
            }
        }
        for item_cac40 in data_cac40
        for gagnant_id, max_voix in [
            max(
                (
                    (e['id_candidat'], sum(
                        e2['nombre_voix'] for e2 in data_election
                        if e2['annee'] == item_cac40['annee'] and e2['id_candidat'] == e['id_candidat'] and e2[
                            'id_scrutin'] == type_election_id and e2['id_tour'] == tour_2
                    ))
                    for e in data_election
                    if
                e['annee'] == item_cac40['annee'] and e['id_scrutin'] == type_election_id and e['id_tour'] == tour_2
                ),
                key=lambda x: x[1],
                default=(None, 0)
            )
        ]
    ]
    return merged_data