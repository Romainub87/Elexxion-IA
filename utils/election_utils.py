import numpy as np
from google.cloud import bigquery
from utils.indicators_utils import getCAC40perYear, getPollutionByYear, getResultatElection, getChomagePerYear, getCandidats, \
    getTypeElection, getPopulationByYear, getTourElection
from utils.securite_utils import getSecuriteByYearAndType

from functools import lru_cache

@lru_cache(maxsize=1)
def getAllDataWhereElectionPerYear():
    data_cac40 = getCAC40perYear()
    data_chomage = getChomagePerYear()
    data_pollution = getPollutionByYear()
    data_election = getResultatElection()
    data_candidats = getCandidats()
    data_scrutins = getTypeElection()
    data_tours = getTourElection()
    data_population = getPopulationByYear()
    data_securite = getSecuriteByYearAndType()

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
    ]) if data_election else 0

    merged_data = []
    for item_cac40 in data_cac40:
        candidats_annee = [
            (
                e['id_candidat'],
                sum(
                    e2['nombre_voix'] for e2 in data_election
                    if e2['annee'] == item_cac40['annee']
                    and e2['id_candidat'] == e['id_candidat']
                    and e2['id_scrutin'] == type_election_id
                    and e2['id_tour'] == tour_2
                )
            )
            for e in data_election
            if e['annee'] == item_cac40['annee']
            and e['id_scrutin'] == type_election_id
            and e['id_tour'] == tour_2
        ]
        if not candidats_annee:
            continue

        gagnant_id, max_voix = max(candidats_annee, key=lambda x: x[1], default=(None, 0))
        if gagnant_id is None or max_voix == 0:
            continue

        participation = particip_moy
        total_inscrits = sum(
            e2['nombre_inscrit'] for e2 in data_election
            if e2['annee'] == item_cac40['annee']
            and e2['id_scrutin'] == type_election_id
            and e2['id_tour'] == tour_2
        )
        total_voix = sum(
            e2['nombre_voix'] for e2 in data_election
            if e2['annee'] == item_cac40['annee']
            and e2['id_scrutin'] == type_election_id
            and e2['id_tour'] == tour_2
        )
        if total_inscrits > 0:
            participation = round((total_voix / total_inscrits) * 100, 2)

        securite_annee = [
            item_securite
            for infractions in data_securite.values()
            for item_securite in infractions
            if item_securite['annee'] == item_cac40['annee']
        ]

        merged_data.append({
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
            'securite': securite_annee,
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
                'participationPourcentage': participation,
            }
        })

    return merged_data

def getCandidatsPresidentielle():
    client = bigquery.Client(project="epsi-454815")
    type_election = getTypeElection()
    type_election_id = next(
        (item['id_scrutin'] for item in type_election if item['type_scrutin'] == 'Présidentielle'),
        None
    )

    query = """
            SELECT DISTINCT id_candidat
            FROM `epsi-454815.election.fait_resultat_election`
            WHERE id_scrutin = @type_election_id
            """
    job_config = bigquery.QueryJobConfig()
    job_config.query_parameters = [
        bigquery.ScalarQueryParameter("type_election_id", "STRING", type_election_id)
    ]
    query_job = client.query(query, job_config=job_config)
    candidats_ids = [row['id_candidat'] for row in query_job.result()]

    query = """
            SELECT id_candidat, nom, orientation_politique
            FROM `epsi-454815.election.dimension_candidat`
            """
    query_job = client.query(query)
    data = [dict(row) for row in query_job.result()]

    candidats = [
        {
            'id_candidat': row['id_candidat'],
            'nom': row['nom'],
            'orientation_politique': row['orientation_politique']
        }
        for row in data if row['id_candidat'] in candidats_ids
    ]
    return candidats

def getCandidatsClassementEtLabelEncoder(merged_data):
    from collections import Counter
    from sklearn.preprocessing import LabelEncoder

    candidats_presidentielle = getCandidatsPresidentielle()
    votes_par_candidat = Counter()
    for item in merged_data:
        gagnant = item.get('gagnant', {})
        if gagnant and gagnant.get('id_candidat'):
            votes_par_candidat[gagnant['id_candidat']] += gagnant.get('nombre_voix', 0)

    candidats_tries = sorted(
        candidats_presidentielle,
        key=lambda c: votes_par_candidat.get(c['id_candidat'], 0),
        reverse=True
    )
    candidats_par_orientation = {c['orientation_politique']: c for c in candidats_tries}
    le = LabelEncoder()
    orientations = list(
        set(c['orientation_politique'] for c in candidats_presidentielle if c.get('orientation_politique')))
    le.fit(orientations)
    return candidats_presidentielle, candidats_tries, candidats_par_orientation, le, orientations

