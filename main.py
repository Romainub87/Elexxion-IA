import uuid
from flask import Flask, jsonify, request, abort, current_app
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from utils.election_utils import getAllDataWhereElectionPerYear
from utils.securite_utils import getSecuriteByYearAndType
from utils.indicators_utils import getAllDataPerYear
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from utils.election_utils import getCandidatsClassementEtLabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error

app = Flask(__name__)

api_keys = []

api_key = "rM+_8rujGnjryM%jvC@bPMgjfgPe0S&6"

@app.before_request
def verify_api_key():
    if request.path.lstrip('/').split('/')[0] not in ['', 'static']:
        header_key = request.headers.get('x-api-key')
        if header_key != api_key:
            abort(401, description="Clé d'API invalide ou manquante")

@app.route('/predict', methods=['GET'])
def predict():
    if request.args.get('train') == '1':
        merged_data = getAllDataPerYear()
        X = np.array([[item['annee']] for item in merged_data])
        y = np.array([
            [
                item.get('point_bourse') or -1,
                item.get('taux_chomage') or -1,
                item.get('nombre_jour_pic_particules_fines') or -1,
                item['gagnant'].get('participationPourcentage') or -1,
                item.get('population'),
            ]
            for item in merged_data if item
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(X),
            np.array(y),
            test_size=0.05,
            random_state=42
        )
        model = MultiOutputRegressor(LinearRegression())
        model.fit(X_train, y_train)
        joblib.dump(model, r'models/model.h5')

    annee_courante = int(request.args.get('annee', 2025))
    annees = np.array([[annee_courante + i] for i in range(1, 4)])

    model = joblib.load(r'models/model.h5')

    # Effectuer les prédictions des indicateurs socio-économiques
    predictions = model.predict(annees)

    result = []
    for i, prediction in enumerate(predictions):
        result.append({
            'annee': annee_courante + i + 1,
            'point_bourse': round(float(prediction[0]), 2),
            'population': round(float(prediction[4]), 2),
            'taux_chomage': round(float(prediction[1]), 2),
            'nombre_jour_pic_particules_fines': round(float(prediction[2]), 2),
            'participation_tour2': round(float(prediction[3]), 2)
        })

    return jsonify(result)

@app.route('/predict/securite', methods=['GET'])
def predict_securite():
    annee_depart = int(request.args.get('annee', 2025))
    result = getSecuriteByYearAndType()

    predictions_result = {}

    for annee in [annee_depart + i for i in range(1, 4)]:
        predictions_result[annee] = []
        for infraction_type, data in result.items():
            annees = np.array([d["annee"] for d in data]).reshape(-1, 1)
            valeurs = np.array([d["nombre_infraction"] for d in data])
            if len(annees) > 1:
                reg = LinearRegression()
                reg.fit(annees, valeurs)
                prediction = int(reg.predict(np.array([[annee]]))[0])
                predictions_result[annee].append({
                    "nombre_infraction": prediction,
                    "type_infraction": infraction_type,
                })

    return jsonify(predictions_result)

@app.route('/predict/election', methods=['GET'])
def predict_election():
    annee_courante = int(request.args.get('annee', 2025))
    annees = np.array([[annee_courante + i] for i in range(1, 4)])

    merged_data = getAllDataWhereElectionPerYear()

    # Prédire la sécurité pour les années futures
    with current_app.test_request_context():
        securite_predictions = getSecuriteByYearAndType()
        securite_future = {}
        for annee in [annee_courante + i for i in range(1, 4)]:
            securite_future[annee] = {}
            for infraction_type, data in securite_predictions.items():
                annees_hist = np.array([d["annee"] for d in data]).reshape(-1, 1)
                valeurs = np.array([d["nombre_infraction"] for d in data])
                if len(annees_hist) > 1:
                    reg = LinearRegression()
                    reg.fit(annees_hist, valeurs)
                    prediction = int(reg.predict(np.array([[annee]]))[0])
                    securite_future[annee][infraction_type] = prediction
                else:
                    securite_future[annee][infraction_type] = valeurs[0] if len(valeurs) else 0

    if request.args.get('train') == '1':
        all_infraction_types = sorted({
            d['type_infraction']
            for data in getSecuriteByYearAndType().values()
            for d in data
        })

        X_classif = []
        for item in merged_data:
            features = [
                item['annee'],
                item.get('point_bourse', -1),
                item.get('taux_chomage', -1),
                item.get('nombre_jour_pic_particules_fines', -1),
                item.get('population')
            ]
            securite_annee = {s['type_infraction']: s['nombre_infraction'] for s in item.get('securite', [])}
            features += [securite_annee.get(t, 0) for t in all_infraction_types]
            X_classif.append(features)
        X_classif = np.array(X_classif)

        y_classif = np.array([
            f"{item['gagnant'].get('orientation_politique', '')}__{item['annee']}_{item.get('point_bourse', -1)}_{item.get('taux_chomage', -1)}_{item.get('nombre_jour_pic_particules_fines', -1)}_{item.get('population')}"
            for item in merged_data
        ])
        le = LabelEncoder()
        y_classif_encoded = le.fit_transform(y_classif)
        model_classif = RandomForestClassifier()
        model_classif.fit(X_classif, y_classif_encoded)
        joblib.dump(model_classif, r'models/model_classif.h5')
        joblib.dump(all_infraction_types, r'models/securite_types.h5')

    model_classif = joblib.load(r'models/model_classif.h5')
    model_reg = joblib.load(r'models/model.h5')
    all_infraction_types = joblib.load(r'models/securite_types.h5')

    predictions_reg = model_reg.predict(annees)

    X_pred = []
    for i, pred in enumerate(predictions_reg):
        annee = annee_courante + i + 1
        features = [
            annee,
            pred[0],
            pred[1],
            pred[2],
            pred[4],
        ]
        securite_annee = securite_future.get(annee, {})
        features += [securite_annee.get(t, 0) for t in all_infraction_types]
        X_pred.append(features)
    X_pred = np.array(X_pred)

    candidats_presidentielle, candidats_tries, candidats_par_orientation, le, orientations = getCandidatsClassementEtLabelEncoder(
        merged_data)

    predictions_classif = model_classif.predict(X_pred)
    orientations_predites = le.inverse_transform(predictions_classif)

    result = []
    for i, orientation_predite in enumerate(orientations_predites):
        gagnant = candidats_par_orientation.get(orientation_predite, {})
        result.append({
            'annee': annee_courante + i + 1,
            'gagnant': {
                'nom_candidat': gagnant.get('nom'),
                'orientation_politique': gagnant.get('orientation_politique')
            }
        })

    return jsonify(result)

@app.route('/predict/accuracy', methods=['GET'])
def predict_accuracy():
    # Charger les données et le modèle
    merged_data = getAllDataPerYear()
    X = np.array([[item['annee']] for item in merged_data])
    y = np.array([
        [
            item.get('point_bourse') or -1,
            item.get('taux_chomage') or -1,
            item.get('nombre_jour_pic_particules_fines') or -1,
            item['gagnant'].get('participationPourcentage') or -1,
            item.get('population'),
        ]
        for item in merged_data if item
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = joblib.load(r'models/model.h5')

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calcul de la MAE et RMSE pour chaque indicateur
    mae = np.mean(np.abs(y_test - y_pred), axis=0)

    return jsonify({
        "mae": [round(float(m), 2) for m in mae],
    })

@app.route('/predict/securite/accuracy', methods=['GET'])
def predict_securite_accuracy():
    result = getSecuriteByYearAndType()

    mae_per_type = {}
    for infraction_type, data in result.items():
        annees = np.array([d["annee"] for d in data]).reshape(-1, 1)
        valeurs = np.array([d["nombre_infraction"] for d in data])
        if len(annees) > 1:
            # Diviser les données en train/test
            X_train, X_test, y_train, y_test = train_test_split(
                annees, valeurs, test_size=0.2, random_state=42
            )
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            # Calculer la MAE pour ce type d'infraction
            mae_per_type[infraction_type] = mean_absolute_error(y_test, y_pred)

    return jsonify({infraction: round(mae, 2) for infraction, mae in mae_per_type.items()})

@app.route('/')
def index():
    description = """
    <!DOCTYPE html>
    <html lang='fr'>
    <head>
        <meta charset='UTF-8'>
        <title>Elexxion AI - Documentation API</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f7f7fa; color: #222; margin: 0; padding: 0; }
            .container { max-width: 700px; margin: 40px auto; background: #fff; border-radius: 10px; box-shadow: 0 2px 8px #0001; padding: 32px; }
            h1 { color: #2a4d9b; }
            h2 { color: #3b3b3b; margin-top: 32px; }
            ul { padding-left: 24px; }
            li { margin-bottom: 10px; }
            code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }
            .logo { font-size: 2.5em; font-weight: bold; color: #2a4d9b; margin-bottom: 0.2em; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">Elexxion AI</div>
            <p>Elexxion AI est une API de prédiction basée sur des données socio-économiques et électorales françaises.</p>
            <h2>Endpoints disponibles :</h2>
            <ul>
                <li style="white-space: pre-line; word-break: break-word;"><code>GET /</code> : Affiche la description de l&#39;application et la liste des endpoints.</li>
                <li style="white-space: pre-line; word-break: break-word;"><code>GET /predict?annee=?</code> : Prédit les indicateurs socio-économiques et électoraux pour les 3 prochaines années par rapport à l&#39;année en paramètre (default: 2025).<br><b>Exemple de réponse&nbsp;:</b><br><code>[{<br>"annee":2026,<br>"point_bourse":1234.56,<br>"population":67000000,<br>"taux_chomage":8.5,<br>"nombre_jour_pic_particules_fines":12,<br>"participation_tour2":75.2<br>},...]</code></li>
                <li style="white-space: pre-line; word-break: break-word;"><code>GET /predict/securite</code> : Prédit le nombre d&#39;infractions par type pour les 3 prochaines années (ex: 2026,2027,2028).<br><b>Exemple de réponse&nbsp;:</b><br><code>{
                "2026":[{<br>"nombre_infraction":123,
                "type_infraction":"Vol"
                }],...}</code></li>
                <li style="white-space: pre-line; word-break: break-word;"><code>GET /predict/election?annee=?</code> : Prédit l&#39;orientation politique du gagnant de la présidentielle pour les 3 prochaines années à partir de l&#39;année donnée (default: 2025).<br><b>Exemple de réponse&nbsp;:</b><br><code>[{<br>"annee":2026,<br>"gagnant": {"nom_candidat": "Dupont", "orientation_politique": "Centre"}<br>},...]</code></li>
            </ul>
        </div>
    </body>
    </html>
    """
    return description, 200, {'Content-Type': 'text/html'}

if __name__ == '__main__':
    app.run(debug=True)