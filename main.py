import uuid
from flask import Flask, jsonify, request, abort
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from securite_utils import getSecuriteByYearAndType
from utils import getAllDataPerYear
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Simuler une base de données pour stocker les clés d'API
api_keys = []

# Route pour générer une clé d'API
@app.route('/generate_api_key', methods=['GET'])
def generate_api_key():
    api_key = str(uuid.uuid4())
    api_keys.append(api_key)
    return jsonify({"api_key": api_key})

# Middleware pour vérifier la clé d'API
@app.before_request
def verify_api_key():
    if request.path.lstrip('/').split('/')[0] not in ['generate_api_key', '']:
        api_key = request.headers.get('x-api-key')
        if api_key not in api_keys:
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
        joblib.dump(model, r'model.h5')

    annee_courante = int(request.args.get('annee', 2025))
    annees = np.array([[annee_courante + i] for i in range(1, 4)])

    model = joblib.load(r'model.h5')

    # Effectuer les prédictions
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
    result = getSecuriteByYearAndType()

    predictions_result = {}

    for annee in [2025, 2026, 2027]:
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
                <li style="white-space: pre-line; word-break: break-word;"><code>GET /generate_api_key</code> : Génère une clé d&#39;API pour un utilisateur.<br><b>Exemple de réponse&nbsp;:</b><br><code>{"api_key": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}</code></li>
                <li style="white-space: pre-line; word-break: break-word;"><code>GET /predict?annee=?</code> : Prédit les indicateurs socio-économiques et électoraux pour les 3 prochaines années par rapport à l&#39;année en paramètre (default: 2025).<br><b>Exemple de réponse&nbsp;:</b><br><code>[{<br>"annee":2026,<br>"point_bourse":1234.56,<br>"population":67000000,<br>"taux_chomage":8.5,<br>"nombre_jour_pic_particules_fines":12,<br>"participation_tour2":75.2<br>},...]</code></li>
                <li style="white-space: pre-line; word-break: break-word;"><code>GET /predict/securite</code> : Prédit le nombre d&#39;infractions par type pour les 3 prochaines années (2026,2027,2028).<br><b>Exemple de réponse&nbsp;:</b><br><code>{
                "2026":[{<br>"nombre_infraction":123,
                "type_infraction":"Vol"
                }],...}</code></li>
            </ul>
        </div>
    </body>
    </html>
    """
    return description, 200, {'Content-Type': 'text/html'}

if __name__ == '__main__':
    app.run(debug=True)