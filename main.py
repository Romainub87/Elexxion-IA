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
api_keys = {}

# Route pour générer une clé d'API
@app.route('/generate_api_key', methods=['POST'])
def generate_api_key():
    user_id = request.json.get('user_id')
    if not user_id:
        abort(400, description="L'identifiant utilisateur est requis")

    api_key = str(uuid.uuid4())
    api_keys[user_id] = api_key
    return jsonify({"user_id": user_id, "api_key": api_key})

# Middleware pour vérifier la clé d'API
@app.before_request
def verify_api_key():
    if request.endpoint not in ['generate_api_key', 'index']:
        api_key = request.headers.get('x-api-key')
        if api_key not in api_keys.values():
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
    description = {
        "description": "Elexxion AI est une API de prédiction basée sur des données socio-économiques et électorales françaises.",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Affiche la description de l'application et la liste des endpoints."},
            {"path": "/generate_api_key", "method": "POST", "description": "Génère une clé d'API pour un utilisateur."},
            {"path": "/predict", "method": "GET", "description": "Prédit les indicateurs socio-économiques et électoraux pour les 3 prochaines années par rapport à l'année en paramètre (default: 2025)."},
            {"path": "/predict/securite", "method": "GET", "description": "Prédit le nombre d'infractions par type pour les 3 prochaines années (2026,2027,2028)."}
        ]
    }
    return jsonify(description)

if __name__ == '__main__':
    app.run(debug=True)