from flask import Flask, jsonify, request
import numpy as np
import joblib
from keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utils import getAllDataPerYear
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    annee = int(request.args.get('annee', 2025))

    model = models.load_model('model.h5')

    # Préparer les données d'entrée
    input_data = np.array([[annee]])

    # Effectuer la prédiction
    prediction = model.predict(input_data)

    return jsonify({
        'point_bourse': float(prediction[0][0]),
        'taux_chomage': float(prediction[0][1])
    })
@app.route('/train', methods=['GET'])
def train_model():
    merged_data = getAllDataPerYear()

    return merged_data

    X = np.array([[item['annee']] for item in merged_data])
    y = np.array([[item['point_bourse'], item['taux_chomage']] for item in merged_data])

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X),
        np.array(y),
        test_size=0.05,
        random_state=42
    )

    model = Sequential([
        Dense(128, input_dim=1, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

    model.save('model.h5')

    return jsonify({'message': 'Model trained and saved successfully.'})

if __name__ == '__main__':
    app.run(debug=True)