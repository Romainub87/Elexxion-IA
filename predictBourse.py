import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from utils import getAllDataPerYear

# Load the data
data = getAllDataPerYear()

# Convert to DataFrame
df = pd.DataFrame(data)

# Handle missing values (e.g., replace None with 0 for simplicity)
df['nombre_jour_pic_particules_fines'] = df['nombre_jour_pic_particules_fines'].fillna(0)

# Features and targets
X = df[['annee' ]].values
y = df[['point_bourse', 'taux_chomage']].values

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='linear')  # Output layer for 2 targets
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Make predictions
predictions = model.predict(X_test)
predictions = scaler_y.inverse_transform(predictions)  # Inverse transform to original scale

# Example prediction for a specific year
example_input = np.array([[2025]])
example_input_scaled = scaler_X.transform(example_input)
example_prediction_scaled = model.predict(example_input_scaled)
example_prediction = scaler_y.inverse_transform(example_prediction_scaled)
print(f"Prediction for 2025: {example_prediction}")