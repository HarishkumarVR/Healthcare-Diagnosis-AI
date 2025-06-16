import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load('ml/disease_prediction_model.pkl')

# Load symptom columns from training data
X_train = pd.read_csv('ml/X_train.csv')
symptom_columns = X_train.columns.tolist()

# Load LabelEncoder to decode predicted class
from sklearn.preprocessing import LabelEncoder
import pickle

# Load target encoder (disease labels)
with open('ml/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# --- USER INPUT SECTION ---
# Example symptoms provided by user
input_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions']

# Create input vector
input_vector = [0] * len(symptom_columns)
for symptom in input_symptoms:
    if symptom in symptom_columns:
        index = symptom_columns.index(symptom)
        input_vector[index] = 1
    else:
        print(f"⚠️ Warning: '{symptom}' is not a recognized symptom")

# Predict disease
prediction = model.predict([input_vector])[0]
predicted_disease = le.inverse_transform([prediction])[0]
print(f"✅ Predicted Disease: {predicted_disease}")
