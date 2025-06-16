from flask import Flask, render_template, request
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

# Load trained model
model = joblib.load('ml/disease_prediction_model.pkl')

# Load symptom columns from training data
X_train = pd.read_csv('ml/X_train.csv')
symptom_columns = X_train.columns.tolist()

# Load label encoder for decoding predicted classes
with open('ml/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get symptoms from form as list
        input_symptoms = request.form.getlist('symptoms')

        # Create input vector of zeros
        input_vector = [0] * len(symptom_columns)
        for symptom in input_symptoms:
            if symptom in symptom_columns:
                idx = symptom_columns.index(symptom)
                input_vector[idx] = 1

        # Predict disease
        pred_class = model.predict([input_vector])[0]
        prediction = le.inverse_transform([pred_class])[0]

    return render_template('index.html', symptom_columns=symptom_columns, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
