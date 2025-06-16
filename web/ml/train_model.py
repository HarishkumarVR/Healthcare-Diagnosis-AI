import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load preprocessed data
X_train = pd.read_csv('ml/X_train.csv')
X_test = pd.read_csv('ml/X_test.csv')
y_train = pd.read_csv('ml/y_train.csv')['Disease']
y_test = pd.read_csv('ml/y_test.csv')['Disease']

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", accuracy)

# Save model
joblib.dump(model, 'ml/disease_prediction_model.pkl')
print("✅ Model saved as 'disease_prediction_model.pkl'")
