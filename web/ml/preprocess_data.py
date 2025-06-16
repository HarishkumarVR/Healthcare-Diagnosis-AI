import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load raw dataset
df = pd.read_csv('data/dataset.csv')

# Separate features (X) and labels (y)
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Encode the disease labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save the label encoder
with open('ml/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Save the processed data
X_train.to_csv('ml/X_train.csv', index=False)
X_test.to_csv('ml/X_test.csv', index=False)
pd.DataFrame(y_train, columns=['Disease']).to_csv('ml/y_train.csv', index=False)
pd.DataFrame(y_test, columns=['Disease']).to_csv('ml/y_test.csv', index=False)

print("✅ Preprocessing complete.")
