import pandas as pd
import csv

# Load raw CSV (one row = disease + list of symptoms)
with open('data/archive/raw_dataset.csv', 'r') as file:
    rows = list(csv.reader(file))

# Extract unique symptoms
all_symptoms = set()
data = []

for row in rows:
    disease = row[0].strip()
    symptoms = [s.strip().lower() for s in row[1:] if s.strip()]
    data.append((disease, symptoms))
    all_symptoms.update(symptoms)

# Sort symptoms for column order
all_symptoms = sorted(list(all_symptoms))

# Build final dataset
final_rows = []

for disease, symptoms in data:
    symptom_vector = [1 if s in symptoms else 0 for s in all_symptoms]
    symptom_vector.append(disease)
    final_rows.append(symptom_vector)

# Create DataFrame
df = pd.DataFrame(final_rows, columns=all_symptoms + ['prognosis'])

# Save processed CSV
df.to_csv('data/archive/dataset.csv', index=False)
print("✅ Converted and saved as data/dataset.csv")
