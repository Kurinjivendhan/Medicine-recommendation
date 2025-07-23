import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
df = pd.read_csv("dataset.csv")

# Step 2: Encode categorical features
le_gender = LabelEncoder()
le_sym1 = LabelEncoder()
le_sym2 = LabelEncoder()
le_disease = LabelEncoder()
le_medicine = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Symptom1'] = le_sym1.fit_transform(df['Symptom1'])
df['Symptom2'] = le_sym2.fit_transform(df['Symptom2'])
df['Existing Disease'] = le_disease.fit_transform(df['Existing Disease'])
df['Recommended Medicine'] = le_medicine.fit_transform(df['Recommended Medicine'])

# Step 3: Split features and target
X = df.drop("Recommended Medicine", axis=1)
y = df["Recommended Medicine"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and train the KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Define recommendation function
def recommend_medicine(age, gender, sym1, sym2, disease):
    # Validate user input
    valid_genders = list(le_gender.classes_)
    valid_sym1 = list(le_sym1.classes_)
    valid_sym2 = list(le_sym2.classes_)
    valid_disease = list(le_disease.classes_)

    if gender not in valid_genders:
        raise ValueError(f"Invalid gender. Valid options: {valid_genders}")
    if sym1 not in valid_sym1:
        raise ValueError(f"Invalid Symptom1. Valid options: {valid_sym1}")
    if sym2 not in valid_sym2:
        raise ValueError(f"Invalid Symptom2. Valid options: {valid_sym2}")
    if disease not in valid_disease:
        raise ValueError(f"Invalid Existing Disease. Valid options: {valid_disease}")

    gender_enc = le_gender.transform([gender])[0]
    sym1_enc = le_sym1.transform([sym1])[0]
    sym2_enc = le_sym2.transform([sym2])[0]
    disease_enc = le_disease.transform([disease])[0]

    input_data = [[age, gender_enc, sym1_enc, sym2_enc, disease_enc]]
    pred = model.predict(input_data)
    medicine = le_medicine.inverse_transform(pred)[0]
    return medicine

# Step 8: User input for recommendation
print("\n--- Personalized Medicine Recommendation ---")
print("Valid Genders:", list(le_gender.classes_))
print("Valid Symptom1:", list(le_sym1.classes_))
print("Valid Symptom2:", list(le_sym2.classes_))
print("Valid Existing Diseases:", list(le_disease.classes_))

age = int(input("Enter age: "))
gender = input("Enter gender: ")
sym1 = input("Enter first symptom: ")
sym2 = input("Enter second symptom: ")
disease = input("Enter existing disease: ")

try:
    recommendation = recommend_medicine(age, gender, sym1, sym2, disease)
    print(f"\n✅ Recommended Medicine: {recommendation}")
except Exception as e:
    print(f"\n⚠️ Invalid input: {e}")
