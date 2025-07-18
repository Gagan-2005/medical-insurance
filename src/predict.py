import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv("medical_insurance.csv")

# Check and clean column names if needed
data.columns = data.columns.str.strip()

# Encode categorical columns
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

data['sex'] = le_sex.fit_transform(data['sex'])
data['smoker'] = le_smoker.fit_transform(data['smoker'])
data['region'] = le_region.fit_transform(data['region'])

# Features and target
X = data.drop(['charges'], axis=1)
y = data['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "insurance_model.pkl")
joblib.dump(le_sex, "encoder_sex.pkl")
joblib.dump(le_smoker, "encoder_smoker.pkl")
joblib.dump(le_region, "encoder_region.pkl")

# Predict for user input
def predict_charges():
    print("Enter patient details for insurance cost prediction:\n")

    age = int(input("Age: "))
    sex = input("Sex (male/female): ")
    bmi = float(input("BMI: "))
    children = int(input("Number of children: "))
    smoker = input("Smoker (yes/no): ")
    region = input("Region (southwest, southeast, northwest, northeast): ")

    # Load encoders
    le_sex = joblib.load("encoder_sex.pkl")
    le_smoker = joblib.load("encoder_smoker.pkl")
    le_region = joblib.load("encoder_region.pkl")

    # Encode inputs
    sex_encoded = le_sex.transform([sex])[0]
    smoker_encoded = le_smoker.transform([smoker])[0]
    region_encoded = le_region.transform([region])[0]

    # Form input array
    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])

    # Load model and predict
    model = joblib.load("insurance_model.pkl")
    prediction = model.predict(input_data)

    print(f"\nEstimated Insurance Cost: ${prediction[0]:.2f}")

if __name__ == "__main__":
    predict_charges()
