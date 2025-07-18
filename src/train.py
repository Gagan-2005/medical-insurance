import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv("medical_insurance.csv")

# Clean column names if needed
df.columns = df.columns.str.strip()

# Encode categorical variables
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['region'] = le_region.fit_transform(df['region'])

# Features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and encoders
joblib.dump(model, "insurance_model.pkl")
joblib.dump(le_sex, "encoder_sex.pkl")
joblib.dump(le_smoker, "encoder_smoker.pkl")
joblib.dump(le_region, "encoder_region.pkl")

print("✅ Model trained and saved as 'insurance_model.pkl'")
