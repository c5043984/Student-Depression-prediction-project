import subprocess
import sys

# Install required packages if not available
required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'openpyxl']
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_excel("student.xlsx")

print("First 5 rows:\n", df.head())

# =========================
# 2. PREPROCESSING
# =========================

# Drop unnecessary columns if exist
for col in ["id", "Name"]:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Handle missing values
for col in df.select_dtypes(include=np.number):
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# =========================
# 3. VISUALIZATION
# =========================

sns.countplot(x='Depression', data=df)
plt.title("Depression Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================
# 4. MODEL BUILDING
# =========================

X = df.drop("Depression", axis=1)
y = df["Depression"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# 5. EVALUATION
# =========================

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# =========================
# 6. PREDICTION
# =========================

# Example new data (modify based on your dataset)
new_data = pd.DataFrame([X.iloc[0]])

new_scaled = scaler.transform(new_data)
prediction = model.predict(new_scaled)

print("\nPrediction for new student:", "Depressed" if prediction[0] == 1 else "Not Depressed")