# churn_predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("data/customer_churn_data.csv")

# Preprocessing
df = df.drop(['customerID'], axis=1, errors='ignore')
df = df.dropna()

# Label Encoding for target
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

# One-hot encode categorical features
df = pd.get_dummies(df)

# Features and labels
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns[indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:10], y=features[:10])
plt.title("Top 10 Features Influencing Churn")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# SHAP Explainability
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.savefig("shap_summary_plot.png")