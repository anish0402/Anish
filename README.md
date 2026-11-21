This is my first AI project 
# heart_disease.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset load
df = pd.read_csv("heart.csv")
print("Shape:", df.shape)
print(df.head())

# 2. Missing values check
print("\nMissing values:\n", df.isnull().sum())
df = df.dropna()

# 3. Target distribution graph
df["target"].value_counts().plot(kind="bar", title="Heart Disease Count (0 = No, 1 = Yes)")
plt.xlabel("Target")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("target_distribution.png")
plt.close()

# 4. Features and target split
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)

# 7. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# 8. Results
print(f"\nLogistic Regression Accuracy: {lr_acc:.2f}")
print(f"Random Forest Accuracy:       {rf_acc:.2f}")

print("\nRandom Forest Report:\n", classification_report(y_test, rf_pred))

# 9. Best model pick
best_model = rf_model if rf_acc >= lr_acc else lr_model
print("Best model:", "Random Forest" if best_model is rf_model else "Logistic Regression")
