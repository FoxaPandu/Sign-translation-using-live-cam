import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# FORCE CORRECT CSV COLUMN NAMES
# ===============================
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]
columns.append("label")

# Read CSV WITHOUT treating first row as header
df = pd.read_csv("gestures.csv", header=None, names=columns)

print("CSV loaded successfully!")
print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# ===============================
# FEATURES AND LABELS
# ===============================
X = df.drop("label", axis=1)
y = df["label"]

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# TRAIN MODEL
# ===============================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ===============================
# PREDICT AND EVALUATE
# ===============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# SAVE MODEL
# ===============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")