"""
=============================================================
  STEP 2: TRAIN THE MACHINE LEARNING MODEL
=============================================================
  This script reads leads_dataset.csv, trains a Random Forest
  classifier to predict Green / Yellow / Red, evaluates it,
  and saves the trained model for the Streamlit app to use.

  Output:
    - lead_scoring_model.pkl  (the trained model)
    - label_encoder.pkl       (encodes text columns to numbers)
    - feature_names.pkl       (list of features the model uses)
    - confusion_matrix.png    (visual of model accuracy)
=============================================================
"""

import pandas as pd
import numpy as np
import joblib        # for saving/loading the model
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (classification_report,
                                      confusion_matrix,
                                      accuracy_score)

# ── matplotlib for the confusion matrix chart ──
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no popup window)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================
# 1. LOAD THE DATASET
# =============================================================
print("📂 Loading dataset...")
df = pd.read_csv("leads_dataset.csv")
print(f"   {len(df):,} rows loaded.\n")


# =============================================================
# 2. ENCODE TEXT COLUMNS INTO NUMBERS
# =============================================================
# Machine learning models only understand numbers, not words like
# "Government" or "Facebook Ad". A LabelEncoder converts:
#   "Facebook Ad" → 0,  "Agent Referral" → 1,  etc.

text_columns = [
    "employment_type",
    "location_match",
    "bedroom_match",
    "lead_source",
    "buy_purpose"
]

encoders = {}  # we'll save one encoder per text column

for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # replace text with numbers
    encoders[col] = le                   # save so we can decode later

print("✅ Text columns encoded.")


# =============================================================
# 3. SPLIT INTO FEATURES (X) AND TARGET (y)
# =============================================================
# X = everything the model LOOKS AT (all columns except the label)
# y = what the model PREDICTS (the label: 0, 1, or 2)

feature_columns = [col for col in df.columns if col != "lead_label"]
X = df[feature_columns]
y = df["lead_label"]

print(f"   Features used ({len(feature_columns)}):")
for f in feature_columns:
    print(f"     • {f}")
print()


# =============================================================
# 4. SPLIT INTO TRAINING SET AND TEST SET
# =============================================================
# We train on 80% of data and test on the remaining 20%.
# The test set is data the model has NEVER seen — this gives
# us a fair measure of real-world accuracy.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,    # 20% for testing
    random_state=42,   # fixed seed for reproducibility
    stratify=y         # keep same label ratio in train & test
)

print(f"🔀 Train size : {len(X_train):,} rows")
print(f"   Test size  : {len(X_test):,} rows\n")


# =============================================================
# 5. TRAIN THE RANDOM FOREST MODEL
# =============================================================
# Random Forest = a "committee" of 300 decision trees that each
# vote on the answer. The majority vote wins.
# It's robust, handles mixed data well, and tells us which
# features matter most.

print("🌲 Training Random Forest... (this takes a few seconds)")

model = RandomForestClassifier(
    n_estimators=300,      # 300 trees in the forest
    max_depth=10,          # each tree can be at most 10 levels deep
    min_samples_leaf=5,    # each leaf needs at least 5 data points
    class_weight="balanced",  # handles unequal Red/Yellow/Green counts
    random_state=42,
    n_jobs=-1              # use all CPU cores to speed up training
)

model.fit(X_train, y_train)
print("✅ Training complete.\n")


# =============================================================
# 6. EVALUATE THE MODEL
# =============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Overall Accuracy : {accuracy * 100:.1f}%\n")

label_names = ["🔴 Red (0)", "🟡 Yellow (1)", "🟢 Green (2)"]
print("📊 Detailed Report:")
print(classification_report(y_test, y_pred, target_names=label_names))

# Explanation of the metrics:
# Precision: Of all leads flagged Green, what % were actually Green?
# Recall   : Of all actual Green leads, what % did we catch?
# F1-score : Balanced average of precision and recall


# =============================================================
# 7. FEATURE IMPORTANCE  (which signals matter most?)
# =============================================================

importances = model.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature":    feature_columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("🔍 Top 10 Most Important Features:")
print(feat_imp_df.head(10).to_string(index=False))
print()


# =============================================================
# 8. SAVE THE CONFUSION MATRIX AS AN IMAGE
# =============================================================
# A confusion matrix shows us where the model makes mistakes.
# Rows = actual label, Columns = predicted label.
# A perfect model has all numbers on the diagonal.

cm = confusion_matrix(y_test, y_pred)
short_labels = ["Red", "Yellow", "Green"]
colors       = ["#e74c3c", "#f39c12", "#2ecc71"]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(short_labels, fontsize=13)
ax.set_yticklabels(short_labels, fontsize=13)
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_ylabel("Actual Label",    fontsize=13)
ax.set_title(f"Confusion Matrix  |  Accuracy: {accuracy*100:.1f}%", fontsize=14)

# Add numbers inside each cell
for row in range(3):
    for col in range(3):
        ax.text(col, row, str(cm[row, col]),
                ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white" if cm[row, col] > cm.max() / 2 else "black")

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("💾 confusion_matrix.png saved.")


# =============================================================
# 9. SAVE THE MODEL AND ENCODERS
# =============================================================
# joblib.dump() serialises Python objects to disk.
# The Streamlit app will load these with joblib.load().

joblib.dump(model,            "lead_scoring_model.pkl")
joblib.dump(encoders,         "label_encoders.pkl")
joblib.dump(feature_columns,  "feature_names.pkl")
joblib.dump(feat_imp_df,      "feature_importance.pkl")

print("💾 lead_scoring_model.pkl saved.")
print("💾 label_encoders.pkl saved.")
print("💾 feature_names.pkl saved.")
print("💾 feature_importance.pkl saved.")
print("\n🏁 All done! You can now run:  streamlit run app.py")