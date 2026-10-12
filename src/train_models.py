import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("KICKSTARTER SUCCESS PREDICTION - MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("data/ks-projects-201801.csv")

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# 2. KEEP ONLY SUCCESSFUL AND FAILED PROJECTS
# ============================================================

df = df[df["state"].isin(["successful", "failed"])].copy()

print("After removing unknown states:", df.shape)


# ============================================================
# 3. CREATE TARGET VARIABLE
# ============================================================

df["target"] = df["state"].map({
    "failed": 0,
    "successful": 1
})

df.drop("state", axis=1, inplace=True)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

df["launched"] = pd.to_datetime(df["launched"])
df["deadline"] = pd.to_datetime(df["deadline"])

# Project duration
df["duration_days"] = (
    df["deadline"] - df["launched"]
).dt.days

# Launch year
df["launch_year"] = df["launched"].dt.year

# Launch month
df["launch_month"] = df["launched"].dt.month

# Project name length
df["name_length"] = (
    df["name"]
    .fillna("")
    .str.len()
)


# ============================================================
# 5. SELECT FEATURES
# ============================================================

features = [
    "goal",
    "duration_days",
    "launch_year",
    "launch_month",
    "name_length",
    "main_category",
    "category",
    "country",
    "currency"
]

X = df[features]
y = df["target"]


print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_features = [
    "goal",
    "duration_days",
    "launch_year",
    "launch_month",
    "name_length"
]

categorical_features = [
    "main_category",
    "category",
    "country",
    "currency"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest (100 trees)": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest (200 trees)": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

final_model = None
final_predictions = None


for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    # Create complete pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate metrics automatically
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print results
    print("\nResults:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Failed", "Successful"]
        )
    )

    # Store metrics automatically
    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })


    # --------------------------------------------------------
    # Save Random Forest 200 as final model
    # --------------------------------------------------------

    if model_name == "Random Forest (200 trees)":

        final_model = model
        final_predictions = y_pred


# ============================================================
# 10. SAVE MODEL COMPARISON RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_path = "results/model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nModel comparison saved to:")
print(results_path)


# ============================================================
# 11. SAVE FINAL MODEL PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": final_predictions
})

prediction_path = "results/final_predictions.csv"

prediction_results.to_csv(
    prediction_path,
    index=False
)

print("\nFinal model predictions saved to:")
print(prediction_path)


# ============================================================
# 12. SAVE FINAL MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model_path = "models/random_forest_model.joblib"

joblib.dump(
    final_model,
    model_path
)

print("\nFinal model saved successfully!")
print("Saved to:", model_path)


# ============================================================
# 13. FINAL MODEL SUMMARY
# ============================================================

final_result = results_df[
    results_df["Model"] == "Random Forest (200 trees)"
].iloc[0]

print("\n" + "=" * 60)
print("FINAL MODEL: RANDOM FOREST (200 TREES)")
print("=" * 60)

print(f"Accuracy:  {final_result['Accuracy']:.4f}")
print(f"Precision: {final_result['Precision']:.4f}")
print(f"Recall:    {final_result['Recall']:.4f}")
print(f"F1 Score:  {final_result['F1 Score']:.4f}")

print("\nTraining completed successfully!")