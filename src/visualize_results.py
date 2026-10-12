import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD MODEL COMPARISON RESULTS
# ============================================================

results_path = "results/model_comparison.csv"

df = pd.read_csv(results_path)

print("=" * 60)
print("CREATING MODEL VISUALIZATIONS")
print("=" * 60)

print("\nModel comparison results:")
print(df.to_string(index=False))


# ============================================================
# 2. MODEL COMPARISON GRAPH
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

ax = df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. LOAD ACTUAL AND PREDICTED VALUES
# ============================================================

prediction_path = "results/final_predictions.csv"

prediction_data = pd.read_csv(
    prediction_path
)

y_test = prediction_data["actual"]
y_pred = prediction_data["predicted"]


# ============================================================
# 4. CALCULATE CONFUSION MATRIX AUTOMATICALLY
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 5. CREATE CONFUSION MATRIX VISUALIZATION
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Failed",
        "Successful"
    ]
)

disp.plot()

plt.title(
    "Random Forest (200 Trees) - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("GRAPHS SAVED SUCCESSFULLY!")
print("=" * 60)

print("Saved:")
print("- results/model_comparison.png")
print("- results/confusion_matrix.png")