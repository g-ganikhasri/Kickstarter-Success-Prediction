import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/random_forest_model.joblib")

print("=" * 50)
print("KICKSTARTER PROJECT SUCCESS PREDICTOR")
print("=" * 50)

# Get project details from user
goal = float(input("Enter project goal amount: "))
duration_days = int(input("Enter project duration (days): "))
launch_year = int(input("Enter launch year: "))
launch_month = int(input("Enter launch month (1-12): "))
name_length = int(input("Enter project name length: "))

main_category = input("Enter main category: ")
category = input("Enter category: ")
country = input("Enter country code: ")
currency = input("Enter currency: ")

# Create input dataframe
input_data = pd.DataFrame([{
    "goal": goal,
    "duration_days": duration_days,
    "launch_year": launch_year,
    "launch_month": launch_month,
    "name_length": name_length,
    "main_category": main_category,
    "category": category,
    "country": country,
    "currency": currency
}])

# Make prediction
prediction = model.predict(input_data)[0]

# Get prediction probabilities
probabilities = model.predict_proba(input_data)[0]

failed_probability = probabilities[0] * 100
successful_probability = probabilities[1] * 100

print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

if prediction == 1:
    print("Prediction: SUCCESSFUL")
else:
    print("Prediction: FAILED")

print(f"Probability of Failure: {failed_probability:.2f}%")
print(f"Probability of Success: {successful_probability:.2f}%")

print("=" * 50)