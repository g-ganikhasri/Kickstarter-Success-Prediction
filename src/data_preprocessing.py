import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/ks-projects-201801.csv")

print("Original dataset shape:", df.shape)


# ==========================================
# 2. KEEP ONLY SUCCESSFUL AND FAILED PROJECTS
# ==========================================

df = df[df["state"].isin(["successful", "failed"])].copy()

print("After removing unknown states:", df.shape)

print("\nProject states:")
print(df["state"].value_counts())


# ==========================================
# 3. CREATE TARGET VARIABLE
# ==========================================

df["target"] = df["state"].map({
    "failed": 0,
    "successful": 1
})

# Remove original state column
df.drop("state", axis=1, inplace=True)


# ==========================================
# 4. CREATE USEFUL FEATURES
# ==========================================

# Convert dates
df["launched"] = pd.to_datetime(df["launched"])
df["deadline"] = pd.to_datetime(df["deadline"])

# Campaign duration in days
df["duration_days"] = (
    df["deadline"] - df["launched"]
).dt.days

# Launch year
df["launch_year"] = df["launched"].dt.year

# Launch month
df["launch_month"] = df["launched"].dt.month

# Name length
df["name_length"] = df["name"].fillna("").str.len()


# ==========================================
# 5. SELECT FEATURES
# ==========================================

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


# ==========================================
# 6. DISPLAY FINAL DATA
# ==========================================

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nMissing values:")
print(X.isnull().sum())

print("\nFinal feature data:")
print(X.head())

print("\nFinal dataset shape:")
print("X:", X.shape)
print("y:", y.shape)