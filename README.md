# Kickstarter Project Success Prediction

## Project Overview

This project uses Machine Learning to predict whether a Kickstarter crowdfunding project will be **successful or failed** based on information available about the project.

The project implements an end-to-end machine learning pipeline including:

- Data preprocessing
- Feature engineering
- Categorical feature encoding
- Numerical feature scaling
- Model training
- Model comparison
- Performance evaluation
- Confusion matrix analysis
- Individual project prediction

The project evaluates multiple classification algorithms and selects a final model based on its performance on unseen test data.

---

## Problem Statement

Crowdfunding platforms such as Kickstarter host thousands of projects across different categories. However, not every project successfully reaches its funding goal.

The objective of this project is to develop a machine learning model that can predict whether a Kickstarter project will be **successful or failed** using project-level metadata.

The prediction is made using information such as:

- Funding goal
- Campaign duration
- Launch date
- Project category
- Country
- Currency
- Project name characteristics

The project avoids using variables that directly represent the final performance of a campaign in order to prevent **data leakage**.

---

## Project Objective

The main objectives of this project are:

1. Analyze historical Kickstarter project data.
2. Preprocess and clean the dataset.
3. Engineer useful features from existing project information.
4. Train multiple machine learning classification models.
5. Compare the models using standard classification metrics.
6. Select a suitable final model.
7. Evaluate the final model using a confusion matrix.
8. Provide a script for predicting the outcome of an individual project.

---

## Dataset

The project uses the Kickstarter Projects dataset containing historical Kickstarter campaigns.

### Dataset File

```text
ks-projects-201801.csv
```

The original dataset contains:

- **378,661 projects**
- **15 variables**

After removing projects whose state was neither `successful` nor `failed`, the final dataset contains:

- **331,675 projects**
- **197,719 failed projects**
- **133,956 successful projects**

### Target Variable

The target variable is created from the original `state` column:

| Original State | Target |
|---|---:|
| failed | 0 |
| successful | 1 |

Projects with other states are removed from the modelling dataset.

---

## Dataset Source

The dataset is commonly distributed as:

```text
ks-projects-201801.csv
```

The dataset is not included in this GitHub repository because of its size.

Place the downloaded CSV file at:

```text
data/
└── ks-projects-201801.csv
```

---

## Features Used

The following features are used for prediction:

| Feature | Type | Description |
|---|---|---|
| `goal` | Numerical | Funding goal set by the creator |
| `duration_days` | Numerical | Length of the campaign in days |
| `launch_year` | Numerical | Year in which the project was launched |
| `launch_month` | Numerical | Month in which the project was launched |
| `name_length` | Numerical | Number of characters in the project name |
| `main_category` | Categorical | Main project category |
| `category` | Categorical | More specific project category |
| `country` | Categorical | Country associated with the project |
| `currency` | Categorical | Currency used by the project |

---

## Data Leakage Prevention

Several variables in the original dataset were deliberately excluded because they contain information about the final outcome of the campaign.

The following variables were **not used as input features**:

- `pledged`
- `backers`
- `usd pledged`
- `usd_pledged_real`

These variables are strongly related to the final campaign performance and would allow the model to indirectly see the outcome it is supposed to predict.

Therefore, they were excluded to reduce **target leakage** and make the prediction more meaningful.

---

## Data Preprocessing

The following preprocessing steps are performed:

### 1. Remove Unknown Project States

Only projects with the following states are retained:

```text
successful
failed
```

### 2. Create Target Variable

```text
failed → 0
successful → 1
```

### 3. Date Processing

The `launched` and `deadline` columns are converted to datetime format.

Campaign duration is calculated as:

```text
duration_days = deadline - launched
```

Additional time-based features are extracted:

```text
launch_year
launch_month
```

### 4. Name Length

The length of each project name is calculated and stored as:

```text
name_length
```

### 5. Categorical Encoding

Categorical variables are converted into numerical representations using:

```text
OneHotEncoder
```

The encoder uses:

```text
handle_unknown="ignore"
```

so that previously unseen categories do not cause prediction errors.

### 6. Numerical Scaling

Numerical variables are standardized using:

```text
StandardScaler
```

### 7. Train-Test Split

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

A stratified split is used so that the class distribution is maintained between the training and testing sets.

---

## Machine Learning Models

Four classification models were trained and evaluated.

### 1. Logistic Regression

Used as a linear classification baseline.

### 2. Random Forest – 100 Trees

An ensemble of decision trees using 100 trees.

### 3. Random Forest – 200 Trees

A larger Random Forest model using 200 trees.

### 4. Gradient Boosting

A boosting-based ensemble model in which multiple weak learners are trained sequentially.

---

## Evaluation Metrics

The models are evaluated using four metrics:

### Accuracy

Measures the overall percentage of correctly classified projects.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many projects predicted as successful were actually successful.

### Recall

Measures how many of the actually successful projects were correctly identified.

### F1 Score

The harmonic mean of precision and recall.

F1 score is useful when both false positives and false negatives are important.

---

## Model Results

The models were evaluated on the same unseen test dataset.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6683 | 0.6144 | 0.4797 | 0.5388 |
| Random Forest (100 trees) | 0.6736 | 0.6122 | 0.5233 | 0.5643 |
| Random Forest (200 trees) | 0.6755 | 0.6144 | 0.5276 | **0.5677** |
| Gradient Boosting | **0.6800** | **0.6472** | 0.4569 | 0.5356 |

### Key Observations

- Gradient Boosting achieved the highest overall accuracy: **68.00%**.
- Gradient Boosting also achieved the highest precision: **64.72%**.
- Random Forest with 200 trees achieved the highest F1 score: **56.77%**.
- Random Forest with 200 trees also achieved the highest recall for the successful class among the tested models.

---

## Final Model

The final selected model is:

```text
Random Forest Classifier
200 Trees
```

### Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 67.55% |
| Precision | 61.44% |
| Recall | 52.76% |
| F1 Score | 56.77% |

The 200-tree Random Forest was selected because it achieved the strongest **F1 score** and successful-class **recall** among the tested models, providing a better balance between identifying successful and failed projects.

---

## Confusion Matrix

The confusion matrix for the final Random Forest model is:

```text
                 Predicted
                 Failed   Successful

Actual Failed     30673      8871

Actual Successful 12655     14136
```

### Interpretation

- **30,673** failed projects were correctly classified as failed.
- **8,871** failed projects were incorrectly classified as successful.
- **12,655** successful projects were incorrectly classified as failed.
- **14,136** successful projects were correctly classified as successful.

The confusion matrix provides a more detailed view of the model's classification performance than accuracy alone.

---

## Project Structure

```text
Kickstarter-Success-Prediction/
│
├── data/
│   └── ks-projects-201801.csv
│
├── models/
│   └── random_forest_model.joblib
│
├── results/
│   ├── model_comparison.csv
│   ├── final_predictions.csv
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── predict.py
│   └── visualize_results.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

> Note: The dataset and trained model file are excluded from the GitHub repository using `.gitignore`.

---

# Installation

## Prerequisites

Make sure the following are installed:

- Python 3.10 or later
- Git
- pip

---

## 1. Clone the Repository

```bash
git clone https://github.com/g-ganikhasri/Kickstarter-Success-Prediction.git
cd Kickstarter-Success-Prediction
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate the Virtual Environment

### Windows

```powershell
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

The main libraries used in this project include:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

---

## 5. Add the Dataset

Download the Kickstarter dataset and place the CSV file inside the `data` directory.

The final structure should be:

```text
data/
└── ks-projects-201801.csv
```

---

# Usage

## Step 1: Run Data Preprocessing

To inspect the dataset, target distribution, selected features and missing values:

```bash
python src/data_preprocessing.py
```

This script:

- Loads the dataset
- Removes unknown project states
- Creates the target variable
- Performs date processing
- Creates engineered features
- Displays dataset information

---

## Step 2: Train the Models

Run:

```bash
python src/train_models.py
```

This script:

1. Loads the dataset.
2. Filters successful and failed projects.
3. Creates the target variable.
4. Performs feature engineering.
5. Splits the data into training and testing sets.
6. Applies preprocessing.
7. Trains four machine learning models.
8. Calculates evaluation metrics.
9. Compares the models.
10. Saves the predictions.
11. Saves the final Random Forest model.

The following files are generated:

```text
results/model_comparison.csv
results/final_predictions.csv
models/random_forest_model.joblib
```

---

## Step 3: Generate Visualizations

Run:

```bash
python src/visualize_results.py
```

This generates:

```text
results/model_comparison.png
results/confusion_matrix.png
```

### Model Comparison

The model comparison graph shows the performance of all four trained models using:

- Accuracy
- Precision
- Recall
- F1 Score

### Confusion Matrix

The confusion matrix visualization shows the classification performance of the final Random Forest model.

---

## Step 4: Make an Individual Prediction

Run:

```bash
python src/predict.py
```

The prediction script loads the trained model and performs a prediction for an individual Kickstarter project using project metadata.

The output classifies the project as:

```text
Successful
```

or

```text
Failed
```

---

# Complete Execution Flow

```text
Kickstarter Dataset
        ↓
Data Cleaning
        ↓
Target Creation
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Preprocessing
        ↓
┌──────────────────────────────┐
│ Machine Learning Models      │
│                              │
│ • Logistic Regression        │
│ • Random Forest – 100 trees  │
│ • Random Forest – 200 trees  │
│ • Gradient Boosting          │
└──────────────────────────────┘
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Random Forest – 200 Trees
        ↓
Individual Prediction
```

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | Machine learning and preprocessing |
| Matplotlib | Visualization |
| Seaborn | Visualization support |
| Joblib | Model serialization |
| Git | Version control |
| GitHub | Source code repository |

---

# Important Files

### `src/data_preprocessing.py`

Performs initial data loading, filtering, feature engineering and dataset inspection.

### `src/train_models.py`

Trains and evaluates all machine learning models and saves the final model.

### `src/visualize_results.py`

Generates the model comparison chart and confusion matrix.

### `src/predict.py`

Loads the trained model and performs individual predictions.

### `requirements.txt`

Contains the Python dependencies required to run the project.

### `.gitignore`

Prevents large/local files such as the dataset, virtual environment and trained model from being uploaded to GitHub.

---

# Results and Generated Files

After training and visualization, the following files are produced:

```text
results/
├── model_comparison.csv
├── final_predictions.csv
├── model_comparison.png
└── confusion_matrix.png
```

These files provide:

- Model performance comparison
- Test-set predictions
- Performance visualization
- Confusion matrix visualization

---

# Limitations

The current implementation has several limitations:

1. The model uses only project-level metadata.
2. Text information from project descriptions is not currently used.
3. No extensive hyperparameter optimization was performed.
4. The model performance is moderate, indicating that Kickstarter success depends on factors beyond the selected metadata.
5. Historical Kickstarter data may contain patterns that do not fully generalize to future campaigns.
6. The dataset is historical and therefore cannot guarantee future campaign outcomes.

---

# Future Improvements

Potential improvements include:

- Incorporating project description text using TF-IDF or word embeddings.
- Using NLP techniques to analyze project names and descriptions.
- Hyperparameter tuning using Grid Search or Randomized Search.
- Applying cross-validation.
- Testing additional algorithms.
- Feature importance analysis.
- Using ensemble methods to combine multiple models.
- Handling class imbalance with suitable sampling or class-weight techniques.
- Building an interactive web interface for predictions.
- Deploying the model as an online prediction service.

---

# Reference Paper

The project was inspired by research on predicting Kickstarter project success using machine learning techniques.

**Paper:**

> Khosla, S., Reinecke, J., & Wittenbrink, B.  
> *Crowdfunding: Predicting Kickstarter Project Success.*

The referenced study investigates Kickstarter project success prediction using multiple machine learning approaches, including logistic regression, random forests, neural networks, SVM and gradient boosting.

---

# Deliverables

This repository contains the main source code, results and project documentation required for the machine learning mini-project.

### Included

- Complete source code
- Data preprocessing implementation
- Model training implementation
- Prediction script
- Visualization script
- Model comparison results
- Confusion matrix results
- README documentation
- Requirements file
- Git version control

### Excluded from Repository

The following files are kept locally because of their size:

```text
data/ks-projects-201801.csv
models/random_forest_model.joblib
```

They can be regenerated or obtained separately by following the setup instructions above.

---

# Project Information

**Course:** Machine Learning Mini-Project

**Project Type:** Classification

**Task:** Kickstarter Project Success Prediction

**Final Model:** Random Forest Classifier (200 Trees)

---

# Conclusion

This project demonstrates an end-to-end machine learning approach for predicting Kickstarter project outcomes.

Four classification algorithms were trained and evaluated using project metadata. Gradient Boosting achieved the highest accuracy, while the 200-tree Random Forest achieved the highest F1 score and successful-class recall among the tested models.

The final system provides a complete workflow from raw Kickstarter data to preprocessing, model training, evaluation, visualization and individual prediction.

The results demonstrate that machine learning can identify useful patterns in crowdfunding project metadata, while also showing that predicting campaign success remains challenging using metadata alone.
