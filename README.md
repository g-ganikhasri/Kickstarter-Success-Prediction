# Kickstarter Project Success Prediction

A machine learning project that predicts whether a Kickstarter crowdfunding project will be **successful or failed** using information available about the project before its final outcome.

---

## Project Overview

Crowdfunding platforms such as Kickstarter host thousands of projects across different categories. However, many projects fail to reach their funding goals.

This project develops and compares multiple machine learning classification models to predict Kickstarter project outcomes based on project characteristics such as:

- Funding goal
- Campaign duration
- Launch year and month
- Project name length
- Main category
- Sub-category
- Country
- Currency

The project focuses on avoiding **data leakage** by excluding variables that directly reveal the final outcome of a campaign.

---

## Problem Statement

The objective is to build a machine learning model that can predict whether a Kickstarter project will be:

- **Successful (1)**
- **Failed (0)**

The prediction is based only on information that could reasonably be available when evaluating a project, rather than using its final funding performance.

---

## Dataset

The project uses the **Kickstarter Projects 2018** dataset.

### Dataset source

**Kaggle:** Kickstarter Projects dataset by `kemical`

The dataset contains Kickstarter campaigns collected between **2009 and 2018**.

### Original dataset

- **378,661 projects**
- **15 variables**

### After preprocessing

Projects with states other than `successful` and `failed` were removed.

Final dataset:

- **331,675 projects**
- **9 input features**
- **1 binary target variable**

### Target distribution

| Target | Meaning | Count |
|---|---|---:|
| 0 | Failed | 197,719 |
| 1 | Successful | 133,956 |

---

## Features Used

The following features are used for prediction:

| Feature | Description | Type |
|---|---|---|
| `goal` | Funding goal of the project | Numerical |
| `duration_days` | Campaign duration in days | Numerical |
| `launch_year` | Year the campaign was launched | Numerical |
| `launch_month` | Month the campaign was launched | Numerical |
| `name_length` | Number of characters in project name | Numerical |
| `main_category` | Main project category | Categorical |
| `category` | Specific project category | Categorical |
| `country` | Country of the project | Categorical |
| `currency` | Campaign currency | Categorical |

### Features excluded to prevent data leakage

The following variables were deliberately excluded:

- `pledged`
- `backers`
- `usd pledged`
- `usd_pledged_real`

These variables contain information about the funding performance of a project and therefore would reveal information about the final outcome.

---

## Data Preprocessing

The following preprocessing steps are performed:

1. Remove projects whose state is not `successful` or `failed`.
2. Convert the project state into a binary target:
   - `failed → 0`
   - `successful → 1`
3. Convert `launched` and `deadline` into datetime values.
4. Calculate campaign duration in days.
5. Extract launch year.
6. Extract launch month.
7. Calculate project name length.
8. Separate numerical and categorical features.
9. Standardize numerical features using `StandardScaler`.
10. One-hot encode categorical features using `OneHotEncoder`.
11. Split the dataset into training and testing sets using an 80/20 stratified split.

---

## Machine Learning Models

Four classification approaches were evaluated:

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Random Forest – 100 Trees

An ensemble of 100 decision trees.

### 3. Random Forest – 200 Trees

An ensemble of 200 decision trees. This model was selected as the final model based on its overall F1 performance and successful-class recall.

### 4. Gradient Boosting

An ensemble boosting method that builds trees sequentially to improve prediction performance.

---

## Evaluation Metrics

The models were evaluated using:

### Accuracy

The proportion of total predictions that were correct.

### Precision

Measures how many projects predicted as successful were actually successful.

### Recall

Measures how many of the actually successful projects were correctly identified.

### F1 Score

The harmonic mean of precision and recall, providing a balance between the two.

---

## Results

The models were evaluated on the same held-out test set.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 66.83% | 61.44% | 47.97% | 53.88% |
| Random Forest (100 trees) | 67.36% | 61.22% | 52.33% | 56.43% |
| **Random Forest (200 trees)** | **67.55%** | **61.44%** | **52.76%** | **56.77%** |
| Gradient Boosting | **68.00%** | **64.72%** | 45.69% | 53.56% |

### Model selection

Gradient Boosting achieved the highest overall accuracy at **68.00%**.

However, the **200-tree Random Forest** achieved:

- The highest F1 score: **56.77%**
- The highest recall for the successful class: **52.76%**
- A balanced precision-recall performance

Therefore, the **Random Forest with 200 trees** was selected as the final model.

This selection prioritizes a better balance between identifying successful projects and avoiding incorrect predictions rather than relying on accuracy alone.

---

## Confusion Matrix

The final Random Forest model produced the following confusion matrix on the test set:

```text
                    Predicted
                  Failed  Successful

Actual Failed      30673      8871
Actual Successful  12655     14136
