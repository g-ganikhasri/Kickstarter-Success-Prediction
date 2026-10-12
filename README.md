# Kickstarter Project Success Prediction

A machine learning mini-project that predicts whether a Kickstarter crowdfunding project will be successful or failed using project information available before the campaign outcome is known.

## 1. Problem Statement

Kickstarter hosts a large number of crowdfunding campaigns across different categories, countries, and currencies. The success of a campaign can depend on factors such as its funding goal, campaign duration, category, launch timing, and other project metadata.

The objective of this project is to build a classification system that uses these pre-outcome project characteristics to predict whether a Kickstarter project will be successful (1) or failed (0).

## 2. Dataset

The project uses the Kickstarter Projects 2018 dataset.

- Dataset file: `ks-projects-201801.csv`
- Original dataset size: 378,661 projects
- Projects retained for this study: 331,675
- Only projects with `successful` or `failed` states are used.
- Projects with other/unknown states are removed.

### Target Variable

| Value | Meaning |
|---|---|
| `0` | Failed |
| `1` | Successful |

## 3. Features Used

Only information that can be available before the final campaign outcome is used as a predictor.

### Numerical Features

- `goal` — funding goal of the project
- `duration_days` — campaign duration calculated from launch and deadline
- `launch_year` — year in which the campaign was launched
- `launch_month` — month in which the campaign was launched
- `name_length` — number of characters in the project name

### Categorical Features

- `main_category`
- `category`
- `country`
- `currency`

### Features Excluded

Outcome-related variables such as `pledged`, `backers`, `usd pledged`, and `usd_pledged_real` are not used because they are associated with the campaign outcome and would introduce data leakage.

## 4. Data Preprocessing

The following preprocessing steps are performed:

1. Load the Kickstarter CSV dataset.
2. Keep only projects labelled `successful` or `failed`.
3. Convert the project state into a binary target variable.
4. Convert launch and deadline columns to datetime format.
5. Calculate campaign duration.
6. Extract launch year and launch month.
7. Calculate project-name length.
8. Separate numerical and categorical features.
9. Standardize numerical features using `StandardScaler`.
10. One-hot encode categorical features using `OneHotEncoder`.
11. Split the dataset into training and testing sets using an 80:20 stratified split.

The preprocessing and model are combined using a scikit-learn `Pipeline` and `ColumnTransformer`.

## 5. Machine Learning Models

Four classification models are evaluated:

1. Logistic Regression
2. Random Forest — 100 trees
3. Random Forest — 200 trees
4. Gradient Boosting

The purpose of comparing multiple models is to determine which approach provides the most useful balance for the binary classification task.

## 6. Evaluation Metrics

The models are evaluated using:

- Accuracy — proportion of all predictions that are correct.
- Precision — proportion of predicted successful projects that are actually successful.
- Recall — proportion of actual successful projects correctly identified.
- F1 Score — harmonic mean of precision and recall.

A confusion matrix is also generated for the final model.

## 7. Results

The results below are generated automatically by `train_models.py` from predictions on the held-out test set.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 66.83% | 61.44% | 47.97% | 53.88% |
| Random Forest (100 trees) | 67.36% | 61.22% | 52.33% | 56.43% |
| **Random Forest (200 trees)** | **67.55%** | **61.44%** | **52.76%** | **56.77%** |
| Gradient Boosting | **68.00%** | **64.72%** | 45.69% | 53.56% |

### Final Model

The Random Forest with 200 trees is selected as the final model because it achieved the highest F1 score and the highest recall for the successful class among the tested models, providing a better balance between identifying successful and failed projects.

The final model achieved:

- Accuracy: **67.55%**
- Precision: **61.44%**
- Recall: **52.76%**
- F1 Score: **56.77%**

Gradient Boosting achieved slightly higher accuracy and precision, but its recall and F1 score were lower than those of the selected 200-tree Random Forest.

## 8. Project Structure

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

## 9. Installation

### Step 1: Clone the repository

```bash
git clone <YOUR_PRIVATE_GITHUB_REPOSITORY_URL>
cd Kickstarter-Success-Prediction
```

### Step 2: Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

## 10. Dataset Setup

Place the dataset inside the `data` folder:

```text
data/
└── ks-projects-201801.csv
```

The dataset is not generated by the project code, so it must be downloaded separately if it is not included in the repository.

## 11. How to Run

All commands should be executed from the project root directory.

### Train the Models

```powershell
python src/train_models.py
```

This will:

- Load and preprocess the dataset.
- Train all four models.
- Evaluate each model.
- Generate `results/model_comparison.csv`.
- Generate `results/final_predictions.csv`.
- Save the final Random Forest model to `models/random_forest_model.joblib`.

### Generate Visualizations

```powershell
python src/visualize_results.py
```

This generates:

```text
results/model_comparison.png
results/confusion_matrix.png
```

The confusion matrix is calculated directly from the actual and predicted labels saved during model evaluation.

### Make a New Prediction

```powershell
python src/predict.py
```

The script asks for project information such as:

- Funding goal
- Campaign duration
- Launch year
- Launch month
- Project-name length
- Main category
- Category
- Country
- Currency

It then outputs:

- Predicted result: SUCCESSFUL or FAILED
- Probability of failure
- Probability of success

## 12. Example Prediction

Example input:

```text
Goal: 5000
Duration: 30 days
Launch year: 2017
Launch month: 6
Name length: 25
Main category: Games
Category: Tabletop Games
Country: US
Currency: USD
```

Example output from the trained model:

```text
Prediction: SUCCESSFUL
Probability of Failure: 35.00%
Probability of Success: 65.00%
```

The probability values depend on the trained model and input characteristics.

## 13. Reproducibility

The project uses fixed random seeds where applicable:

```python
random_state=42
```

The train/test split is stratified to preserve the class distribution between the training and testing sets.

Model evaluation metrics and visualization data are generated programmatically rather than manually entered.

## 14. Limitations

The current model has several limitations:

- It uses mainly project metadata and does not perform detailed natural-language analysis of project descriptions.
- It does not use image or video information from campaigns.
- A random train/test split is used rather than a time-based split.
- The model's performance is limited by the selected features and algorithms.
- The prediction should be treated as a machine-learning estimate rather than a guarantee of campaign success.

## 15. Future Improvements

Possible extensions include:

- TF-IDF or transformer-based analysis of project descriptions.
- Sentiment analysis of project text.
- Word embeddings for project titles and descriptions.
- Feature engineering based on campaign timing and category trends.
- Hyperparameter tuning.
- Cross-validation.
- Time-based evaluation.
- Ensemble or stacking approaches.
- Incorporation of additional pre-launch project information.

## 16. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## 17. Conclusion

This project demonstrates an end-to-end machine learning workflow for Kickstarter project success prediction. The workflow includes data preprocessing, feature engineering, multiple classification models, automated evaluation, model comparison, visualization, model persistence, and interactive prediction.

Among the tested models, the 200-tree Random Forest provided the strongest F1 score and successful-class recall and was therefore selected as the final model.
