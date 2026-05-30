# Maternal Health Risk Prediction - PRML course project

## Overview
This project predicts maternal health risk levels - low, mid, and high - from a small set of clinical measurements. The goal is not just to build a classifier, but to understand which signals matter, compare several models fairly, and see how far we can push performance on a structured healthcare dataset.

The notebook now follows a clean end-to-end flow: exploratory data analysis, baseline model comparison, model tuning, a stronger SMOTE-based experiment, and dimensionality reduction with PCA and LDA.

## Dataset
The dataset comes from the UCI Maternal Health Risk repository and contains about 1,014 records.

Features used in the notebook:
- Age
- Systolic blood pressure
- Diastolic blood pressure
- Blood sugar
- Body temperature
- Heart rate

Target:
- RiskLevel with three classes: low risk, mid risk, and high risk

## Notebook workflow

### 1. Data loading and overview
The notebook begins by loading the CSV file, checking the dataset shape, verifying the column names, and reviewing missing values, duplicates, and descriptive statistics.

### 2. Exploratory data analysis
The EDA section looks at:
- class balance in the target variable
- distributions of the numerical features
- correlations between features
- feature behavior across the three risk classes using boxplots and grouped summaries

This helps explain what the model is learning before training starts.

### 3. Baseline model comparison
Several models are trained and compared on train, validation, and test splits:
- K-Nearest Neighbors
- Support Vector Machine
- Random Forest
- XGBoost
- Gradient Boosting
- Decision Tree

Each model is tuned with GridSearchCV where appropriate so the comparison is fair.

### 4. Validation-based model selection
The notebook uses a train / validation / test split instead of only one test set. That makes the selection process cleaner because the best model is chosen using validation accuracy, not test accuracy.

### 5. Stronger SMOTE-based experiment
To improve performance, the notebook also applies SMOTE on the training data only. This balances the classes and helps the models learn minority cases better.

### 6. Reduced-feature experiments
The notebook compares PCA and LDA after scaling and SMOTE. These experiments check whether dimensionality reduction can keep accuracy high while making the model more compact.

## Main results
The notebook shows two useful takeaways:
- The tuned baseline models already perform reasonably well.
- The SMOTE-based ensemble experiments produce the strongest overall results, especially for Gradient Boosting and Random Forest.

PCA with 4 components is the best compact representation in the reduced-feature experiments.

## Requirements
Install the required Python packages with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## How to run
1. Open the notebook file.
2. Make sure the dataset CSV is in the same folder.
3. Run the cells in order from top to bottom.

## Project structure
```
├── Maternal Health Risk Data Set.csv
├── maternal_health_risk_analysis.ipynb
└── README.md
```

## Team contributions
Everyone on the team contributed to model building, tuning, and evaluation. The work was split so each person owned specific models and also helped with the shared analysis flow.

- Iftikhar: data loading, cleaning, EDA plots, and baseline model checks such as KNN and SVM
- Gurtej: feature analysis, correlation study, and tree-based models including Decision Tree and Random Forest
- Deepansh: tuned ensemble models, especially Gradient Boosting and XGBoost, plus validation comparison tables
- Devang: SMOTE experiments, PCA and LDA comparisons, confusion matrices, and final performance analysis

In practice, everyone also reviewed results together and helped refine the final notebook so the model comparison stayed consistent.

## Conclusion
This project presents a complete machine learning workflow for maternal health risk prediction. The notebook combines EDA, model comparison, tuning, class balancing, and dimensionality reduction in a way that is easy to follow and suitable for submission.

The strongest practical results come from the SMOTE-based ensemble models, while PCA offers a useful lower-dimensional alternative with only a modest accuracy trade-off.

## Future work
- Try more domain-specific medical features
- Test calibrated probabilities for clinical use
- Explore stacking or soft-voting ensembles
- Evaluate the model on a larger and more diverse dataset
