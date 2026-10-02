# Customer Churn Predictor

A beginner-friendly machine learning project that predicts whether a customer is likely to churn using **Logistic Regression**.

The project demonstrates a complete classification workflow, including data preparation, categorical feature encoding, model training, evaluation, prediction probabilities, model persistence, and interactive predictions.

> **Note:** This project uses a small synthetic dataset created for learning purposes. The model's evaluation results should not be considered representative of real-world customer churn prediction.

---

## Features

* Load and inspect customer data using Pandas
* Separate features and target
* Split data into training and testing sets
* Handle categorical features using One-Hot Encoding
* Use `ColumnTransformer` for preprocessing
* Build a Scikit-learn `Pipeline`
* Train a Logistic Regression classification model
* Evaluate the model using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
  * Classification Report
* Generate churn prediction probabilities
* Save and load the complete ML pipeline using Joblib
* Make predictions for new customers through an interactive Python script
* Validate user input

---

## Machine Learning Workflow

```text
Customer Data
      ↓
Pandas DataFrame
      ↓
Feature / Target Separation
      ↓
Train / Test Split
      ↓
Categorical Feature Detection
      ↓
One-Hot Encoding
      ↓
ColumnTransformer
      ↓
Logistic Regression
      ↓
Pipeline
      ↓
Model Training
      ↓
Predictions
      ↓
Model Evaluation
      ↓
Save Trained Pipeline
```

---

## Dataset

The dataset contains 24 synthetic customer records.

### Features

| Feature            | Description                                             | Type        |
| ------------------ | ------------------------------------------------------- | ----------- |
| `tenure`           | Number of months the customer has been with the company | Numeric     |
| `monthly_charges`  | Customer's monthly charges                              | Numeric     |
| `support_calls`    | Number of support calls made by the customer            | Numeric     |
| `contract_type`    | Customer's contract type                                | Categorical |
| `internet_service` | Customer's internet service type                        | Categorical |
| `churn`            | Whether the customer left the service                   | Target      |

### Target

```text
0 = Customer stayed
1 = Customer churned
```

---

## Categorical Feature Encoding

Machine learning models cannot directly work with categorical strings such as:

```text
Month-to-month
One year
Two year
```

and:

```text
Fiber
DSL
```

The project uses `OneHotEncoder` to convert these categories into numerical features.

`ColumnTransformer` is then used to apply:

* One-Hot Encoding to categorical features
* Pass-through processing to numerical features

---

## Model

The project uses **Logistic Regression** for binary classification.

The preprocessing and model are combined into a single Scikit-learn `Pipeline`:

```text
Raw Customer Data
       ↓
ColumnTransformer
       ↓
OneHotEncoder
       ↓
Logistic Regression
       ↓
Prediction
```

This allows the same preprocessing steps to be automatically applied when making predictions on new customers.

---

## Model Evaluation

The model is evaluated using the test dataset.

Example results from the current dataset:

```text
Accuracy: 0.80

              precision    recall  f1-score   support

           0       1.00      0.50      0.67         2
           1       0.75      1.00      0.86         3

    accuracy                           0.80         5
   macro avg       0.88      0.75      0.76         5
weighted avg       0.85      0.80      0.78         5
```

Because the test set contains only 5 customers, these metrics are highly sensitive to individual predictions and should not be interpreted as evidence of real-world model performance.

---

## Prediction Probabilities

In addition to predicting `0` or `1`, the model can provide probabilities.

For example:

```text
Churn probability: 73.42%
```

The probability represents the model's estimated probability that the customer belongs to the churn (`1`) class.

---

## Project Structure

```text
customer-churn-predictor/
│
├── data/
│   └── customers.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── .env
```

---

## Installation

Clone the repository and navigate into the project:

```bash
git clone <your-repository-url>
cd customer-churn-predictor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training the Model

From the project root, run:

```bash
python src/train.py
```

This will:

1. Load the customer dataset
2. Split the data into training and testing sets
3. Encode categorical features
4. Train the Logistic Regression model
5. Evaluate the model
6. Save the trained pipeline to:

```text
models/model.pkl
```

---

## Making Predictions

After training the model, run:

```bash
python src/predict.py
```

The program will ask for information about a customer:

```text
Enter tenure:
Enter monthly charges:
Enter number of support calls:
Enter contract type:
Enter internet service:
```

It will then display a prediction such as:

```text
Customer is likely to churn.
Churn probability: 73.42%
```

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## What I Practiced

This project was built as a hands-on exercise to practice machine learning and Python syntax.

Topics practiced:

* Pandas DataFrames
* Feature and target selection
* Train/test splitting
* Classification
* Logistic Regression
* Categorical data
* One-Hot Encoding
* `ColumnTransformer`
* Scikit-learn Pipelines
* Model training and prediction
* Prediction probabilities
* Confusion matrices
* Accuracy
* Precision
* Recall
* F1-score
* Classification reports
* Model persistence with Joblib
* Interactive Python input
* Input validation

---

## Limitations

This project is intended for learning and experimentation.

The dataset is:

* Small
* Synthetic
* Not representative of a real customer population

Therefore, the model's performance metrics should not be used to judge how the model would perform on real customer data.

A real-world churn prediction system would require a much larger and representative dataset, appropriate feature engineering, validation, and additional model evaluation.
