import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

encoder = OneHotEncoder()
model = LogisticRegression()

df = pd.read_csv("./data/customers.csv")

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

categorical_columns = X_train.select_dtypes(include="str").columns

numerical_columns = X_train.select_dtypes(include=("float", "int")).columns


preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", encoder, categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print(classification_report(y_test, predictions))

joblib.dump(pipeline, "models/model.pkl")
print("Model saved to models/model.pkl")



