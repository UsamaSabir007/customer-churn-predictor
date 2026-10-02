import joblib
import pandas as pd

pipeline = joblib.load("models/model.pkl")

def get_float_input(prompt):
    while True:
        try:
            user_input = float(input(prompt))
            if (user_input < 0):
                print("Please enter a non-negative number.")
            else:
                return user_input    
        except ValueError:
            print("Please enter a valid number.")

def get_int_input(prompt):
    while True:
        try:
            user_input = int(input(prompt))
            if (user_input < 0):
                print("Please enter a non-negative number.")
            else:
                return user_input    
        except ValueError:
            print("Please enter a valid integer.")

def get_contract_type(prompt):
    while True:
        user_input = input(prompt).strip().lower()

        if user_input == "one year":
            return "One year"
        elif user_input == "two year":
            return "Two year"
        elif user_input == "month-to-month":
            return "Month-to-month"
        else:
            print("Error: Please enter one of: 'one year', 'two year', 'month-to-month'.")

def get_internet_service(prompt):
    while True:
        user_input = input(prompt).strip().lower()

        if user_input == "fiber":
            return "Fiber"
        elif user_input == "dsl":
            return "DSL"
        else:
            print("Error: Please enter either 'fiber' or 'dsl'.")
                   

tenure = get_int_input("Enter tenure: ")
monthly_charges = get_float_input("Enter monthly charges: ")
support_calls = get_int_input("Enter number of support calls: ")
contract_type = get_contract_type("Enter contract type: ")
internet_service = get_internet_service("Enter internet service: ")

customer_churn = pd.DataFrame({
    "tenure": [tenure],
    "monthly_charges": [monthly_charges],
    "support_calls": [support_calls],
    "contract_type": [contract_type],
    "internet_service": [internet_service]
})

prediction = pipeline.predict(customer_churn)

if prediction[0] == 0:
    print("Customer is likely to stay.")
else:
    print("Customer is likely to churn.")

probability = pipeline.predict_proba(customer_churn)

print(f"Churn probability: {probability[0][1] * 100:.2f}%")
