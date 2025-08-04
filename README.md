# car_cat.py

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load data
df = pd.read_csv("car_price_dataset.csv")

# Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Categorical features
cat_features = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Doors', 'Owner_Count']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
reg_model = CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, cat_features=cat_features, verbose=0)
reg_model.fit(X_train, y_train)

# Predict and evaluate
reg_preds = reg_model.predict(X_test)
mse = mean_squared_error(y_test, reg_preds)
print(f"Regression MSE: {mse:.2f}")





#--------------------------------------------------------------------------


# Convert Price into categories
def categorize_price(price):
    if price < 10000:
        return 'Affordable'
    elif price < 30000:
        return 'Mid'
    else:
        return 'Expensive'

df["Price_Category"] = df["Price"].apply(categorize_price)




from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

# Features and target
X = df.drop(['Price', 'Price_Category'], axis=1)
y = df["Price_Category"]

# Categorical features
cat_features = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Doors', 'Owner_Count']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
clf_model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, cat_features=cat_features, verbose=0)
clf_model.fit(X_train, y_train)

# Predict and evaluate
clf_preds = clf_model.predict(X_test)
print(classification_report(y_test, clf_preds))
