
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_model(data_path, model_path="ml_model.pkl"):
    df = pd.read_csv(data_path, encoding="latin1")

    mapping = {}
    for col in df.columns:
        name = col.strip().lower().replace(" ", "").replace("_", "")
        if name in ["quantity", "unitssold", "sold"]:
            mapping[col] = "Quantity"
        elif name in ["unitprice", "price"]:
            mapping[col] = "UnitPrice"

    df = df.rename(columns=mapping)

    if not {"Quantity", "UnitPrice"}.issubset(df.columns):
        raise ValueError("Required columns are missing")

    df = df.dropna(subset=["Quantity", "UnitPrice"])
    df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]

    if len(df) < 2:
        raise ValueError("Not enough valid records")

    X = df[["Quantity", "UnitPrice"]]
    y = df["Quantity"] * df["UnitPrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    joblib.dump(model, model_path)

    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions)
    }
