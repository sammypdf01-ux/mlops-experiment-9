
import pandas as pd
from train import train_model

def test_model_training(tmp_path):
    data = pd.DataFrame({
        "Quantity": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "UnitPrice": [10, 12, 15, 20, 22, 25, 30, 35, 40, 45]
    })

    csv_path = tmp_path / "retail.csv"
    model_path = tmp_path / "model.pkl"

    data.to_csv(csv_path, index=False)

    results = train_model(str(csv_path), str(model_path))

    assert model_path.exists()
    assert results["MAE"] >= 0
    assert results["MSE"] >= 0
