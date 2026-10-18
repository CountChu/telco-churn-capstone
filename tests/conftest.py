"""Shared test data: 60 fake customers in the raw IBM file format.

The tests never read the real CSV, so they run the same way on any machine
and in GitHub Actions.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    rng = np.random.default_rng(seed=7)
    n = 60
    yes_no = ["Yes", "No"]
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n)
    phone = rng.choice(yes_no, size=n, p=[0.9, 0.1])

    def internet_addon():
        return np.where(internet == "No", "No internet service", rng.choice(yes_no, size=n))

    tenure = rng.integers(1, 73, size=n)
    tenure[:2] = 0  # two brand-new customers, as in the real file ...
    monthly = rng.uniform(18, 120, size=n).round(2)
    total = (monthly * tenure).round(2).astype(str)
    total[:2] = " "  # ... whose TotalCharges is a blank string, as in the real file

    return pd.DataFrame({
        "customerID": [f"{i:04d}-TEST" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], size=n),
        "SeniorCitizen": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        "Partner": rng.choice(yes_no, size=n),
        "Dependents": rng.choice(yes_no, size=n),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(yes_no, size=n)),
        "InternetService": internet,
        "OnlineSecurity": internet_addon(),
        "OnlineBackup": internet_addon(),
        "DeviceProtection": internet_addon(),
        "TechSupport": internet_addon(),
        "StreamingTV": internet_addon(),
        "StreamingMovies": internet_addon(),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaperlessBilling": rng.choice(yes_no, size=n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            size=n,
        ),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(np.arange(n) % 3 == 0, "Yes", "No"),  # 20 churners, 40 stayers
    })
