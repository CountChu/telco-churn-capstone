"""Build, train and use the churn classifiers.

Each model is a scikit-learn Pipeline: the preprocessor (scale + one-hot
encode) followed by the classifier. Whatever the preprocessor learns from the
training rows is applied unchanged to every row we predict on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.preprocessing import build_preprocessor

MODEL_NAMES = ("logistic_regression", "decision_tree")


def build_model(model_name: str = config.PRIMARY_MODEL) -> Pipeline:
    """An untrained preprocessing + classifier pipeline."""
    if model_name == "logistic_regression":
        classifier = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
    elif model_name == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=config.RANDOM_STATE, **config.DECISION_TREE_PARAMS)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(MODEL_NAMES)}")
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_name: str = config.PRIMARY_MODEL) -> Pipeline:
    """Fit the pipeline on the training rows only."""
    model = build_model(model_name)
    model.fit(X_train, y_train)
    return model


def predict(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class (1 = churn) and churn probability for every row of X."""
    churn_column = list(model.classes_).index(1)
    return model.predict(X), model.predict_proba(X)[:, churn_column]


def feature_weights(model: Pipeline) -> pd.DataFrame:
    """What the fitted model learned per encoded feature, strongest first.

    Logistic Regression: coefficient (> 0 pushes towards churn, < 0 away from it).
    Decision Tree: importance (how much the feature is used to split customers).
    """
    names = model.named_steps["preprocessor"].get_feature_names_out()
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        table = pd.DataFrame({"feature": names, "coefficient": classifier.coef_[0]})
        order = table["coefficient"].abs().sort_values(ascending=False).index
    else:
        table = pd.DataFrame({"feature": names, "importance": classifier.feature_importances_})
        order = table["importance"].sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)
