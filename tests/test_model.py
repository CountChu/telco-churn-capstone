import numpy as np
import pytest

from src.evaluation import PREDICTION_COLUMNS, build_predictions_frame, calculate_metrics
from src.model import MODEL_NAMES, build_model, predict, train_model
from src.preprocessing import clean_data, split_features_target, split_train_test


@pytest.fixture
def split_sample(raw_sample):
    clean, _ = clean_data(raw_sample)
    return split_train_test(*split_features_target(clean))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_trains_on_a_small_sample_and_predicts_valid_classes(split_sample, model_name):
    X_train, X_test, y_train, _, _, _ = split_sample

    fitted = train_model(X_train, y_train, model_name)
    predicted, probability = predict(fitted, X_test)

    assert len(predicted) == len(X_test)
    assert set(np.unique(predicted)) <= {0, 1}
    assert np.all((probability >= 0) & (probability <= 1))


def test_an_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("random_forest")


def test_prediction_output_contains_the_required_columns(split_sample):
    X_train, X_test, y_train, y_test, _, ids_test = split_sample
    fitted = train_model(X_train, y_train)
    predicted, probability = predict(fitted, X_test)

    output = build_predictions_frame(ids_test, y_test, predicted, probability)

    assert list(output.columns) == PREDICTION_COLUMNS
    assert len(output) == len(X_test)
    assert set(output["customerID"]) == set(ids_test)
    assert output["churn_probability"].is_monotonic_decreasing  # highest risk first


def test_metrics_match_a_hand_calculation():
    # 3 true positives, 1 false negative, 2 false positives, 2 true negatives
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 1, 0, 0]

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(5 / 8)
    assert metrics["precision"] == pytest.approx(3 / 5)
    assert metrics["recall"] == pytest.approx(3 / 4)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["false_negatives"] == 1
