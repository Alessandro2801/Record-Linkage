import numpy as np
import pandas as pd

from src.evaluation.compare import build_global_predictions, compute_confusion_metrics


def test_build_global_predictions_and_confusion_metrics():
    gt_test = pd.DataFrame(
        {
            "id_A": ["1", "2", "3", "4"],
            "id_B": ["10", "20", "30", "40"],
            "label": [1, 0, 1, 0],
        }
    )

    candidates = pd.DataFrame(
        {
            "id_A": ["1", "20"],
            "id_B": ["10", "2"],
            "label": [1, 0],
        }
    )
    candidate_pred = np.array([1, 1], dtype=int)

    y_pred_global = build_global_predictions(gt_test, candidates, candidate_pred)
    np.testing.assert_array_equal(y_pred_global, np.array([1, 1, 0, 0], dtype=int))

    y_true = gt_test["label"].to_numpy()
    metrics = compute_confusion_metrics(y_true, y_pred_global)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
