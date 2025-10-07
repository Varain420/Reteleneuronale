
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from timed_decorator.simple_timed import timed
from typing import Tuple

# --- Data Setup ---
# Small sample arrays for correctness check
predicted = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
actual = np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0])

# Large arrays for performance testing
big_size = 500000
big_actual = np.repeat(actual, big_size)
big_predicted = np.repeat(predicted, big_size)


# --- Exercise 1: Confusion Matrix ---

@timed(use_seconds=True, show_args=False)
def tp_fp_fn_tn_sklearn(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    """Calculates confusion matrix values using Scikit-learn."""
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    return tp, fp, fn, tn


@timed(use_seconds=True, show_args=False)
def tp_fp_fn_tn_numpy(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:

    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tn = np.sum((gt == 0) & (pred == 0))

    return int(tp), int(fp), int(fn), int(tn)


# --- Exercise 2: Accuracy ---

@timed(use_seconds=True, show_args=False)
def accuracy_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    """Calculates accuracy using Scikit-learn."""
    return accuracy_score(gt, pred)


@timed(use_seconds=True, show_args=False)
def accuracy_numpy(gt: np.ndarray, pred: np.ndarray) -> float:

    # np.mean on a boolean array calculates the proportion of True values,
    # which is exactly the accuracy.
    return np.mean(gt == pred)


# --- Exercise 3: F1-Score ---

@timed(use_seconds=True, show_args=False)
def f1_score_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    """Calculates F1-score using Scikit-learn."""
    return f1_score(gt, pred)


@timed(use_seconds=True, show_args=False)
def f1_score_numpy(gt: np.ndarray, pred: np.ndarray) -> float:

    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))

    # Calculate precision, handling division by zero
    precision_denominator = tp + fp
    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = tp / precision_denominator

    # Calculate recall, handling division by zero
    recall_denominator = tp + fn
    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = tp / recall_denominator

    # Calculate F1-score, handling division by zero
    f1_denominator = precision + recall
    if f1_denominator == 0:
        return 0.0
    else:
        return 2 * (precision * recall) / f1_denominator


# --- Main execution block for testing and verification ---

if __name__ == "__main__":
    print("--- Correctness Check on Small Arrays ---")

    # Exercise 1
    print("\n[Exercise 1: Confusion Matrix]")
    sklearn_result_1 = tp_fp_fn_tn_sklearn(actual, predicted)
    numpy_result_1 = tp_fp_fn_tn_numpy(actual, predicted)
    assert sklearn_result_1 == numpy_result_1
    print(f"Results match: {numpy_result_1}")

    # Exercise 2
    print("\n[Exercise 2: Accuracy]")
    sklearn_result_2 = accuracy_sklearn(actual, predicted)
    numpy_result_2 = accuracy_numpy(actual, predicted)
    assert np.isclose(sklearn_result_2, numpy_result_2)
    print(f"Results match: {numpy_result_2:.4f}")

    # Exercise 3
    print("\n[Exercise 3: F1-Score]")
    sklearn_result_3 = f1_score_sklearn(actual, predicted)
    numpy_result_3 = f1_score_numpy(actual, predicted)
    assert np.isclose(sklearn_result_3, numpy_result_3)
    print(f"Results match: {numpy_result_3:.4f}")

    print("\n" + "=" * 40)
    print("--- Performance Test on Large Arrays ---")

    # Exercise 1
    print("\n[Exercise 1: Confusion Matrix]")
    rez_1_sklearn = tp_fp_fn_tn_sklearn(big_actual, big_predicted)
    rez_1_numpy = tp_fp_fn_tn_numpy(big_actual, big_predicted)
    assert rez_1_sklearn == rez_1_numpy

    # Exercise 2
    print("\n[Exercise 2: Accuracy]")
    rez_2_sklearn = accuracy_sklearn(big_actual, big_predicted)
    rez_2_numpy = accuracy_numpy(big_actual, big_predicted)
    assert np.isclose(rez_2_sklearn, rez_2_numpy)

    # Exercise 3
    print("\n[Exercise 3: F1-Score]")
    rez_3_sklearn = f1_score_sklearn(big_actual, big_predicted)
    rez_3_numpy = f1_score_numpy(big_actual, big_predicted)
    assert np.isclose(rez_3_sklearn, rez_3_numpy)

    print("\nAll tests passed!")