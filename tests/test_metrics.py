"""Unit tests for symformer_tb/metrics.py -- numpy only, no torch, no GPU, no data.

These are hand-computable on purpose. The metric code decides what every result in the project
means, so every assertion here is a number worked out by hand rather than a snapshot of whatever the
implementation happened to produce.

Run:  python -m pytest tests/test_metrics.py    (or)   python tests/test_metrics.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symformer_tb.metrics import (  # noqa: E402
    best_operating_point, box_iou, classification_report, confusion_counts, f1_from_pr, fitness,
    gather_tp, match_image, pr_curve, roc_auc, xywh_to_xyxy,
)


# ---------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------
def test_xywh_to_xyxy():
    got = xywh_to_xyxy([[10, 20, 30, 40]])
    assert np.allclose(got, [[10, 20, 40, 60]]), got


def test_box_iou_identical_is_one():
    a = np.array([[0, 0, 10, 10]])
    assert np.isclose(box_iou(a, a)[0, 0], 1.0)


def test_box_iou_disjoint_is_zero():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[20, 20, 30, 30]])
    assert np.isclose(box_iou(a, b)[0, 0], 0.0)


def test_box_iou_hand_computed():
    # 10x10 boxes offset by 5 in x: intersection 5*10 = 50, union 100+100-50 = 150 -> 1/3
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 0, 15, 10]])
    assert np.isclose(box_iou(a, b)[0, 0], 50 / 150), box_iou(a, b)


def test_box_iou_empty_inputs():
    assert box_iou(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)
    assert box_iou(np.array([[0, 0, 1, 1]]), np.zeros((0, 4))).shape == (1, 0)


# ---------------------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------------------
def test_match_perfect_prediction():
    gt = np.array([[0, 0, 10, 10]])
    tp = match_image(gt.copy(), np.array([0.9]), gt)
    assert tp.tolist() == [1.0]


def test_match_below_threshold_is_false_positive():
    gt = np.array([[0, 0, 10, 10]])
    pred = np.array([[8, 0, 18, 10]])          # IoU = 20/180 = 0.11 < 0.5
    tp = match_image(pred, np.array([0.9]), gt)
    assert tp.tolist() == [0.0]


def test_duplicate_predictions_only_one_counts():
    """Two good boxes on one GT: the higher-scoring one is the TP, the other a FP.

    This is the property that stops a model from farming AP by predicting the same box repeatedly.
    """
    gt = np.array([[0, 0, 10, 10]])
    preds = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    tp = match_image(preds, np.array([0.6, 0.9]), gt)
    assert tp.tolist() == [0.0, 1.0], tp          # index 1 has the higher score, so it wins


def test_greedy_assignment_prefers_higher_score_first():
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    preds = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    tp = match_image(preds, np.array([0.5, 0.9]), gt)
    assert tp.tolist() == [1.0, 1.0]


def test_no_gt_means_every_prediction_is_a_false_positive():
    """The all-images evaluation mode depends on this: negatives have no boxes, so anything the
    detector draws on a healthy chest must be counted against it."""
    tp = match_image(np.array([[0, 0, 10, 10]]), np.array([0.9]), np.zeros((0, 4)))
    assert tp.tolist() == [0.0]


def test_gather_tp_counts_gt_from_empty_images():
    results = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 2, "bbox": [0, 0, 10, 10], "score": 0.8},   # image 2 is a negative
    ]
    gt = {1: np.array([[0.0, 0.0, 10.0, 10.0]]), 2: np.zeros((0, 4))}
    tp, conf, n_gt = gather_tp(results, gt)
    assert n_gt == 1
    assert sorted(tp.tolist()) == [0.0, 1.0]
    assert sorted(conf.tolist()) == [0.8, 0.9]


# ---------------------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------------------
def test_pr_curve_hand_computed():
    # 4 predictions sorted by score: TP, FP, TP, FP with 3 GT total
    tp = np.array([1.0, 0.0, 1.0, 0.0])
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    p, r, c = pr_curve(tp, conf, n_gt=3)
    assert np.allclose(p, [1 / 1, 1 / 2, 2 / 3, 2 / 4]), p
    assert np.allclose(r, [1 / 3, 1 / 3, 2 / 3, 2 / 3]), r
    assert np.allclose(c, conf)


def test_f1_hand_computed():
    p = np.array([1.0, 0.5])
    r = np.array([0.5, 0.5])
    f1 = f1_from_pr(p, r)
    assert np.isclose(f1[0], 2 * 1.0 * 0.5 / 1.5), f1
    assert np.isclose(f1[1], 0.5), f1


def test_best_operating_point_picks_max_f1():
    tp = np.array([1.0, 1.0, 0.0])
    conf = np.array([0.9, 0.8, 0.7])
    point = best_operating_point(tp, conf, n_gt=2)
    # after 2 predictions: P = 1.0, R = 1.0, F1 = 1.0 -- the maximum
    assert np.isclose(point["f1"], 1.0), point
    assert np.isclose(point["precision"], 1.0), point
    assert np.isclose(point["recall"], 1.0), point
    assert np.isclose(point["conf"], 0.8), point


def test_empty_predictions_do_not_crash():
    point = best_operating_point(np.zeros(0), np.zeros(0), n_gt=5)
    assert point == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "conf": 0.0}


def test_confusion_counts():
    tp = np.array([1.0, 0.0, 1.0])
    conf = np.array([0.9, 0.8, 0.4])
    # threshold 0.5 keeps the first two: 1 TP, 1 FP; 3 GT total -> 2 missed
    assert confusion_counts(tp, conf, n_gt=3, conf_thr=0.5) == (1, 1, 2)


# ---------------------------------------------------------------------------------------
# fitness
# ---------------------------------------------------------------------------------------
def test_fitness_modes():
    assert np.isclose(fitness(ap=30.0, ap50=80.0, mode="ap50"), 80.0)
    assert np.isclose(fitness(ap=30.0, ap50=80.0, mode="ap"), 30.0)
    assert np.isclose(fitness(ap=30.0, ap50=80.0, mode="ultralytics"), 0.1 * 80 + 0.9 * 30)


def test_fitness_weights_the_strict_metric():
    """A model that only localises loosely must not win best.pt on AP50 alone."""
    loose = fitness(ap=25.0, ap50=85.0)
    tight = fitness(ap=35.0, ap50=80.0)
    assert tight > loose, (tight, loose)


# ---------------------------------------------------------------------------------------
# classification (stage 2)
# ---------------------------------------------------------------------------------------
def test_roc_auc_perfect_separation():
    assert np.isclose(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)


def test_roc_auc_inverted_is_zero():
    assert np.isclose(roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)


def test_roc_auc_all_ties_is_half():
    assert np.isclose(roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)


def test_roc_auc_single_class_is_nan():
    assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_classification_report_hand_computed():
    # classes: 0 healthy, 1 sick_non_tb, 2 tb
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    scores = [0.1, 0.2, 0.1, 0.3, 0.9, 0.4]
    rep = classification_report(y_true, y_pred, scores, tb_index=2)

    assert np.isclose(rep["accuracy"], 4 / 6 * 100), rep["accuracy"]
    # TB binarisation: 2 true TB, 1 predicted correctly -> sensitivity 1/2
    assert np.isclose(rep["sensitivity"], 50.0), rep
    # 4 non-TB, none predicted TB -> specificity 4/4
    assert np.isclose(rep["specificity"], 100.0), rep
    assert rep["confusion"] == {"tp": 1, "tn": 4, "fp": 0, "fn": 1}, rep["confusion"]


def test_classification_report_specificity_is_the_interesting_column():
    """A model that shouts TB at everything has perfect sensitivity and useless specificity.

    This is exactly the failure mode the paper reports for the baselines in Table 3, and the reason
    stage 2 exists -- so the metric must actually expose it.
    """
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [2, 2, 2, 2, 2, 2]
    rep = classification_report(y_true, y_pred, [0.9] * 6, tb_index=2)
    assert np.isclose(rep["sensitivity"], 100.0)
    assert np.isclose(rep["specificity"], 0.0)


# ---------------------------------------------------------------------------------------
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
