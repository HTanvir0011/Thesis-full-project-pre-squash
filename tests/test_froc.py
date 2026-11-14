import numpy as np
import pytest

from s4 import froc


def box(x, y, w=10, h=10):
    return [x, y, x + w, y + h]


def test_iou_known_values():
    assert froc.iou_matrix([box(0, 0)], [box(0, 0)])[0, 0] == pytest.approx(1.0)
    assert froc.iou_matrix([box(0, 0)], [box(5, 0)])[0, 0] == pytest.approx(50 / 150)
    assert froc.iou_matrix([box(0, 0)], [box(20, 20)])[0, 0] == 0


def test_perfect_detector_scores_one():
    gts = [(np.array([box(0, 0)]), np.array([0])), (np.array([box(30, 30)]), np.array([0]))]
    preds = [(np.array([box(0, 0)]), np.array([0.9]), np.array([0])), (np.array([box(30, 30)]), np.array([0.8]), np.array([0]))]
    assert froc.froc(preds, gts)["cpm"] == pytest.approx(1.0)
    assert froc.average_precision(preds, gts) == pytest.approx(1.0)


def test_false_positives_cost_sensitivity_at_low_fp_rate():
    gts = [(np.array([box(0, 0)]), np.array([0])), (np.zeros((0, 4)), np.array([]))]
    preds = [(np.array([box(0, 0)]), np.array([0.5]), np.array([0])),
             (np.array([box(50, 50)]), np.array([0.9]), np.array([0]))]  # high-scoring FP on the normal image
    r = froc.froc(preds, gts, rates=(0.25, 1))
    assert r["sens"][0.25] == 0.0 and r["sens"][1] == 1.0  # the FP ranks first: 0.5 FP/img before the TP is reached
    assert froc.average_precision(preds, gts) == pytest.approx(0.5)


def test_each_gt_matches_once_and_recall_flags():
    gts = [(np.array([box(0, 0)]), np.array([0]))]
    preds = [(np.array([box(0, 0), box(1, 1)]), np.array([0.9, 0.8]), np.array([0, 0]))]
    s, tp, _ = froc.match(preds, gts, 0.3)
    assert tp.tolist() == [True, False]
    assert froc.recall_at_fp(preds, gts, 2.0).tolist() == [True]
