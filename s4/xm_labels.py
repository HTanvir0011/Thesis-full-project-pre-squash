"""Label spaces of the cross-modality study (DESIGN_v4.md section 2), fixed before any v4 run.

The Qure.ai ct2xr projections carry six findings from their CT-RATE scans (results/qp_meta.csv). Each maps to NIH ChestX-ray14 and
VinBigData classes; a finding with several classes is positive when any of them is, and scores the largest probability.
"""
import numpy as np

from .prep import NIH

SHARED = ("lung_opacity", "pleural_effusion", "atelectasis", "cardiomegaly", "consolidation", "lung_lesion")
TO_NIH = {"lung_opacity": ("Infiltration",), "pleural_effusion": ("Effusion",), "atelectasis": ("Atelectasis",),
          "cardiomegaly": ("Cardiomegaly",), "consolidation": ("Consolidation",), "lung_lesion": ("Mass", "Nodule")}
TO_VIN = {"lung_opacity": (7,), "pleural_effusion": (10,), "atelectasis": (1,), "cardiomegaly": (3,), "consolidation": (4,),
          "lung_lesion": (8,)}  # VinBigData class_id order (tools/hybrid_stats.py VIN)
EXACT_NIH = SHARED[1:]  # lung_opacity -> Infiltration is approximate: reported, not tested (SAP_v4.md H11)


def columns(dataset):
    """Class indices of `dataset` ('nih' | 'vin') for each shared finding, in SHARED order."""
    return [[NIH.index(c) for c in TO_NIH[s]] if dataset == "nih" else list(TO_VIN[s]) for s in SHARED]


def to_shared(y, dataset):
    """[N, 14] labels or probabilities of NIH or VinBigData -> [N, 6] in SHARED order (largest over the mapped classes)."""
    y = np.asarray(y)
    return np.stack([y[:, c].max(1) for c in columns(dataset)], 1)


def from_meta(rows):
    """qp_meta.csv rows (dicts) -> [N, 6] float32 labels in SHARED order."""
    return np.array([[float(r[s]) for s in SHARED] for r in rows], np.float32)
