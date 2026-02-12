from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DriftMetrics:
    tdr: float
    far: float
    mtd_norm: float
    m1: float
    true_detections: int
    false_alarms: int
    repeated_alarms: int
    detected_drifts: int
    total_drifts: int


def _sorted_unique(xs: Sequence[int]) -> List[int]:
    return sorted(set(int(x) for x in xs))


def compute_drift_metrics(
    drifts: Sequence[int],
    detections: Sequence[int],
    buffer_window: int,
) -> DriftMetrics:
    """
    drifts: ground-truth drift positions (indices)
    detections: detector alarm positions (indices)
    buffer_window: W

    True detection if detection in [drift, drift+W].
    Repeated alarms: alarms after the first true detection for a drift but still within that same window.
    False alarms: alarms not in any drift window and not repeated.
    """
    if buffer_window <= 0:
        raise ValueError("buffer_window must be > 0")

    drifts = _sorted_unique(drifts)
    detections = _sorted_unique(detections)

    total_drifts = len(drifts)
    if total_drifts == 0:
        # no drifts: convention (tdr=1?) usually undefined; choose 0 and treat all detections as false alarms
        false_alarms = len(detections)
        far = 1.0 if false_alarms > 0 else 0.0
        return DriftMetrics(
            tdr=0.0,
            far=far,
            mtd_norm=1.0,
            m1=0.0,
            true_detections=0,
            false_alarms=false_alarms,
            repeated_alarms=0,
            detected_drifts=0,
            total_drifts=0,
        )

    # For each drift, store its first matched detection (if any)
    drift_first_det: List[Optional[int]] = [None] * total_drifts

    # We'll classify detections by scanning in time order.
    true_detections = 0
    repeated_alarms = 0
    false_alarms = 0

    # pointer to current/next drift that could match this detection
    j = 0

    for d in detections:
        # advance j while detection is beyond the current drift window
        while j < total_drifts and d > drifts[j] + buffer_window:
            j += 1

        # Check if detection falls in some window.
        matched = False

        # Candidate drifts are around j (current) and maybe earlier if d < drifts[j] (but we enforce d>=drift).
        # With the while above, if j>0, windows before j are already expired for this detection.
        if j < total_drifts:
            drift = drifts[j]
            if drift <= d <= drift + buffer_window:
                matched = True
                if drift_first_det[j] is None:
                    drift_first_det[j] = d
                    true_detections += 1
                else:
                    repeated_alarms += 1

        if not matched:
            # Could it match a future drift? Not if we require d >= drift.
            # If you want symmetric window around drift, change logic here.
            false_alarms += 1

    detected_drifts = sum(1 for x in drift_first_det if x is not None)

    # TDR
    tdr = detected_drifts / total_drifts

    # FAR
    effective_detections = true_detections + false_alarms
    far = (false_alarms / effective_detections) if effective_detections > 0 else 0.0

    # MTD_norm
    if detected_drifts == 0:
        mtd_norm = 1.0
    else:
        delays = []
        for drift, det in zip(drifts, drift_first_det):
            if det is not None:
                delays.append(det - drift)
        mtd = sum(delays) / len(delays)
        mtd_norm = mtd / buffer_window
        # clamp just in case
        if mtd_norm < 0.0:
            mtd_norm = 0.0
        if mtd_norm > 1.0:
            mtd_norm = 1.0

    # M1 geometric mean
    a = tdr
    b = 1.0 - far
    c = 1.0 - mtd_norm
    # clamp negatives due to numeric issues
    a = max(0.0, min(1.0, a))
    b = max(0.0, min(1.0, b))
    c = max(0.0, min(1.0, c))
    m1 = (a * b * c) ** (1.0 / 3.0) if (a * b * c) > 0 else 0.0

    return DriftMetrics(
        tdr=tdr,
        far=far,
        mtd_norm=mtd_norm,
        m1=m1,
        true_detections=true_detections,
        false_alarms=false_alarms,
        repeated_alarms=repeated_alarms,
        detected_drifts=detected_drifts,
        total_drifts=total_drifts,
    )
