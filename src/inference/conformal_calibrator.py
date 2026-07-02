"""
conformal_calibrator.py — BioGemma-Guard Conformal Prediction Safety Layer
==========================================================================

Implements **Split / Inductive Conformal Prediction** for classification
to provide finite-sample, distribution-free coverage guarantees on medical
diagnoses.

Mathematical Foundation
-----------------------
Given a calibration set of *n* held-out examples with softmax probability
vectors and true labels:

1. **Non-conformity score** for sample *i*:

   .. math::

       s_i = 1 - \\hat{\\pi}_{y_i}

   where :math:`\\hat{\\pi}_{y_i}` is the model's predicted probability
   for the *true* class :math:`y_i`.  A high score means the model was
   uncertain about the correct answer.

2. **Critical quantile** :math:`\\hat{q}`:

   .. math::

       q_{\\text{level}} = \\frac{\\lceil (n+1)(1-\\alpha) \\rceil}{n}

   :math:`\\hat{q}` is the :math:`q_{\\text{level}}`-quantile of the
   calibration scores.  This inflated quantile ensures the marginal
   coverage guarantee:

   .. math::

       \\mathbb{P}\\bigl(y_{\\text{test}} \\in C(x_{\\text{test}})\\bigr)
       \\;\\geq\\; 1 - \\alpha

3. **Prediction set** for a new test input:

   .. math::

       C(x) = \\bigl\\{\\, k : 1 - \\hat{\\pi}_k \\leq \\hat{q} \\,\\bigr\\}
            = \\bigl\\{\\, k : \\hat{\\pi}_k \\geq 1 - \\hat{q} \\,\\bigr\\}

4. **Abstention rule** — the core safety mechanism:

   * ``|C(x)| == 0`` → model cannot even pass the threshold → **abstain**
   * ``|C(x)| == 1`` → exactly one diagnosis survives → **diagnose**
   * ``|C(x)| >  1`` → statistically ambiguous → **abstain**

Integration
-----------
This module is designed as an inference-time wrapper.  After the GRPO-
trained model produces softmax probabilities, the calibrator decides
whether to trust the top prediction or trigger abstention.

>>> from src.inference.conformal_calibrator import MedicalConformalCalibrator
>>>
>>> calibrator = MedicalConformalCalibrator(alpha=0.05)
>>> calibrator.fit(calib_probs, calib_labels)
>>> if calibrator.should_abstain(test_probs):
...     response = "I am uncertain. Insufficient data to provide a safe diagnosis."
... else:
...     response = class_names[test_probs.argmax()]

Author : BioGemma-Guard Team
License: Apache-2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ──────────────────────────────────────────────────────────────────────
# 1.  CALIBRATION RESULT DATACLASS
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Immutable snapshot of a completed calibration.

    Attributes
    ----------
    alpha : float
        Requested error rate.
    q_hat : float
        Critical quantile (non-conformity score threshold).
    q_level : float
        Adjusted quantile level used for computing ``q_hat``.
    n_calibration : int
        Number of calibration samples used.
    scores_mean : float
        Mean of calibration non-conformity scores.
    scores_std : float
        Standard deviation of calibration non-conformity scores.
    """
    alpha: float
    q_hat: float
    q_level: float
    n_calibration: int
    scores_mean: float
    scores_std: float

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CalibrationResult(α={self.alpha}, q̂={self.q_hat:.4f}, "
            f"q_level={self.q_level:.4f}, n={self.n_calibration}, "
            f"scores μ={self.scores_mean:.4f} σ={self.scores_std:.4f})"
        )


# ──────────────────────────────────────────────────────────────────────
# 2.  CORE CALIBRATOR CLASS
# ──────────────────────────────────────────────────────────────────────

class MedicalConformalCalibrator:
    """Split / Inductive Conformal Prediction calibrator for medical
    classification with a built-in abstention mechanism.

    Parameters
    ----------
    alpha : float
        Desired maximum error rate.  For example, ``alpha=0.05``
        targets ≥95 % marginal coverage.  Must be in ``(0, 1)``.

    Raises
    ------
    ValueError
        If *alpha* is outside the open interval ``(0, 1)``.

    Examples
    --------
    >>> cal = MedicalConformalCalibrator(alpha=0.05)
    >>> cal.fit(calib_probs, calib_labels)
    >>> pred_set = cal.predict_set(test_probs)   # boolean mask
    >>> cal.should_abstain(test_probs)            # True / False
    """

    # ── construction ────────────────────────────────────────────────

    def __init__(self, alpha: float = 0.05) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"alpha must be in (0, 1), got {alpha!r}"
            )
        self._alpha: float = alpha
        self._q_hat: float | None = None
        self._calibration: CalibrationResult | None = None
        self._scores: NDArray[np.floating] | None = None

    # ── public properties ───────────────────────────────────────────

    @property
    def alpha(self) -> float:
        """Requested error rate."""
        return self._alpha

    @property
    def q_hat(self) -> float:
        """Critical quantile threshold (set after :meth:`fit`)."""
        if self._q_hat is None:
            raise RuntimeError(
                "Calibrator has not been fitted yet.  Call .fit() first."
            )
        return self._q_hat

    @property
    def calibration(self) -> CalibrationResult:
        """Full calibration result snapshot."""
        if self._calibration is None:
            raise RuntimeError(
                "Calibrator has not been fitted yet.  Call .fit() first."
            )
        return self._calibration

    # ── fit ──────────────────────────────────────────────────────────

    def fit(
        self,
        calib_probs: ArrayLike,
        calib_labels: ArrayLike,
    ) -> CalibrationResult:
        """Calibrate the conformal threshold on held-out data.

        Parameters
        ----------
        calib_probs : array-like, shape ``(n, K)``
            Softmax probability matrix from the model on the
            calibration split.  Each row sums to ≈1.
        calib_labels : array-like, shape ``(n,)``
            Integer ground-truth class indices for each calibration
            sample.

        Returns
        -------
        CalibrationResult
            Frozen dataclass with all calibration diagnostics.

        Raises
        ------
        ValueError
            If inputs have incompatible shapes or labels are out of
            range.

        Notes
        -----
        **Quantile level formula** (finite-sample correction):

        .. math::

            q_{\\text{level}}
            = \\frac{\\lceil\\,(n+1)(1-\\alpha)\\,\\rceil}{n}

        When *n* is too small to satisfy the requested *α*, the
        quantile level exceeds 1.0.  In that case, :math:`\\hat{q}` is
        clamped to 1.0, which means *every* class is included in
        every prediction set (maximally conservative — always abstain).
        A warning is emitted so the user can supply more calibration
        data.
        """
        probs = np.asarray(calib_probs, dtype=np.float64)
        labels = np.asarray(calib_labels, dtype=np.intp)

        # ── input validation ────────────────────────────────────────
        if probs.ndim != 2:
            raise ValueError(
                f"calib_probs must be 2-D (n, K), got shape {probs.shape}"
            )
        n, K = probs.shape
        if labels.shape != (n,):
            raise ValueError(
                f"calib_labels shape {labels.shape} does not match "
                f"calib_probs row count {n}"
            )
        if labels.min() < 0 or labels.max() >= K:
            raise ValueError(
                f"Labels must be in [0, {K}), got range "
                f"[{labels.min()}, {labels.max()}]"
            )

        # ── non-conformity scores ───────────────────────────────────
        # s_i = 1 − π̂_{y_i}  (high score ⇒ model was wrong / unsure)
        true_class_probs = probs[np.arange(n), labels]
        scores = 1.0 - true_class_probs

        # ── quantile level (finite-sample adjusted) ─────────────────
        q_level_raw = math.ceil((n + 1) * (1.0 - self._alpha)) / n

        if q_level_raw > 1.0:
            import warnings
            warnings.warn(
                f"Calibration set too small (n={n}) for α={self._alpha:.3f}. "
                f"Computed q_level={q_level_raw:.4f} > 1.0 — clamping to "
                f"1.0 (all classes will be included → always abstain).  "
                f"Minimum recommended n ≈ {math.ceil(1.0 / self._alpha)}.",
                stacklevel=2,
            )
            q_level = 1.0
        else:
            q_level = q_level_raw

        # ── critical quantile q̂ ────────────────────────────────────
        # Use 'higher' interpolation so the coverage guarantee holds.
        q_hat = float(np.quantile(scores, q_level, method="higher"))

        # ── persist state ───────────────────────────────────────────
        self._q_hat = q_hat
        self._scores = scores
        self._calibration = CalibrationResult(
            alpha=self._alpha,
            q_hat=q_hat,
            q_level=q_level,
            n_calibration=n,
            scores_mean=float(scores.mean()),
            scores_std=float(scores.std()),
        )
        return self._calibration

    # ── predict_set ─────────────────────────────────────────────────

    def predict_set(
        self,
        test_probs: ArrayLike,
    ) -> NDArray[np.bool_]:
        """Compute the conformal prediction set for one or more samples.

        Parameters
        ----------
        test_probs : array-like, shape ``(K,)`` or ``(m, K)``
            Softmax probabilities for one sample (1-D) or a batch of
            *m* samples (2-D).

        Returns
        -------
        NDArray[np.bool\_], same shape as *test_probs*
            Boolean mask where ``True`` indicates the class is
            included in the prediction set.

            For a single sample ``(K,)`` → ``(K,)`` mask.
            For a batch ``(m, K)`` → ``(m, K)`` mask.

        Notes
        -----
        A class *k* is included iff:

        .. math::

            1 - \\hat{\\pi}_k \\;\\leq\\; \\hat{q}
            \\quad\\Longleftrightarrow\\quad
            \\hat{\\pi}_k \\;\\geq\\; 1 - \\hat{q}
        """
        q = self.q_hat                            # raises if not fitted
        probs = np.asarray(test_probs, dtype=np.float64)

        if probs.ndim not in (1, 2):
            raise ValueError(
                f"test_probs must be 1-D (K,) or 2-D (m, K), "
                f"got shape {probs.shape}"
            )

        # C(x) = { k : π̂_k ≥ 1 − q̂ }
        threshold = 1.0 - q
        return probs >= threshold

    # ── should_abstain ──────────────────────────────────────────────

    def should_abstain(
        self,
        test_probs: ArrayLike,
    ) -> bool | NDArray[np.bool_]:
        """Decide whether the model should abstain from diagnosing.

        Parameters
        ----------
        test_probs : array-like, shape ``(K,)`` or ``(m, K)``
            Softmax probabilities for one or more test samples.

        Returns
        -------
        bool (single sample) or NDArray[np.bool\_] (batch)
            ``True`` when the prediction set size ≠ 1 (i.e. the model
            is either vacuously empty or ambiguous between multiple
            diagnoses).

        Decision Logic
        --------------
        +------------------+-------------------------------------------+
        | ``|C(x)|``       | Action                                    |
        +==================+===========================================+
        | 0                | Abstain — no class passes threshold       |
        | 1                | **Diagnose** — exactly one class survives |
        | > 1              | Abstain — statistically ambiguous         |
        +------------------+-------------------------------------------+
        """
        pred_set = self.predict_set(test_probs)

        # Count included classes along the last axis (class axis).
        set_sizes = pred_set.sum(axis=-1)          # scalar or (m,)

        # Abstain when set size ≠ 1.
        abstain_mask = set_sizes != 1

        # Return a plain bool for a single sample.
        if abstain_mask.ndim == 0:
            return bool(abstain_mask)
        return abstain_mask

    # ── utilities ───────────────────────────────────────────────────

    def prediction_set_sizes(
        self,
        test_probs: ArrayLike,
    ) -> NDArray[np.intp]:
        """Return the number of classes in each prediction set.

        Useful for monitoring average set size as a calibration health
        metric.
        """
        return self.predict_set(test_probs).sum(axis=-1).astype(np.intp)

    def prediction_set_classes(
        self,
        test_probs: ArrayLike,
        class_names: list[str] | None = None,
    ) -> list[list[str | int]]:
        """Return the list of class labels in each prediction set.

        Parameters
        ----------
        test_probs : array-like, shape ``(K,)`` or ``(m, K)``
            Softmax probabilities.
        class_names : list[str] | None
            Optional human-readable class names.  If ``None``, integer
            indices are returned.

        Returns
        -------
        list[list[str | int]]
            One inner list per sample containing the included classes.
        """
        probs = np.asarray(test_probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]           # (1, K)

        mask = self.predict_set(probs)              # (m, K)
        result: list[list[str | int]] = []
        for row in mask:
            indices = np.where(row)[0].tolist()
            if class_names is not None:
                result.append([class_names[i] for i in indices])
            else:
                result.append(indices)
        return result

    def __repr__(self) -> str:  # noqa: D105
        status = "fitted" if self._q_hat is not None else "unfitted"
        q_str = f", q̂={self._q_hat:.4f}" if self._q_hat is not None else ""
        return f"MedicalConformalCalibrator(α={self._alpha}{q_str}, {status})"


# ──────────────────────────────────────────────────────────────────────
# 3.  SELF-TEST
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    CLASSES = [
        "Acute MI",
        "Pneumonia",
        "Stroke",
        "Appendicitis",
        "Healthy",
    ]
    K = len(CLASSES)
    N_CALIB = 100
    ALPHA = 0.05

    print("=" * 68)
    print("  BioGemma-Guard — Conformal Prediction Calibrator Self-Test")
    print("=" * 68)

    # ── Generate synthetic calibration data ─────────────────────────
    rng = np.random.default_rng(42)

    # Simulate a reasonably well-calibrated model:
    # Draw Dirichlet probabilities, then bias toward the true class.
    calib_labels = rng.integers(0, K, size=N_CALIB)
    raw = rng.dirichlet(np.ones(K), size=N_CALIB)
    # Boost the true-class probability to simulate a decent model.
    for i in range(N_CALIB):
        raw[i, calib_labels[i]] += rng.uniform(1.5, 4.0)
    calib_probs = raw / raw.sum(axis=1, keepdims=True)

    # ── Fit the calibrator ──────────────────────────────────────────
    cal = MedicalConformalCalibrator(alpha=ALPHA)
    result = cal.fit(calib_probs, calib_labels)

    print(f"\n  Configuration")
    print(f"  {'─' * 50}")
    print(f"  α (error rate)          : {ALPHA}")
    print(f"  Coverage target         : {1 - ALPHA:.0%}")
    print(f"  Number of classes (K)   : {K}")
    print(f"  Calibration samples (n) : {N_CALIB}")

    print(f"\n  Calibration Result")
    print(f"  {'─' * 50}")
    print(f"  q_level (adjusted)      : {result.q_level:.4f}")
    print(f"  q̂ (critical quantile)   : {result.q_hat:.4f}")
    print(f"  Probability threshold   : {1 - result.q_hat:.4f}")
    print(f"  Scores μ ± σ            : {result.scores_mean:.4f} ± {result.scores_std:.4f}")

    # ── Test Case 1: Highly confident prediction ────────────────────
    print(f"\n  {'─' * 50}")
    print("  Test Case 1: Highly Confident Prediction")
    print(f"  {'─' * 50}")
    confident_probs = np.array([0.92, 0.03, 0.02, 0.02, 0.01])
    pred_set_1 = cal.predict_set(confident_probs)
    abstain_1 = cal.should_abstain(confident_probs)
    classes_1 = cal.prediction_set_classes(confident_probs, CLASSES)[0]
    set_size_1 = int(pred_set_1.sum())

    print(f"  Input probabilities : {confident_probs}")
    print(f"  Prediction set      : {classes_1}")
    print(f"  Set size            : {set_size_1}")
    print(f"  Should abstain?     : {'🛑 YES' if abstain_1 else '✅ NO → Diagnose'}")

    # ── Test Case 2: Uncertain prediction (ambiguous) ───────────────
    print(f"\n  {'─' * 50}")
    print("  Test Case 2: Uncertain Prediction (Ambiguous)")
    print(f"  {'─' * 50}")
    uncertain_probs = np.array([0.30, 0.28, 0.22, 0.12, 0.08])
    pred_set_2 = cal.predict_set(uncertain_probs)
    abstain_2 = cal.should_abstain(uncertain_probs)
    classes_2 = cal.prediction_set_classes(uncertain_probs, CLASSES)[0]
    set_size_2 = int(pred_set_2.sum())

    print(f"  Input probabilities : {uncertain_probs}")
    print(f"  Prediction set      : {classes_2}")
    print(f"  Set size            : {set_size_2}")
    print(f"  Should abstain?     : {'🛑 YES → Abstain' if abstain_2 else '✅ NO'}")

    # ── Test Case 3: Batch inference ────────────────────────────────
    print(f"\n  {'─' * 50}")
    print("  Test Case 3: Batch Inference (3 samples)")
    print(f"  {'─' * 50}")
    batch_probs = np.array([
        [0.95, 0.02, 0.01, 0.01, 0.01],    # very confident
        [0.35, 0.30, 0.20, 0.10, 0.05],    # ambiguous
        [0.05, 0.05, 0.05, 0.05, 0.80],    # confident (Healthy)
    ])
    abstain_batch = cal.should_abstain(batch_probs)
    sizes_batch = cal.prediction_set_sizes(batch_probs)
    classes_batch = cal.prediction_set_classes(batch_probs, CLASSES)

    for i in range(len(batch_probs)):
        status = "🛑 Abstain" if abstain_batch[i] else "✅ Diagnose"
        print(f"  Sample {i + 1}: {batch_probs[i]}  →  "
              f"|C|={sizes_batch[i]}  {classes_batch[i]}  →  {status}")

    # ── Test Case 4: Edge case — tiny calibration set ───────────────
    print(f"\n  {'─' * 50}")
    print("  Test Case 4: Edge Case — Tiny Calibration Set (n=5)")
    print(f"  {'─' * 50}")
    tiny_cal = MedicalConformalCalibrator(alpha=0.05)
    tiny_probs = rng.dirichlet(np.ones(K), size=5)
    for i in range(5):
        tiny_probs[i, i % K] += 2.0
    tiny_probs = tiny_probs / tiny_probs.sum(axis=1, keepdims=True)
    tiny_labels = np.array([0, 1, 2, 3, 4])

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tiny_result = tiny_cal.fit(tiny_probs, tiny_labels)
        if w:
            print(f"  ⚠️  Warning: {w[0].message}")
    print(f"  q̂ = {tiny_result.q_hat:.4f}, q_level = {tiny_result.q_level:.4f}")
    print(f"  All samples should abstain: {all(tiny_cal.should_abstain(batch_probs))}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'─' * 68}")
    passed = (
        (not abstain_1)                          # confident → diagnose
        and abstain_2                            # uncertain → abstain
        and (not abstain_batch[0])               # batch confident → diagnose
        and abstain_batch[1]                     # batch uncertain → abstain
        and (not abstain_batch[2])               # batch confident → diagnose
    )
    print(f"  Overall: {'✅ All assertions passed' if passed else '❌ Some assertions failed'}")
    print("=" * 68)
