"""
medical_rewards.py — BioGemma-Guard Reward Functions for GRPO Training
=====================================================================

Reward module designed for the GRPOTrainer (trl library) that enforces
three pillars of clinical reasoning:

    1. Accuracy   → Correct diagnoses are strongly rewarded (+2.0).
    2. Honesty    → Abstention on ambiguous inputs is positively
                    reinforced (+0.8), preventing dangerous guesses.
    3. Safety     → Hallucinated / incorrect answers receive a severe
                    penalty (-3.0) to discourage medical misinformation.

An additional Chain-of-Thought bonus (+0.2) incentivises transparent
reasoning traces before the final answer.

Compatibility
-------------
- trl.GRPOTrainer reward function signature:
      reward_fn(prompts, completions, **kwargs) → list[float]
- The ``ground_truth_answers`` are forwarded by GRPOTrainer through
  the dataset column mapping (see ``reward_kwargs`` in GRPOConfig).

Usage
-----
>>> from src.rewards.medical_rewards import medical_reward_func
>>>
>>> trainer = GRPOTrainer(
...     ...
...     reward_funcs=[medical_reward_func],
...     reward_kwargs={"ground_truth_answers": "answer"},  # dataset col
... )

Author : BioGemma-Guard Team
License: Apache-2.0
"""

from __future__ import annotations

import re
from typing import Any

from thefuzz import fuzz

# ──────────────────────────────────────────────────────────────────────
# 1.  CONFIGURABLE REWARD CONSTANTS
# ──────────────────────────────────────────────────────────────────────

REWARD_CORRECT: float = 2.0
"""Reward granted when the completion contains the correct answer."""

REWARD_ABSTENTION: float = 0.8
"""Reward granted for an honest abstention on an ambiguous question."""

REWARD_HALLUCINATION: float = -3.0
"""Penalty applied when the completion provides an incorrect answer."""

BONUS_CHAIN_OF_THOUGHT: float = 0.2
"""Small bonus for including visible reasoning / thinking steps."""

# ──────────────────────────────────────────────────────────────────────
# 2.  ABSTENTION PHRASE PATTERNS  (compiled once at import time)
# ──────────────────────────────────────────────────────────────────────

_ABSTENTION_PHRASES: list[str] = [
    # English abstention signals
    r"i am uncertain",
    r"i'm uncertain",
    r"i am not (entirely )?sure",
    r"i'm not (entirely )?sure",
    r"insufficient data",
    r"insufficient information",
    r"not enough (clinical )?data",
    r"not enough (clinical )?information",
    r"cannot (safely |reliably )?determine",
    r"unable to (safely |reliably )?determine",
    r"cannot provide a (safe |reliable |definitive )?diagnosis",
    r"unable to provide a (safe |reliable |definitive )?diagnosis",
    r"more (clinical )?(data|information|context) (is )?needed",
    r"requires? (additional|further|more) (clinical )?(data|information|context|evaluation|assessment)",
    r"i (would )?recommend (consulting|seeking) (a )?(specialist|physician|doctor)",
    r"beyond (the scope of )?my (current )?(capabilities|knowledge)",
    r"clinical (judgment|assessment) (is )?required",
    r"this case (is )?(ambiguous|unclear|inconclusive)",
    r"i (must |should )?abstain",
    r"abstaining from (a )?diagnosis",
    r"confidence (level |score )?(is )?(too )?low",
    r"high (degree of )?uncertainty",
    r"the (available )?evidence is (insufficient|inconclusive|ambiguous)",
]

_ABSTENTION_RE: re.Pattern = re.compile(
    "|".join(f"(?:{p})" for p in _ABSTENTION_PHRASES),
    flags=re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────────
# 3.  CHAIN-OF-THOUGHT DETECTION PATTERNS
# ──────────────────────────────────────────────────────────────────────

_COT_MARKERS: list[str] = [
    # ── Explicit section markers (XML-style or markdown) ────────────
    r"<think(ing)?[^>]*>",
    r"</think(ing)?>",
    r"\*\*think(ing)?\*\*",
    r"\*\*reason(ing)?\*\*",
    r"\*\*analysis\*\*",
    r"##?\s*(think(ing)?|reason(ing)?|analysis|step[- ]by[- ]step)",

    # ── Numbered reasoning steps ───────────────────────────────────
    r"step\s*\d+\s*[:\-]",

    # ── Inline reasoning signals ───────────────────────────────────
    r"(first|second|third|finally|therefore|thus|hence|because|consequently|given that),?\s",
    r"(let me|let's)\s+(think|reason|analyze|consider|evaluate|assess)",
    r"(rule out|ruling out|r/o)\s",
    r"(based on|considering|given)\s+(the |these |this )?(symptoms?|findings?|presentation|vitals?|lab(s| results)?)",

    # ── SOAP Note Structure ────────────────────────────────────────
    r"\b(subjective|objective|assessment|plan)\s*[:\-]",

    # ── Clinical Evaluation Terms ──────────────────────────────────
    r"\b(chief complaint|cc)\s*[:\-]",
    r"\b(hpi|history of present illness)\s*[:\-]",
    r"\b(vital signs|vitals)\s*[:\-]",
    r"\b(physical exam(ination)?|pe)\s*[:\-]",
    r"\b(labs?|lab(oratory)? results?)\s*[:\-]",
    r"\b(imaging|radiolog(y|ical) findings?)\s*[:\-]",

    # ── Diagnostic Logic ───────────────────────────────────────────
    r"\b(differential diagnosis|differentials?|ddx)\s*[:\-]",
    r"\b(working diagnosis|provisional diagnosis)\s*[:\-]",

    # ── Triage & Clinical Scoring ──────────────────────────────────
    r"\b(triage level|triage category)\s*[:\-]",
    r"\bgcs\s*(score)?\s*[:\-=]",
    r"\bnews2?\s*(score)?\s*[:\-=]",
]

_COT_RE: re.Pattern = re.compile(
    "|".join(f"(?:{p})" for p in _COT_MARKERS),
    flags=re.IGNORECASE,
)

# Minimum number of distinct reasoning markers to qualify for the CoT bonus.
_COT_MIN_MARKER_COUNT: int = 2

# ──────────────────────────────────────────────────────────────────────
# 4.  AMBIGUITY DETECTION  (ground-truth only — no prompt scanning)
# ──────────────────────────────────────────────────────────────────────

# Special ground-truth tokens that signal an ambiguous / unanswerable question.
# Ambiguity is determined *exclusively* from the ground-truth label to avoid
# false positives (e.g. "patient has an unclear history, but labs show…" is
# answerable despite containing the word "unclear").
_AMBIGUOUS_GROUND_TRUTH_TOKENS: set[str] = {
    "abstain",
    "uncertain",
    "ambiguous",
    "unanswerable",
    "insufficient",
    "n/a",
    "none",
    "not applicable",
    "inconclusive",
}

# ──────────────────────────────────────────────────────────────────────
# 5.  TEXT NORMALISATION & ANSWER MATCHING HELPERS
# ──────────────────────────────────────────────────────────────────────

# Regex to strip XML-style thinking blocks before answer extraction.
_THINK_BLOCK_RE: re.Pattern = re.compile(
    r"<think(ing)?[^>]*>.*?</think(ing)?>",
    flags=re.IGNORECASE | re.DOTALL,
)

# Common answer extraction patterns.
_ANSWER_EXTRACTION_RE: re.Pattern = re.compile(
    r"(?:"
    r"(?:final\s+)?answer\s*[:：]\s*"       # "Answer:" or "Final Answer:"
    r"|(?:diagnosis|dx)\s*[:：]\s*"         # "Diagnosis:"
    r"|(?:conclusion)\s*[:：]\s*"           # "Conclusion:"
    r"|(?:the\s+answer\s+is)\s+"            # "The answer is ..."
    r"|(?:\*\*(?:answer|diagnosis)\*\*\s*[:：]?\s*)"  # **Answer**:
    r")"
    r"(.+?)(?:\n|$)",                        # capture the answer text
    flags=re.IGNORECASE,
)


def _normalise(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for fuzzy match."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)     # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def _extract_answer_segment(completion: str) -> str:
    """
    Try to extract the explicit 'answer' portion from a completion.
    Falls back to the last non-empty line of the completion.
    """
    # Strip away <thinking> blocks first.
    clean = _THINK_BLOCK_RE.sub("", completion).strip()

    match = _ANSWER_EXTRACTION_RE.search(clean)
    if match:
        return match.group(1).strip()

    # Fallback: use the last non-empty line.
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    return lines[-1] if lines else clean


# Fuzzy-match threshold (0–100).  Applied to the best of
# token_set_ratio (handles word reordering / extra tokens) and
# partial_ratio (handles truncated terms like Infarct → Infarction).
_FUZZY_MATCH_THRESHOLD: int = 85


def _fuzzy_score(a: str, b: str) -> int:
    """Return the best fuzzy score between two normalised strings."""
    return max(fuzz.token_set_ratio(a, b), fuzz.partial_ratio(a, b))


def _answer_matches(completion: str, ground_truth: str) -> bool:
    """
    Determine whether the completion contains the ground-truth answer.

    Uses fuzzy matching via ``thefuzz`` to handle real-world medical
    terminology variations (e.g. 'Myocardial Infarction' vs 'Acute
    Myocardial Infarct').  The best of ``token_set_ratio`` (word
    reordering) and ``partial_ratio`` (substring / truncation) is
    compared against the threshold.

    Strategy:
      1. **Extracted segment** — fuzzy-match the explicitly marked
         answer against the ground truth.
      2. **Full-text fallback** — fuzzy-match the entire (normalised)
         completion against the ground truth.
      3. **Multi-choice shorthand** — regex match for option letters.
    """
    gt_norm = _normalise(ground_truth)
    if not gt_norm:
        return False

    # Tier 1: extracted answer segment (fuzzy).
    segment = _extract_answer_segment(completion)
    if _fuzzy_score(gt_norm, _normalise(segment)) >= _FUZZY_MATCH_THRESHOLD:
        return True

    # Tier 2: full completion fallback (fuzzy).
    full_comp = _normalise(completion)
    if _fuzzy_score(gt_norm, full_comp) >= _FUZZY_MATCH_THRESHOLD:
        return True

    # Tier 3: multi-choice shorthand (e.g. ground truth is just "A", "B").
    if len(gt_norm) <= 2 and gt_norm.isalpha():
        mc_pattern = re.compile(
            rf"(?:\(?{re.escape(gt_norm)}\)?|option\s+{re.escape(gt_norm)})\b",
            re.IGNORECASE,
        )
        if mc_pattern.search(completion):
            return True

    return False


def _is_abstention(completion: str) -> bool:
    """Return True if the completion contains any abstention signal."""
    return bool(_ABSTENTION_RE.search(completion))


def _is_ambiguous_question(prompt: str, ground_truth: str) -> bool:
    """Determine whether a question is ambiguous / unanswerable.

    Detection is based **exclusively** on the ground-truth label to
    prevent false positives.  Prompts like 'patient has an unclear
    history, but labs show…' are answerable despite containing
    ambiguity-like keywords.

    Parameters
    ----------
    prompt : str
        The input prompt (unused — kept for API stability).
    ground_truth : str
        The reference answer label.
    """
    gt_norm = _normalise(ground_truth)
    return gt_norm in _AMBIGUOUS_GROUND_TRUTH_TOKENS


def _has_chain_of_thought(completion: str) -> bool:
    """
    Return True if the completion demonstrates Chain-of-Thought
    reasoning — i.e. at least ``_COT_MIN_MARKER_COUNT`` distinct
    reasoning markers are present.
    """
    matches = _COT_RE.findall(completion)
    return len(matches) >= _COT_MIN_MARKER_COUNT


# ──────────────────────────────────────────────────────────────────────
# 6.  CORE REWARD FUNCTION  (GRPOTrainer-compatible)
# ──────────────────────────────────────────────────────────────────────

def medical_reward_func(
    prompts: list[str],
    completions: list[str],
    ground_truth_answers: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Compute per-sample rewards for a batch of GRPO completions.

    Parameters
    ----------
    prompts : list[str]
        The input prompts (questions) sent to the model.
    completions : list[str]
        The model's generated completions / answers.
    ground_truth_answers : list[str] | None
        Reference answers for each prompt.  When a question is
        intentionally ambiguous, the ground truth should be one of the
        tokens in ``_AMBIGUOUS_GROUND_TRUTH_TOKENS`` (e.g. ``"abstain"``).
    **kwargs
        Additional keyword arguments forwarded by GRPOTrainer
        (silently ignored).

    Returns
    -------
    list[float]
        A reward score for each (prompt, completion) pair.

    Reward Logic
    ------------
    +--------------------------+---------+
    | Outcome                  | Reward  |
    +==========================+=========+
    | Correct answer           | +2.0    |
    | Honest abstention        | +0.8    |
    | Incorrect / hallucinated | -3.0    |
    | CoT bonus (additive)     | +0.2    |
    +--------------------------+---------+

    Notes
    -----
    If ``ground_truth_answers`` is not supplied, every completion is
    assigned a neutral reward of 0.0 (no signal).  This lets the
    trainer run safely even when labels are unavailable.
    """
    batch_size = len(completions)

    # Defensive: if ground truths are missing, return neutral rewards.
    if ground_truth_answers is None:
        return [0.0] * batch_size

    assert len(prompts) == batch_size, (
        f"Batch size mismatch: {len(prompts)} prompts vs "
        f"{batch_size} completions."
    )
    assert len(ground_truth_answers) == batch_size, (
        f"Batch size mismatch: {len(ground_truth_answers)} ground truths "
        f"vs {batch_size} completions."
    )

    rewards: list[float] = []

    for prompt, completion, gt in zip(
        prompts, completions, ground_truth_answers, strict=True,
    ):
        reward = _score_single(prompt, completion, gt)
        rewards.append(reward)

    return rewards


def _score_single(prompt: str, completion: str, ground_truth: str) -> float:
    """Score an individual (prompt, completion, ground_truth) triplet.

    This function implements the full decision tree:

        ┌─ Question is ambiguous?
        │   ├─ Model abstains   → +0.8  (honest)
        │   └─ Model answers    → -3.0  (reckless guess)
        └─ Question is answerable
            ├─ Answer correct   → +2.0
            └─ Answer incorrect → -3.0  (hallucination)

        + CoT bonus (+0.2) applied additively in all cases where
          visible reasoning is detected.
    """
    score: float = 0.0
    abstained = _is_abstention(completion)
    ambiguous = _is_ambiguous_question(prompt, ground_truth)

    if ambiguous:
        # ── AMBIGUOUS QUESTION ──────────────────────────────────────
        if abstained:
            score = REWARD_ABSTENTION          # +0.8 — honest refusal
        else:
            score = REWARD_HALLUCINATION       # -3.0 — reckless guess
    else:
        # ── ANSWERABLE QUESTION ─────────────────────────────────────
        if abstained:
            # Model chickened out on a clear question → mild penalty
            # (less harsh than hallucination but not rewarded).
            score = -0.5
        elif _answer_matches(completion, ground_truth):
            score = REWARD_CORRECT             # +2.0 — correct
        else:
            score = REWARD_HALLUCINATION       # -3.0 — hallucination

    # ── CHAIN-OF-THOUGHT BONUS ──────────────────────────────────────
    if _has_chain_of_thought(completion):
        score += BONUS_CHAIN_OF_THOUGHT        # +0.2

    return round(score, 2)


# ──────────────────────────────────────────────────────────────────────
# 7.  CONVENIENCE: PRE-BUILT REWARD LIST FOR GRPOTrainer
# ──────────────────────────────────────────────────────────────────────

# Some projects register multiple reward functions (e.g. accuracy +
# format).  Export a ready-made list for direct use:
#     reward_funcs = BIOGEMMA_REWARD_FUNCS
BIOGEMMA_REWARD_FUNCS: list = [medical_reward_func]


# ──────────────────────────────────────────────────────────────────────
# 8.  SELF-TEST  (run with:  python -m src.rewards.medical_rewards)
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  BioGemma-Guard — Reward Function Self-Test")
    print("=" * 64)

    # ── Test Cases ──────────────────────────────────────────────────
    test_cases = [
        {
            "name": "Correct answer with CoT",
            "prompt": "Patient presents with crushing chest pain radiating to left arm. Diagnosis?",
            "completion": (
                "<thinking>\n"
                "Step 1: The patient has crushing chest pain radiating to the left arm.\n"
                "Step 2: This is a classic presentation of acute myocardial infarction.\n"
                "Therefore, the most likely diagnosis is MI.\n"
                "</thinking>\n"
                "**Answer:** Acute Myocardial Infarction"
            ),
            "ground_truth": "Acute Myocardial Infarction",
            "expected": REWARD_CORRECT + BONUS_CHAIN_OF_THOUGHT,  # 2.2
        },
        {
            "name": "Fuzzy match — terminology variation",
            "prompt": "Patient presents with crushing chest pain. Diagnosis?",
            "completion": "Diagnosis: Acute Myocardial Infarct",
            "ground_truth": "Myocardial Infarction",
            "expected": REWARD_CORRECT,  # 2.0 — fuzzy match catches this
        },
        {
            "name": "Correct answer without CoT",
            "prompt": "What is the first-line treatment for anaphylaxis?",
            "completion": "The answer is Epinephrine.",
            "ground_truth": "Epinephrine",
            "expected": REWARD_CORRECT,  # 2.0
        },
        {
            "name": "Honest abstention on ambiguous question",
            "prompt": "Patient has non-specific symptoms with insufficient clinical data. Diagnosis?",
            "completion": (
                "I am uncertain about this case. "
                "Insufficient data to provide a safe diagnosis. "
                "I recommend consulting a specialist."
            ),
            "ground_truth": "abstain",
            "expected": REWARD_ABSTENTION,  # 0.8
        },
        {
            "name": "Hallucination — wrong answer",
            "prompt": "Patient with sudden severe headache, stiff neck, photophobia. Diagnosis?",
            "completion": "Diagnosis: Common cold. Rest and fluids recommended.",
            "ground_truth": "Subarachnoid Hemorrhage",
            "expected": REWARD_HALLUCINATION,  # -3.0
        },
        {
            "name": "Reckless guess on ambiguous question",
            "prompt": "Patient presents with insufficient clinical data and conflicting findings. Diagnosis?",
            "completion": "Diagnosis: Pneumonia",
            "ground_truth": "abstain",
            "expected": REWARD_HALLUCINATION,  # -3.0
        },
        {
            "name": "No false positive — 'unclear' in prompt but answerable",
            "prompt": "Patient has an unclear history, but labs show TSH 12.5 mIU/L and free T4 0.4 ng/dL. Diagnosis?",
            "completion": "Diagnosis: Hypothyroidism",
            "ground_truth": "Hypothyroidism",
            "expected": REWARD_CORRECT,  # 2.0 — NOT penalised despite 'unclear' in prompt
        },
        {
            "name": "Unnecessary abstention on clear question",
            "prompt": "A 55-year-old diabetic patient with HbA1c of 9.2%. What medication class is first-line?",
            "completion": (
                "I am uncertain about this. "
                "I recommend consulting a specialist for further evaluation."
            ),
            "ground_truth": "Metformin",
            "expected": -0.5,
        },
        {
            "name": "Multi-choice correct (option letter)",
            "prompt": "Which of the following is a beta-blocker? (A) Lisinopril (B) Metoprolol (C) Amlodipine",
            "completion": (
                "Let me analyze each option.\n"
                "First, Lisinopril is an ACE inhibitor.\n"
                "Second, Metoprolol is indeed a beta-blocker.\n"
                "Third, Amlodipine is a calcium channel blocker.\n"
                "The answer is (B)."
            ),
            "ground_truth": "B",
            "expected": REWARD_CORRECT + BONUS_CHAIN_OF_THOUGHT,  # 2.2
        },
        {
            "name": "Honest abstention on ambiguous question with CoT",
            "prompt": "Patient presents with vague symptoms and limited clinical history. What is the diagnosis?",
            "completion": (
                "<thinking>\n"
                "Step 1: The symptoms described are very non-specific.\n"
                "Step 2: Without further lab results or imaging, multiple differential diagnoses are possible.\n"
                "Therefore, I cannot narrow this down.\n"
                "</thinking>\n"
                "I am uncertain. Insufficient data to provide a safe diagnosis."
            ),
            "ground_truth": "uncertain",
            "expected": REWARD_ABSTENTION + BONUS_CHAIN_OF_THOUGHT,  # 1.0
        },
        {
            "name": "SOAP-note style CoT with correct answer",
            "prompt": "72-year-old male, fever 39.2C, productive cough, CXR shows lobar consolidation. Diagnosis?",
            "completion": (
                "Subjective: Patient reports 3-day history of fever and cough.\n"
                "Objective: Temp 39.2C, CXR with lobar consolidation.\n"
                "Assessment: Community-acquired pneumonia\n"
                "Plan: Start empiric antibiotics (amoxicillin-clavulanate)."
            ),
            "ground_truth": "Community-Acquired Pneumonia",
            "expected": REWARD_CORRECT + BONUS_CHAIN_OF_THOUGHT,  # 2.2
        },
    ]

    passed = 0
    failed = 0

    for tc in test_cases:
        result = _score_single(tc["prompt"], tc["completion"], tc["ground_truth"])
        status = "✅ PASS" if abs(result - tc["expected"]) < 1e-9 else "❌ FAIL"

        if "PASS" in status:
            passed += 1
        else:
            failed += 1

        print(f"\n  {status}  {tc['name']}")
        print(f"           Expected: {tc['expected']:+.2f}  |  Got: {result:+.2f}")

    print("\n" + "─" * 64)
    print(f"  Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 64)
