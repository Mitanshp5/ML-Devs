"""Pair features v2: schema-pinned, missingness/conflict aware (F05 12.2).

Fixes cell-24 P1s:
- _char3('') -> {''} made every empty name identical -> empty/short-aware
  ngrams + explicit availability flags; empty equality is NOT positive evidence.
- number intersection treated any shared postal/unit/house alike -> parse
  house/unit/postal separately with equality AND conflict features.
- Column_0.. names -> real feature names asserted at inference.

Country-match inside same-country partitions is constant: kept as a
diagnostic, must not be the transfer signal.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz

_HOUSE_RE = re.compile(r"\b\d{1,6}[A-Za-z]?\b")

FEATURES_V2 = [
    "tfidf_max", "n_channels",
    "name_wratio", "name_set", "name_sort", "name_partial",
    "name_char_jac", "name_len_ratio", "name_exact_nonempty",
    "name_core_jac",
    "addr_sort", "addr_set", "addr_partial", "addr_word_jac",
    "addr_present_both",
    "house_equal", "house_conflict", "unit_equal", "unit_conflict",
    "pin_equal", "pin_conflict",
    "source_is_s2",
    "rrf_best",
]

FEATURES_DENSE = FEATURES_V2 + ["dense_sim"]
FEATURES = FEATURES_V2


def char_ngrams(s: str, n: int = 3) -> set:
    s = s.replace(" ", "")
    if len(s) < n:
        return set()  # short/empty aware: no fake {''} token
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jac(a: set, b: set) -> float:
    if not a and not b:
        return 0.0  # empty-empty is not positive evidence
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _house_unit_nums(addr: str) -> set:
    return set(_HOUSE_RE.findall(addr or ""))


def pair_feature_row(q: dict, t: dict, chmap: dict, rrf: float,
                     src_is_s2: bool, dense_sim: float | None = None) -> list:
    qn, tn = q["name_unicode"], t["name_unicode"]
    qa, ta = q["address_unicode"], t["address_unicode"]
    scores = [s for s, _ in chmap.values()] if chmap else [0.0]
    la, lb = len(qn), len(tn)
    out = [
        max(scores),
        float(len(chmap)),
        fuzz.WRatio(qn, tn) / 100.0,
        fuzz.token_set_ratio(qn, tn) / 100.0,
        fuzz.token_sort_ratio(qn, tn) / 100.0,
        fuzz.partial_ratio(qn, tn) / 100.0,
        _jac(char_ngrams(qn), char_ngrams(tn)),
        (min(la, lb) / max(la, lb)) if max(la, lb) else 0.0,
        1.0 if (qn and qn == tn) else 0.0,
        _jac(set(q.get("name_core", "").split()), set(t.get("name_core", "").split())),
        fuzz.token_sort_ratio(qa, ta) / 100.0,
        fuzz.token_set_ratio(qa, ta) / 100.0,
        fuzz.partial_ratio(qa, ta) / 100.0,
        _jac(set(qa.split()), set(ta.split())),
        1.0 if (qa.strip() and ta.strip()) else 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # numeric slots filled below
        1.0 if src_is_s2 else 0.0,
        float(rrf),
    ]
    qn_nums = q.get("numbers", {})
    tn_nums = t.get("numbers", {})
    qh, th = set(qn_nums.get("house_tokens", [])), set(tn_nums.get("house_tokens", []))
    qu, tu = set(qn_nums.get("unit_tokens", [])), set(tn_nums.get("unit_tokens", []))
    qp = (qn_nums.get("postal_candidates", []) or [None])[0]
    tp = (tn_nums.get("postal_candidates", []) or [None])[0]
    out[15] = 1.0 if (qh & th) else 0.0
    out[16] = 1.0 if (qh and th and not (qh & th)) else 0.0
    out[17] = 1.0 if (qu & tu) else 0.0
    out[18] = 1.0 if (qu and tu and not (qu & tu)) else 0.0
    out[19] = 1.0 if (qp and qp == tp) else 0.0
    out[20] = 1.0 if (qp and tp and qp != tp) else 0.0
    if dense_sim is not None:
        out.append(float(dense_sim))
    return out


def rows_to_matrix(rows: list) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


# ==============================================================================
# FEATURES_V3 (38 Features) - Decomposed channels, competition, diagnostics
# ==============================================================================

FEATURES_V3 = [
    # 1. Granular channel decomposition (replacing mixed tfidf_max)
    "score_joint", "recip_rank_joint",
    "score_name", "recip_rank_name",
    "score_addr", "recip_rank_addr",
    "score_struct", "recip_rank_struct",
    "n_channels",
    # 2. Name string similarities
    "name_wratio", "name_set", "name_sort", "name_partial",
    "name_char_jac", "name_len_ratio", "name_exact_nonempty",
    "name_core_jac",
    # 3. Address string similarities
    "addr_sort", "addr_set", "addr_partial", "addr_word_jac",
    "addr_present_both",
    # 4. Number & postal evidence
    "house_equal", "house_conflict",
    "unit_equal", "unit_conflict",
    "pin_equal", "pin_conflict",
    # 5. Source, RRF, Candidate Rank
    "source_is_s2",
    "rrf_score",
    "candidate_rank",
    # 6. Candidate Competition & Margin Context
    "rrf_diff_to_top",
    "rrf_margin_top1_top2",
    # 7. Targeted Diagnostic Error Flags
    "same_name_diff_addr",
    "same_addr_diff_name",
    # 8. Missingness Flags
    "name_empty_either",
    "addr_empty_either",
    "pin_present_both",
]

FEATURES_V4 = FEATURES_V3 + [
    "name_match_addr_missing", "strict_house_number_conflict",
    "script_mismatch_high_addr", "same_pin_diff_name_samescript",
]

def _has_nonlatin(text: str) -> bool:
    for ch in str(text):
        if unicodedata.category(ch).startswith("L") and "LATIN" not in unicodedata.name(ch, ""):
            return True
    return False

def pair_feature_row_v4(q: dict, t: dict, chmap: dict, rrf: float, rank: int,
                        top1_rrf: float, top2_rrf: float, src_is_s2: bool) -> list:
    """Production FEATURES_V4 row; formulas are pinned to R3b training."""
    base = pair_feature_row_v3(q, t, chmap, rrf, rank, top1_rrf, top2_rrf, src_is_s2)
    name_match_addr_missing = float((base[11] >= .85 or base[9] >= .90) and base[36] == 1.0)
    strict_house_number_conflict = float(base[23] == 1.0 and (base[11] >= .65 or base[17] >= .75))
    script_mismatch_high_addr = float(_has_nonlatin(q.get("name_unicode", "")) != _has_nonlatin(t.get("name_unicode", "")) and base[17] >= .70)
    same_pin_diff_name_samescript = float(base[26] == 1.0 and base[11] < .35 and (_has_nonlatin(q.get("name_unicode", "")) == _has_nonlatin(t.get("name_unicode", ""))))
    return base + [name_match_addr_missing, strict_house_number_conflict, script_mismatch_high_addr, same_pin_diff_name_samescript]


def pair_feature_row_v3(
    q: dict,
    t: dict,
    chmap: dict,
    rrf: float,
    rank: int,
    top1_rrf: float,
    top2_rrf: float,
    src_is_s2: bool,
    dense_sim: float | None = None,
) -> list:
    qn, tn = q["name_unicode"], t["name_unicode"]
    qa, ta = q["address_unicode"], t["address_unicode"]
    la, lb = len(qn), len(tn)

    # 1. Channel decomposition
    sj, rj = chmap.get("joint", (0.0, 999))
    sn, rn = chmap.get("name_only", (0.0, 999))
    sa, ra = chmap.get("address_only", (0.0, 999))
    ss, rs = chmap.get("structured", (0.0, 999))

    recip_j = 1.0 / (rj + 60.0) if "joint" in chmap else 0.0
    recip_n = 1.0 / (rn + 60.0) if "name_only" in chmap else 0.0
    recip_a = 1.0 / (ra + 60.0) if "address_only" in chmap else 0.0
    recip_s = 1.0 / (rs + 60.0) if "structured" in chmap else 0.0

    # 2. Name metrics
    n_exact = 1.0 if (qn and qn == tn) else 0.0
    name_sort = fuzz.token_sort_ratio(qn, tn) / 100.0

    # 3. Address metrics
    addr_jac = _jac(set(qa.split()), set(ta.split()))

    # 4. Numbers & postal
    qn_nums = q.get("numbers", {})
    tn_nums = t.get("numbers", {})
    qh, th = set(qn_nums.get("house_tokens", [])), set(tn_nums.get("house_tokens", []))
    qu, tu = set(qn_nums.get("unit_tokens", [])), set(tn_nums.get("unit_tokens", []))
    qp = (qn_nums.get("postal_candidates", []) or [None])[0]
    tp = (tn_nums.get("postal_candidates", []) or [None])[0]

    h_eq = 1.0 if (qh & th) else 0.0
    h_cf = 1.0 if (qh and th and not (qh & th)) else 0.0
    u_eq = 1.0 if (qu & tu) else 0.0
    u_cf = 1.0 if (qu and tu and not (qu & tu)) else 0.0
    p_eq = 1.0 if (qp and qp == tp) else 0.0
    p_cf = 1.0 if (qp and tp and qp != tp) else 0.0

    # 6. Competition & margin
    diff_to_top = float(rrf - top1_rrf)
    margin_top12 = float(top1_rrf - top2_rrf)

    # 7. Targeted diagnostics
    same_name_diff_addr = 1.0 if (n_exact == 1.0 and (h_cf == 1.0 or p_cf == 1.0 or addr_jac < 0.25)) else 0.0
    same_addr_diff_name = 1.0 if (p_eq == 1.0 and h_eq == 1.0 and name_sort < 0.40) else 0.0

    # 8. Missingness flags
    name_empty = 1.0 if (not qn or not tn) else 0.0
    addr_empty = 1.0 if (not qa or not ta) else 0.0
    pin_both = 1.0 if (qp and tp) else 0.0

    out = [
        sj, recip_j,
        sn, recip_n,
        sa, recip_a,
        ss, recip_s,
        float(len(chmap)),
        fuzz.WRatio(qn, tn) / 100.0,
        fuzz.token_set_ratio(qn, tn) / 100.0,
        name_sort,
        fuzz.partial_ratio(qn, tn) / 100.0,
        _jac(char_ngrams(qn), char_ngrams(tn)),
        (min(la, lb) / max(la, lb)) if max(la, lb) else 0.0,
        n_exact,
        _jac(set(q.get("name_core", "").split()), set(t.get("name_core", "").split())),
        fuzz.token_sort_ratio(qa, ta) / 100.0,
        fuzz.token_set_ratio(qa, ta) / 100.0,
        fuzz.partial_ratio(qa, ta) / 100.0,
        addr_jac,
        1.0 if (qa.strip() and ta.strip()) else 0.0,
        h_eq, h_cf,
        u_eq, u_cf,
        p_eq, p_cf,
        1.0 if src_is_s2 else 0.0,
        float(rrf),
        float(rank),
        diff_to_top,
        margin_top12,
        same_name_diff_addr,
        same_addr_diff_name,
        name_empty,
        addr_empty,
        pin_both,
    ]
    if dense_sim is not None:
        out.append(float(dense_sim))
    return out
