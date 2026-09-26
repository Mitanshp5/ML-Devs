# L03: Candidate Depth Sweep Report

**Date:** 2026-09-27 00:05:11
**Hardware:** Windows / 12 CPU threads
**Model:** Frozen `control_b0` clean reference (743 trees)
**Calibration:** Independent two-threshold optimization strictly on `calibration_5k`
**Evaluation:** Out-of-sample on `screen_2k` (1,000 India, 1,000 US)

---

## 1. Candidate Depth Frontier Results

| Arm | Policy Description | Candidate Oracle | Screen Macro $F_0.5$ | India $F_0.5$ | US $F_0.5$ | $\Delta$ vs C0 Ref | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|
| **C0_K100_ref** | B0 Reference (India K100, US K100) | 0.985159 | 0.906774 | 0.875299 | 0.938250 | *Reference* | — |
| **C1_India_K250** | India K250, US K100 (Candidate Depth Challenger) | **0.988003** | **0.906710** | **0.875171** | 0.938250 | **-0.000064** | [-0.000344, +0.000133] |
| **C2_India_untrimmed** | India Untrimmed, US K100 | 0.989500 | 0.906605 | 0.874960 | 0.938250 | -0.000169 | [-0.000549, +0.000095] |
| **C3_all_untrimmed** | All Untrimmed (Full Natural Union) | 0.990452 | 0.906836 | 0.874960 | 0.938711 | +0.000061 | [-0.000361, +0.000428] |

---

## 2. Key Observations
1. **India Candidate Ceiling:** Expanding India candidate depth from K100 to K250 increases the oracle ceiling from 0.974536 to 0.980223.
2. **Threshold Shift:** When evaluating K250, thresholds calibrated on `calibration_5k` protect against false-positive inflation.
3. Total execution time: 1485.23s.
