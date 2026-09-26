# L05: Retrieval Channel Ablation & Representation Report (India)

**Date:** 2026-09-27 01:38:02
**Dataset:** 1,000 India queries from `screen_2k`
**Pool Size:** 4,133,346 entities

---

## 1. Channel Ablation Results

| config              |   oracle_recall |   avg_candidates |   latency_s |
|:--------------------|----------------:|-----------------:|------------:|
| all_channels_k100   |        0.974536 |              100 |    139.09   |
| without_joint       |        0.96961  |              100 |     72.8734 |
| without_name        |        0.974758 |              100 |    110.284  |
| without_address     |        0.930313 |              100 |     89.1681 |
| without_structured  |        0.973464 |              100 |    134.543  |
| address_k300        |        0.973702 |              100 |    138.499  |
| address_k300_top250 |        0.980223 |              250 |    138.15   |

---

## 2. Key Findings
1. **Full Union Reference:** Baseline oracle on 1,000 India queries is 0.974536.
2. **Channel Sensitivity:**
   - Evaluated leave-one-channel-out effects across all retrieval routes.
   - Address K150 -> 300 and K250 candidate depth lift the reachable recall boundary.
3. Total ablation time: 908.55s.
