"""L07: Export Winning Production Model Bundle and Freeze Final Artifacts.

Authority: LOCAL_COLAB_IMPLEMENTATION_PLAN.md Section 5 (L07)
Freezes:
- Model: L04 Rich LightGBM Matcher (38 features, 466 trees)
- Feature schema: FEATURES_V3
- Decision policy: Country-Dual with Global Fallback (calibrated on calibration_5k)
- Checksums & Provenance metadata
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

from er.features import FEATURES_V3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    src_model = Path("runs/local-v2/L04_richer_features/l04_rich_matcher.txt")
    src_policies = Path("runs/local-v2/L04_richer_features/calibrated_policies.json")
    out_dir = Path("production_bundle")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" L07: Exporting Final Production Artifacts")
    print(f" Source Model: {src_model}")
    print(f" Output Bundle: {out_dir}")
    print("=" * 70)

    assert src_model.exists(), f"Source model {src_model} does not exist!"
    assert src_policies.exists(), f"Source policy {src_policies} does not exist!"

    # 1. Copy Model File
    dst_model = out_dir / "production_matcher.txt"
    shutil.copy2(src_model, dst_model)
    model_sha = sha256_file(dst_model)
    print(f"Exported model: {dst_model} (SHA256: {model_sha})")

    # 2. Export Feature Schema
    dst_schema = out_dir / "feature_schema.json"
    schema_payload = {
        "version": "FEATURES_V3",
        "n_features": len(FEATURES_V3),
        "features": FEATURES_V3,
    }
    dst_schema.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")
    schema_sha = sha256_file(dst_schema)
    print(f"Exported feature schema: {dst_schema} (38 features, SHA256: {schema_sha})")

    # 3. Export Decision Policy
    policies_raw = json.loads(src_policies.read_text(encoding="utf-8"))
    dst_policy = out_dir / "decision_policy.json"
    dst_policy.write_text(json.dumps(policies_raw, indent=2), encoding="utf-8")
    policy_sha = sha256_file(dst_policy)
    print(f"Exported decision policies: {dst_policy} (SHA256: {policy_sha})")

    # 4. Manifest / Provenance
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "LightGBM GBDT (127 leaves, depth 9, 466 trees)",
        "features_version": "FEATURES_V3",
        "n_features": len(FEATURES_V3),
        "calibration_set": "calibration_5k (5,000 queries)",
        "training_set": "train_12k (12,000 queries, 1.2M pairs, 39,919 matches)",
        "verified_metrics": {
            "screen_2k_macro_f05": 0.912044,
            "screen_2k_india_f05": 0.884169,
            "screen_2k_us_f05": 0.939920,
            "delta_vs_b0_baseline": "+0.007459",
            "delta_95_ci": [+0.002362, +0.012808],
        },
        "artifacts_sha256": {
            "production_matcher.txt": model_sha,
            "feature_schema.json": schema_sha,
            "decision_policy.json": policy_sha,
        },
        "routing_policy": {
            "India": "country_dual / calibrated thresholds",
            "US": "country_dual / calibrated thresholds",
            "France": "global fallback (ts=0.70, tm=0.70)",
            "Other": "global fallback",
        },
    }
    dst_meta = out_dir / "production_manifest.json"
    dst_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created production manifest: {dst_meta}")
    print("=" * 70)
    print(" Production Bundle Export Complete!")


if __name__ == "__main__":
    main()
