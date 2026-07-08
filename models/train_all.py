"""
Train all three horizons and print the Section 5 checkpoint:
train/val/test row counts + persistence-vs-XGBoost RMSE, side by side per horizon.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_common import (  # noqa: E402
    load_context, train_one, print_checkpoint, MODEL_DIR, XGB_PARAMS,
    STORM_WEIGHTS, PER_HORIZON_PARAMS,
)

HORIZONS = ["30min", "6h", "12h"]

if __name__ == "__main__":
    ctx = load_context()
    print(f"[data] labelled rows: {ctx['n_labelled']:,}")
    print(f"[data] split boundaries: train < {ctx['t_train_end']} "
          f"<= val < {ctx['t_val_end']} <= test")
    print(f"[data] per-horizon storm sample weight: {STORM_WEIGHTS}")
    print(f"[data] per-horizon param overrides:     {PER_HORIZON_PARAMS}")

    results = [train_one(h, ctx=ctx, storm_weight=STORM_WEIGHTS[h],
                         params={**XGB_PARAMS, **PER_HORIZON_PARAMS[h]})
               for h in HORIZONS]
    print_checkpoint(results)

    (MODEL_DIR / "metrics_section5.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] metrics -> {MODEL_DIR / 'metrics_section5.json'}")
