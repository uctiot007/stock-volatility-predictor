import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_RESULTS_PATH = Path("outputs/results.json")


def build_dashboard_payload(results_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(results_path or DEFAULT_RESULTS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    metrics = {
        "naive_mse": float(results.get("naive_mse", 0.0)),
        "ridge_mse": float(results.get("ridge_single_split_mse", 0.0)),
        "lasso_mse": float(results.get("lasso_single_split_mse", 0.0)),
        "ridge_walk_forward_mse": float(results.get("ridge_walk_forward_avg_mse", 0.0)),
        "arima_mse": float(results.get("arima_rolling_avg_mse", 0.0)),
        "gjr_garch_mse": float(results.get("gjr_garch_t_weekly_refit_mse", 0.0)),
    }

    ranked = sorted(
        [
            ("Naive baseline", metrics["naive_mse"]),
            ("Custom Ridge GD", metrics["ridge_mse"]),
            ("Custom Lasso GD", metrics["lasso_mse"]),
            ("Ridge walk-forward", metrics["ridge_walk_forward_mse"]),
            ("ARIMA rolling", metrics["arima_mse"]),
            ("GJR-GARCH", metrics["gjr_garch_mse"]),
        ],
        key=lambda item: item[1],
    )

    summary = {
        "best_model": ranked[0][0],
        "best_mse": ranked[0][1],
        "naive_vs_best": metrics["naive_mse"] - ranked[0][1],
        "generated_at": results.get("timestamp", "n/a"),
    }

    return {
        "summary": summary,
        "metrics": metrics,
        "config": results.get("config", {}),
        "rankings": [
            {"name": name, "mse": mse} for name, mse in ranked
        ],
        "raw": results,
    }
