import json
import tempfile
import unittest
from pathlib import Path

from src.dashboard import build_dashboard_payload


class DashboardPayloadTests(unittest.TestCase):
    def test_build_dashboard_payload_uses_existing_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.json"
            results_path.write_text(json.dumps({
                "naive_mse": 0.01,
                "ridge_single_split_mse": 0.005,
                "lasso_single_split_mse": 0.004,
                "ridge_walk_forward_avg_mse": 0.006,
                "arima_rolling_avg_mse": 0.007,
                "gjr_garch_t_weekly_refit_mse": 0.008,
                "config": {"n_lags": 5},
                "timestamp": "2026-01-01T00:00:00"
            }))

            payload = build_dashboard_payload(str(results_path))

            self.assertEqual(payload["summary"]["best_model"], "Custom Lasso GD")
            self.assertEqual(payload["metrics"]["lasso_mse"], 0.004)
            self.assertEqual(payload["config"]["n_lags"], 5)


if __name__ == "__main__":
    unittest.main()
