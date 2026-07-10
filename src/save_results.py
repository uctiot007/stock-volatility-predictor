import json
import os
from datetime import datetime


def save_results(results_dict: dict, path: str = "outputs/results.json") -> None:
    """
    Save a dictionary of final results/metrics to a JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    results_dict = dict(results_dict)
    results_dict["timestamp"] = datetime.now().isoformat()

    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"Results saved to {path}")


def load_results(path: str = "outputs/results.json") -> dict:
    """
    Load previously saved results.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m main` first to generate results."
        )
    with open(path, "r") as f:
        return json.load(f)