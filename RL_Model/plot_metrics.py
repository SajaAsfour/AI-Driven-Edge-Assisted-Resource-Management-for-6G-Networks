from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def load_training_metrics(path: Union[str, Path]) -> Dict[str, Any]:
	"""
	Load SAC training metrics from JSON safely.

	Args:
		path: Path to `training_metrics.json`.

	Returns:
		Parsed metrics dictionary.

	Raises:
		FileNotFoundError: If file does not exist.
		ValueError: If JSON is invalid or does not contain an object.
	"""
	metrics_path = Path(path)
	if not metrics_path.exists() or not metrics_path.is_file():
		raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

	try:
		data = json.loads(metrics_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		raise ValueError(f"Invalid JSON in metrics file: {metrics_path}") from exc

	if not isinstance(data, dict):
		raise ValueError("Metrics JSON root must be an object/dict")

	return data


def _safe_series(values: Any) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Convert a list-like sequence to plottable x/y arrays.

	- Handles None/null safely
	- Converts invalid entries to NaN
	- Drops NaN values before plotting
	"""
	if not isinstance(values, list):
		return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

	y_raw: List[float] = []
	for item in values:
		if item is None:
			y_raw.append(np.nan)
		else:
			try:
				y_raw.append(float(item))
			except (TypeError, ValueError):
				y_raw.append(np.nan)

	y = np.asarray(y_raw, dtype=np.float64)
	x = np.arange(1, len(y) + 1, dtype=np.int32)
	valid_mask = np.isfinite(y)
	return x[valid_mask], y[valid_mask]


def _plot_single_curve(
	x: np.ndarray,
	y: np.ndarray,
	title: str,
	xlabel: str,
	ylabel: str,
	save_path: Path,
	show: bool,
) -> None:
	"""Create and save one simple line plot."""
	if x.size == 0 or y.size == 0:
		return

	plt.figure(figsize=(8, 5))
	plt.plot(x, y)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.tight_layout()
	plt.savefig(save_path, dpi=150)

	if show:
		plt.show()

	plt.close()


def plot_training_metrics(
	metrics: Dict[str, Any],
	output_dir: Optional[Union[str, Path]] = None,
	service: Optional[str] = None,
	show: bool = False,
) -> List[Path]:
	"""
	Plot advanced SAC training metrics and save each figure as PNG.

	Plots:
	- actor_loss
	- critic1_loss
	- critic2_loss
	- q_value_loss
	- alpha
	- alpha_loss
	- entropy

	Args:
		metrics: Loaded metrics dictionary.
		output_dir: Directory to write PNG files.
		service: Service name for output file naming.
		show: Whether to display figures interactively.

	Returns:
		List of generated PNG paths.
	"""
	if not isinstance(metrics, dict):
		raise ValueError("metrics must be a dictionary")

	if output_dir is None:
		out_dir = Path.cwd()
	else:
		out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	service_name = str(service if service else metrics.get("service", "run")).strip().lower()
	if not service_name:
		service_name = "run"

	generated: List[Path] = []
	metric_specs = [
		("actor_loss", "Actor Loss", "Loss", f"sac_{service_name}_actor_loss.png"),
		("critic1_loss", "Critic 1 Loss", "Loss", f"sac_{service_name}_critic1_loss.png"),
		("critic2_loss", "Critic 2 Loss", "Loss", f"sac_{service_name}_critic2_loss.png"),
		("q_value_loss", "Q-Value Loss", "Loss", f"sac_{service_name}_q_value_loss.png"),
		("alpha", "Alpha", "Alpha", f"sac_{service_name}_alpha.png"),
		("alpha_loss", "Alpha Loss", "Loss", f"sac_{service_name}_alpha_loss.png"),
		("entropy", "Entropy", "Entropy", f"sac_{service_name}_entropy.png"),
	]

	for metric_key, title, y_label, filename in metric_specs:
		x, y = _safe_series(metrics.get(metric_key, []))
		save_path = out_dir / filename
		_plot_single_curve(x, y, title, "Episode", y_label, save_path, show)
		if save_path.exists():
			generated.append(save_path)

	return generated


def main() -> None:
	"""CLI entry point for training metrics plotting."""
	parser = argparse.ArgumentParser(description="Plot SAC training metrics from JSON")
	parser.add_argument(
		"--metrics",
		required=True,
		help="Path to training_metrics.json",
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Directory to save PNG files (default: next to metrics file)",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display plots interactively",
	)
	parser.add_argument(
		"--service",
		default=None,
		help="Service name used in output filenames (e.g., voip, cbr, streaming)",
	)

	args = parser.parse_args()

	metrics_path = Path(args.metrics)
	metrics = load_training_metrics(metrics_path)

	# If output directory is not provided, save beside the metrics file.
	output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent

	generated = plot_training_metrics(
		metrics=metrics,
		output_dir=output_dir,
		service=args.service,
		show=args.show,
	)
	if not generated:
		print("No plots were generated (metrics may be empty or invalid).")
		return

	print("Generated plot files:")
	for path in generated:
		print(f"- {path}")


if __name__ == "__main__":
	main()
