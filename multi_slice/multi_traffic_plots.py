from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
	from multi_slice.multi_traffic_predictor import MultiTrafficPredictionResult


def _safe_float(value: Any) -> float:
	"""Convert a value to a finite float, or NaN if it cannot be plotted safely."""
	try:
		converted = float(value)
	except (TypeError, ValueError):
		return float("nan")
	return converted if np.isfinite(converted) else float("nan")


def _assign_input_colors(labels: List[str]) -> Dict[str, Any]:
	"""Assign one unique, stable color per input label."""
	cmap = plt.get_cmap("tab10") if len(labels) <= 10 else plt.get_cmap("tab20")
	return {label: cmap(idx % cmap.N) for idx, label in enumerate(labels)}


def _legend_text(label: str, service: str, profile_name: str, profile_values: Any) -> str:
	return f"{label} ({service.upper()}, {profile_name}, {list(profile_values) if profile_values is not None else 'n/a'})"


def _collect_traffic_points(result: "MultiTrafficPredictionResult", labels: List[str]) -> Dict[str, Dict[str, List[float]]]:
	"""Build per-label (x, y) scatter points from the actual per-TTI traffic values in each DTI."""
	points: Dict[str, Dict[str, List[float]]] = {label: {"x": [], "y": []} for label in labels}
	for step in result.steps:
		present = {input_log.input_label: input_log for input_log in step.inputs}
		for label in labels:
			input_log = present.get(label)
			if input_log is None:
				continue
			traffic_dti = input_log.traffic_dti
			count = len(traffic_dti)
			if count == 0:
				continue
			offsets = np.linspace(-0.3, 0.3, count) if count > 1 else np.array([0.0])
			for offset, raw_value in zip(offsets, traffic_dti):
				value = _safe_float(raw_value)
				if not np.isfinite(value):
					continue
				points[label]["x"].append(float(step.dti_index) + float(offset))
				points[label]["y"].append(value)
	return points


def _collect_alloc_beta_series(result: "MultiTrafficPredictionResult", labels: List[str]) -> Dict[str, Dict[str, List[float]]]:
	"""Build per-label allocated-RB/beta series aligned to result.steps order."""
	series: Dict[str, Dict[str, List[float]]] = {label: {"allocated": [], "beta": []} for label in labels}
	for step in result.steps:
		present = {input_log.input_label: input_log for input_log in step.inputs}
		for label in labels:
			input_log = present.get(label)
			if input_log is None:
				series[label]["allocated"].append(float("nan"))
				series[label]["beta"].append(float("nan"))
				continue
			series[label]["allocated"].append(_safe_float(input_log.allocated_rb))
			series[label]["beta"].append(_safe_float(input_log.beta_current))
	return series


def _save_combined_figure(
	output_dir: Path,
	dti_indices: List[int],
	labels: List[str],
	legend_text_by_label: Dict[str, str],
	colors: Dict[str, Any],
	traffic_points: Dict[str, Dict[str, List[float]]],
	series: Dict[str, Dict[str, List[float]]],
	beta_threshold: float,
) -> Path:
	"""Save one figure with traffic / allocated RB / beta side by side, all inputs overlaid."""
	fig, axes = plt.subplots(1, 3, figsize=(18, 5))
	traffic_ax, alloc_ax, beta_ax = axes

	for label in labels:
		color = colors[label]
		legend_label = legend_text_by_label[label]
		traffic_ax.plot(
			traffic_points[label]["x"], traffic_points[label]["y"],
			color=color, linestyle="None", marker=".", markersize=4, label=legend_label,
		)
		alloc_ax.plot(dti_indices, series[label]["allocated"], color=color, marker="o", markersize=3, label=legend_label)
		beta_ax.plot(dti_indices, series[label]["beta"], color=color, marker="o", markersize=3, label=legend_label)

	traffic_ax.set_title("Traffic Profile per DTI")
	traffic_ax.set_xlabel("DTI Index")
	traffic_ax.set_ylabel("Traffic Profile Values")
	traffic_ax.legend(fontsize="small")

	alloc_ax.set_title("Allocated RB per DTI")
	alloc_ax.set_xlabel("DTI Index")
	alloc_ax.set_ylabel("Allocated RB")
	alloc_ax.legend(fontsize="small")

	beta_ax.axhline(beta_threshold, color="black", linestyle="--", linewidth=1, label="beta_threshold")
	beta_ax.set_title("Final Beta per DTI")
	beta_ax.set_xlabel("DTI Index")
	beta_ax.set_ylabel("Beta")
	beta_ax.legend(fontsize="small")

	fig.suptitle(f"Multi-Traffic Inputs — Allocation & Beta (beta_threshold={beta_threshold:.4f})")
	fig.tight_layout()

	out_path = output_dir / "multi_traffic_inputs_alloc_beta.png"
	fig.savefig(out_path)
	plt.close(fig)
	return out_path


def _save_total_rb_figure(output_dir: Path, dti_indices: List[int], total_allocated: List[float], capacity: float) -> Path:
	"""Save total allocated RB vs DTI index, with a dashed capacity line."""
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(dti_indices, total_allocated, color="tab:blue", marker="o", markersize=3, label="Total Allocated RB")
	ax.axhline(capacity, color="black", linestyle="--", linewidth=1, label="Capacity")
	ax.set_title(f"Total Allocated RB vs Capacity (capacity={capacity:.0f})")
	ax.set_xlabel("DTI Index")
	ax.set_ylabel("Total Allocated RB")
	ax.legend(fontsize="small")
	fig.tight_layout()

	out_path = output_dir / "multi_traffic_total_rb_vs_capacity.png"
	fig.savefig(out_path)
	plt.close(fig)
	return out_path


def save_multi_traffic_plots(result: "MultiTrafficPredictionResult") -> List[Path]:
	"""Generate and save the multi-traffic plots: one combined all-inputs figure plus total RB.

	Returns the list of saved plot file paths. Safe against missing/NaN values:
	a single bad data point becomes a gap in the line rather than a crash.
	"""
	if not result.steps:
		return []

	output_dir = result.output_path.parent
	output_dir.mkdir(parents=True, exist_ok=True)

	first_step = result.steps[0]
	labels = [input_log.input_label for input_log in first_step.inputs]
	services = {input_log.input_label: input_log.service for input_log in first_step.inputs}
	profiles = {input_log.input_label: input_log.profile_name for input_log in first_step.inputs}
	profile_values_by_label = {entry["label"]: entry.get("profile_values") for entry in result.inputs}
	colors = _assign_input_colors(labels)
	legend_text_by_label = {
		label: _legend_text(label, services[label], profiles[label], profile_values_by_label.get(label))
		for label in labels
	}

	dti_indices = [int(step.dti_index) for step in result.steps]
	beta_threshold = _safe_float(first_step.beta_threshold)
	capacity = _safe_float(first_step.capacity)
	total_allocated = [_safe_float(step.total_allocated_rb) for step in result.steps]

	traffic_points = _collect_traffic_points(result, labels)
	series = _collect_alloc_beta_series(result, labels)

	plot_paths: List[Path] = [
		_save_combined_figure(
			output_dir, dti_indices, labels, legend_text_by_label, colors, traffic_points, series, beta_threshold
		),
		_save_total_rb_figure(output_dir, dti_indices, total_allocated, capacity),
	]

	return plot_paths
