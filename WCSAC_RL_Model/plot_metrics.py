from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def load_training_metrics(path: Union[str, Path]) -> Dict[str, Any]:
	"""
	Load WCSAC training metrics from JSON safely.

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


def plot_beta_threshold(
	metrics: Dict[str, Any],
	output_dir: Optional[Union[str, Path]] = None,
	service: Optional[str] = None,
	show: bool = False,
) -> List[Path]:
	"""
	Plot beta vs threshold over episodes, with RB allocation and lambda on
	separate panels. This is the clearest indicator of whether the WCSAC
	constraint is being satisfied over training.

	Expected metrics keys (added by trainer.py):
		- episode_mean_beta:  list of mean beta per episode
		- episode_mean_rb:    list of mean RB allocation per episode
		- episode_lambda:     list of Lagrange multiplier per episode
		- beta_threshold:     list (constant) or scalar threshold value
	"""
	if not isinstance(metrics, dict):
		return []

	# ── Extract series ───────────────────────────────────────────────────────
	def _to_array(key: str) -> np.ndarray:
		raw = metrics.get(key, [])
		if not isinstance(raw, list) or len(raw) == 0:
			return np.array([], dtype=np.float64)
		y = []
		for v in raw:
			try:
				y.append(float(v) if v is not None else np.nan)
			except (TypeError, ValueError):
				y.append(np.nan)
		return np.array(y, dtype=np.float64)

	beta_arr   = _to_array("episode_mean_beta")
	rb_arr     = _to_array("episode_mean_rb")
	lambda_arr = _to_array("episode_lambda")
	thresh_raw = metrics.get("beta_threshold", 0.3)

	if len(beta_arr) == 0:
		print("plot_beta_threshold: no episode_mean_beta data found — skipping.")
		return []

	# Threshold may be a list (one per episode) or a scalar
	if isinstance(thresh_raw, list) and len(thresh_raw) > 0:
		try:
			threshold = float(thresh_raw[0])
		except (TypeError, ValueError):
			threshold = 0.3
	else:
		try:
			threshold = float(thresh_raw)
		except (TypeError, ValueError):
			threshold = 0.3

	if output_dir is None:
		out_dir = Path.cwd()
	else:
		out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	service_name = str(service if service else metrics.get("service", "run")).strip().lower()
	if not service_name:
		service_name = "run"

	n = len(beta_arr)
	episodes = np.arange(1, n + 1)

	def smooth(arr: np.ndarray, window: int = 30) -> Tuple[np.ndarray, np.ndarray]:
		"""Return (x, smoothed_y) with valid finite values only."""
		finite_mask = np.isfinite(arr)
		if finite_mask.sum() < window:
			return episodes[finite_mask], arr[finite_mask]
		# Fill NaN with interpolation before smoothing
		filled = arr.copy()
		indices = np.arange(n)
		if not np.all(finite_mask):
			filled = np.interp(indices, indices[finite_mask], arr[finite_mask])
		kernel = np.ones(window) / window
		smoothed = np.convolve(filled, kernel, mode="valid")
		x_smooth = np.arange(window, n + 1)
		return x_smooth, smoothed

	fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
	fig.suptitle(
		f"WCSAC Beta vs QoS Threshold — {service_name.upper()}",
		fontsize=13, fontweight="bold", y=0.98,
	)

	# ── Panel 1: Beta vs Threshold ───────────────────────────────────────────
	ax1 = axes[0]
	valid = np.isfinite(beta_arr)
	ax1.plot(episodes[valid], beta_arr[valid], alpha=0.25, color="steelblue",
			 linewidth=0.7, label="Mean beta (raw)")
	sx, sy = smooth(beta_arr)
	ax1.plot(sx, sy, color="steelblue", linewidth=2.0, label="Mean beta (smoothed)")
	ax1.axhline(threshold, color="crimson", linestyle="--", linewidth=1.5,
				label=f"Threshold = {threshold:.2f}")
	ax1.axhspan(threshold - 0.05, threshold + 0.05, alpha=0.08, color="green",
				label="±0.05 target band")

	# Shade above-threshold region
	ax1.fill_between(
		episodes[valid], beta_arr[valid], threshold,
		where=beta_arr[valid] > threshold,
		alpha=0.12, color="crimson", label="QoS violation region",
	)
	ax1.fill_between(
		episodes[valid], beta_arr[valid], threshold,
		where=beta_arr[valid] <= threshold,
		alpha=0.08, color="green", label="QoS satisfied region",
	)

	# Annotate convergence %
	late_start = int(n * 0.7)
	late_beta = beta_arr[late_start:]
	near = int(np.sum(np.abs(late_beta[np.isfinite(late_beta)] - threshold) <= 0.05))
	total_late = int(np.sum(np.isfinite(late_beta)))
	pct = 100 * near / total_late if total_late > 0 else 0.0
	ax1.text(
		0.98, 0.97,
		f"Last 30%: {pct:.0f}% within ±0.05 of threshold",
		transform=ax1.transAxes, ha="right", va="top", fontsize=9,
		color="green" if pct >= 50 else "crimson",
		bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
				  edgecolor="green" if pct >= 50 else "crimson", alpha=0.8),
	)

	ax1.set_ylabel("Mean Beta per Episode")
	ax1.set_ylim(0, 1)
	ax1.legend(loc="upper right", fontsize=8, ncol=2)
	ax1.grid(True, alpha=0.3)
	ax1.set_title("Beta vs QoS Threshold", fontsize=10)

	# ── Panel 2: RB allocation ───────────────────────────────────────────────
	ax2 = axes[1]
	if len(rb_arr) == n:
		rb_valid = np.isfinite(rb_arr)
		ax2.plot(episodes[rb_valid], rb_arr[rb_valid], alpha=0.25,
				 color="darkorange", linewidth=0.7, label="Mean RB (raw)")
		rsx, rsy = smooth(rb_arr)
		ax2.plot(rsx, rsy, color="darkorange", linewidth=2.0, label="Mean RB (smoothed)")
		ax2.set_ylim(0, 9)
		ax2.set_ylabel("Mean RB Allocated per Episode")
		ax2.legend(loc="upper right", fontsize=8)
	else:
		ax2.text(0.5, 0.5, "No RB data", transform=ax2.transAxes, ha="center")
	ax2.grid(True, alpha=0.3)
	ax2.set_title("Resource Block Allocation", fontsize=10)

	# ── Panel 3: Lagrange multiplier lambda ─────────────────────────────────
	ax3 = axes[2]
	if len(lambda_arr) == n:
		lam_valid = np.isfinite(lambda_arr)
		ax3.plot(episodes[lam_valid], lambda_arr[lam_valid], alpha=0.4,
				 color="purple", linewidth=0.7, label="Lambda (raw)")
		lsx, lsy = smooth(lambda_arr)
		ax3.plot(lsx, lsy, color="purple", linewidth=2.0, label="Lambda (smoothed)")
		ax3.set_ylabel("Lagrange Multiplier (λ)")
		ax3.legend(loc="upper left", fontsize=8)
	else:
		ax3.text(0.5, 0.5, "No lambda data", transform=ax3.transAxes, ha="center")
	ax3.grid(True, alpha=0.3)
	ax3.set_title("Lagrange Multiplier — grows when QoS violated, shrinks when satisfied",
				  fontsize=10)
	ax3.set_xlabel("Episode")

	# ── Save each panel as its own PNG ─────────────────────────────────────
	generated = []

	# Panel 1 — Beta vs Threshold
	p1_path = out_dir / f"wcsac_{service_name}_beta_vs_threshold.png"
	extent1 = ax1.get_window_extent(renderer=fig.canvas.get_renderer())
	extent1 = extent1.transformed(fig.dpi_scale_trans.inverted())
	# Expand to include title and labels
	fig.savefig(p1_path, dpi=150, bbox_inches=ax1.get_tightbbox(
		fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.05, 1.15))
	generated.append(p1_path)
	print(f"plot_beta_threshold: saved {p1_path}")

	# Panel 2 — RB Allocation
	p2_path = out_dir / f"wcsac_{service_name}_rb_allocation.png"
	fig.savefig(p2_path, dpi=150, bbox_inches=ax2.get_tightbbox(
		fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.05, 1.15))
	generated.append(p2_path)
	print(f"plot_beta_threshold: saved {p2_path}")

	# Panel 3 — Lambda
	p3_path = out_dir / f"wcsac_{service_name}_lambda.png"
	fig.savefig(p3_path, dpi=150, bbox_inches=ax3.get_tightbbox(
		fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.05, 1.15))
	generated.append(p3_path)
	print(f"plot_beta_threshold: saved {p3_path}")

	# Also save the combined 3-panel version
	plt.tight_layout(rect=[0, 0, 1, 0.97])
	combined_path = out_dir / f"wcsac_{service_name}_beta_threshold.png"
	plt.savefig(combined_path, dpi=150)
	generated.append(combined_path)
	print(f"plot_beta_threshold: saved {combined_path} (combined)")

	if show:
		plt.show()
	plt.close()

	print(f"plot_beta_threshold: last-30% within ±0.05 = {pct:.0f}%  ({near}/{total_late} episodes)")
	return generated


def plot_beta_per_dti(
	metrics: Dict[str, Any],
	output_dir: Optional[Union[str, Path]] = None,
	service: Optional[str] = None,
	show: bool = False,
) -> List[Path]:
	"""
	Plot beta for every individual DTI step across all episodes.

	This is the primary convergence plot for WCSAC: the goal is not just
	that the *mean* beta per episode reaches the threshold, but that each
	individual DTI allocation satisfies it. This plot shows exactly that.

	Layout (two panels):
	  Top:    Scatter of every DTI beta value vs episode number.
	          Each point = one DTI step. Coloured by DTI position within
	          the episode (DTI 1 dark → DTI 8 light) so you can see if
	          early or late DTIs converge differently.
	          Red dashed line = threshold.
	          Smoothed episode mean overlaid in black.
	          Fraction of DTIs satisfying threshold annotated per region.
	  Bottom: Stacked bar showing per-episode fraction of DTIs above vs
	          below threshold — turns into a solid green bar when converged.
	"""
	if not isinstance(metrics, dict):
		return []

	series = metrics.get("training_episode_dti_series", [])
	if not series:
		print("plot_beta_per_dti: no training_episode_dti_series data — skipping.")
		return []

	# ── Unpack all DTI beta values ───────────────────────────────────────────
	# Build flat arrays: one entry per DTI step
	all_episodes   = []   # episode number for each DTI point
	all_betas      = []   # beta value
	all_dti_pos    = []   # DTI position within episode (0-based)
	episode_means  = []   # (episode, mean_beta) for smoothed overlay
	episode_numbers = []

	for entry in series:
		ep  = int(entry.get("episode", 0))
		bvs = entry.get("beta_values", [])
		valid_bvs = []
		for pos, bv in enumerate(bvs):
			if bv is None:
				continue
			try:
				b = float(bv)
			except (TypeError, ValueError):
				continue
			if not np.isfinite(b):
				continue
			all_episodes.append(ep)
			all_betas.append(b)
			all_dti_pos.append(pos)
			valid_bvs.append(b)
		if valid_bvs:
			episode_means.append(float(np.mean(valid_bvs)))
			episode_numbers.append(ep)

	if not all_betas:
		print("plot_beta_per_dti: all beta values are None/NaN — skipping.")
		return []

	all_episodes  = np.array(all_episodes,  dtype=np.float32)
	all_betas     = np.array(all_betas,     dtype=np.float32)
	all_dti_pos   = np.array(all_dti_pos,   dtype=np.int32)
	episode_means = np.array(episode_means, dtype=np.float32)
	episode_numbers = np.array(episode_numbers, dtype=np.float32)

	# Threshold
	thresh_raw = metrics.get("beta_threshold", 0.3)
	if isinstance(thresh_raw, list) and thresh_raw:
		try:
			threshold = float(thresh_raw[0])
		except (TypeError, ValueError):
			threshold = 0.3
	else:
		try:
			threshold = float(thresh_raw)
		except (TypeError, ValueError):
			threshold = 0.3

	# Output dir
	if output_dir is None:
		out_dir = Path.cwd()
	else:
		out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	service_name = str(service if service else metrics.get("service", "run")).strip().lower()

	# ── Smoothed episode mean ────────────────────────────────────────────────
	def smooth(x, y, window=50):
		if len(y) < window:
			return x, y
		kernel = np.ones(window) / window
		sy = np.convolve(y, kernel, mode="valid")
		sx = x[window - 1:]
		return sx, sy

	smooth_ep, smooth_mean = smooth(episode_numbers, episode_means, window=min(50, len(episode_means)//4 or 1))

	# ── Per-episode fraction satisfying threshold ────────────────────────────
	# For the stacked bar we need one value per episode
	unique_eps = np.array(sorted(set(all_episodes.tolist())))
	ep_frac_ok  = []  # fraction of DTIs <= threshold
	for ep in unique_eps:
		mask = all_episodes == ep
		betas_ep = all_betas[mask]
		ep_frac_ok.append(float(np.mean(betas_ep <= threshold)))
	ep_frac_ok  = np.array(ep_frac_ok)
	ep_frac_bad = 1.0 - ep_frac_ok

	# How many DTIs satisfy threshold overall? Early vs late
	n_eps = len(unique_eps)
	late_start_idx = int(n_eps * 0.7)
	late_mask  = all_episodes >= unique_eps[late_start_idx]
	early_mask = all_episodes < unique_eps[min(late_start_idx, n_eps - 1)]
	pct_late  = 100 * np.mean(all_betas[late_mask]  <= threshold) if late_mask.any()  else 0.0
	pct_early = 100 * np.mean(all_betas[early_mask] <= threshold) if early_mask.any() else 0.0
	pct_all   = 100 * np.mean(all_betas <= threshold)

	# ── Figure ───────────────────────────────────────────────────────────────
	n_dti_positions = int(all_dti_pos.max()) + 1
	cmap = plt.cm.plasma
	dti_colors = [cmap(i / max(n_dti_positions - 1, 1)) for i in range(n_dti_positions)]

	fig, (ax1, ax2) = plt.subplots(
		2, 1, figsize=(14, 10),
		gridspec_kw={"height_ratios": [3, 1]},
		sharex=True,
	)
	fig.suptitle(
		f"WCSAC: Beta per DTI Step — {service_name.upper()}\n"
		f"Goal: every DTI beta ≤ {threshold:.2f} (threshold)",
		fontsize=12, fontweight="bold", y=0.99,
	)

	# ── Panel 1: Scatter of all DTI betas ────────────────────────────────────
	# Plot each DTI position as a separate layer so colours are visible
	for pos in range(n_dti_positions):
		mask = all_dti_pos == pos
		if not mask.any():
			continue
		ax1.scatter(
			all_episodes[mask], all_betas[mask],
			s=2, alpha=0.25, color=dti_colors[pos],
			label=f"DTI {pos + 1}" if n_dti_positions <= 8 else None,
			zorder=2,
		)

	# Smoothed episode mean
	ax1.plot(smooth_ep, smooth_mean, color="black", linewidth=2.0,
			 zorder=5, label="Episode mean (smoothed)")

	# Threshold line
	ax1.axhline(threshold, color="crimson", linestyle="--", linewidth=1.8,
				zorder=6, label=f"Threshold = {threshold:.2f}")

	# Target band
	ax1.axhspan(0, threshold, alpha=0.04, color="green", zorder=1)

	# Annotations
	ax1.text(0.01, 0.97,
			 f"All episodes:  {pct_all:.0f}% of DTIs satisfy threshold",
			 transform=ax1.transAxes, va="top", fontsize=9, color="black")
	ax1.text(0.01, 0.91,
			 f"First 30%:     {pct_early:.0f}% satisfy",
			 transform=ax1.transAxes, va="top", fontsize=9, color="grey")
	ax1.text(0.01, 0.85,
			 f"Last 30%:      {pct_late:.0f}% satisfy",
			 transform=ax1.transAxes, va="top", fontsize=9,
			 color="green" if pct_late >= 60 else "crimson",
			 fontweight="bold")

	ax1.set_ylabel("Beta (per DTI step)")
	ax1.set_ylim(-0.02, 1.05)
	ax1.grid(True, alpha=0.2)
	if n_dti_positions <= 8:
		ax1.legend(loc="upper right", fontsize=8, ncol=3,
					markerscale=3, framealpha=0.9)

	# Colourbar for DTI position
	sm = plt.cm.ScalarMappable(cmap=cmap,
								norm=plt.Normalize(vmin=1, vmax=n_dti_positions))
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax1, pad=0.01, fraction=0.02)
	cbar.set_label("DTI position in episode", fontsize=8)
	cbar.set_ticks([1, n_dti_positions])
	cbar.set_ticklabels(["DTI 1", f"DTI {n_dti_positions}"])

	# ── Panel 2: Stacked bar — fraction satisfying threshold ─────────────────
	# Downsample to at most 300 bars for readability
	stride = max(1, len(unique_eps) // 300)
	bar_eps   = unique_eps[::stride]
	bar_ok    = ep_frac_ok[::stride]
	bar_bad   = ep_frac_bad[::stride]
	bar_width = max(1.0, float(bar_eps[1] - bar_eps[0])) * 0.9 if len(bar_eps) > 1 else 1.0

	ax2.bar(bar_eps, bar_ok,  width=bar_width, color="green",  alpha=0.7, label="DTIs ≤ threshold")
	ax2.bar(bar_eps, bar_bad, width=bar_width, color="crimson", alpha=0.7,
			bottom=bar_ok, label="DTIs > threshold")
	ax2.axhline(1.0, color="green", linestyle=":", linewidth=0.8)
	ax2.set_ylabel("Fraction\nof DTIs", fontsize=9)
	ax2.set_xlabel("Episode")
	ax2.set_ylim(0, 1.05)
	ax2.legend(loc="lower right", fontsize=8)
	ax2.grid(True, alpha=0.2, axis="x")

	# ── Save top scatter panel separately, then save combined ──────────────
	generated_dti = []

	# Top panel only — beta scatter per DTI (the primary convergence plot)
	scatter_path = out_dir / f"wcsac_{service_name}_beta_dti_scatter.png"
	fig.savefig(scatter_path, dpi=150, bbox_inches=ax1.get_tightbbox(
		fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.05, 1.15))
	generated_dti.append(scatter_path)
	print(f"plot_beta_per_dti: saved {scatter_path} (scatter only)")

	# Combined 2-panel version
	plt.tight_layout(rect=[0, 0, 1, 0.97])
	combined_path = out_dir / f"wcsac_{service_name}_beta_per_dti.png"
	plt.savefig(combined_path, dpi=150)
	generated_dti.append(combined_path)
	print(f"plot_beta_per_dti: saved {combined_path} (combined)")

	if show:
		plt.show()
	plt.close()

	print(f"plot_beta_per_dti: {pct_all:.0f}% of all DTI steps satisfy threshold  "
		  f"(early {pct_early:.0f}%  →  late {pct_late:.0f}%)")
	return generated_dti


def plot_evaluation_results(
	metrics: Dict[str, Any],
	output_dir: Optional[Union[str, Path]] = None,
	service: Optional[str] = None,
	show: bool = False,
) -> List[Path]:
	"""
	Plot evaluation results across all evaluation checkpoints during training.

	Produces three separate PNGs:

	1. wcsac_<service>_eval_reward.png
	   Evaluation mean reward at each checkpoint (every evaluation_interval episodes).
	   Shows whether the deterministic policy is improving over training.

	2. wcsac_<service>_eval_beta_vs_threshold.png
	   Mean beta across all DTI steps in each evaluation run, plotted against
	   the threshold. This is the clearest indicator of whether the deterministic
	   policy (no exploration noise) satisfies QoS constraints.

	3. wcsac_<service>_eval_rb.png
	   Mean RB allocation across all DTI steps in each evaluation run.
	   Shows whether the deterministic policy is also being efficient.
	"""
	if not isinstance(metrics, dict):
		return []

	# ── Extract evaluation series ─────────────────────────────────────────────
	eval_rewards_raw   = metrics.get("evaluation_rewards", [])
	eval_beta_raw      = metrics.get("evaluation_mean_beta", [])
	eval_rb_raw        = metrics.get("evaluation_mean_rb", [])

	if not eval_rewards_raw:
		print("plot_evaluation_results: no evaluation_rewards data — skipping.")
		return []

	def unzip_tuples(data):
		"""Convert list of (episode, value) tuples to two arrays."""
		if not data:
			return np.array([]), np.array([])
		eps, vals = zip(*data)
		return np.array(eps, dtype=np.float32), np.array(vals, dtype=np.float32)

	eval_eps,    eval_rew  = unzip_tuples(eval_rewards_raw)
	beta_eps,    eval_beta = unzip_tuples(eval_beta_raw)
	rb_eps,      eval_rb   = unzip_tuples(eval_rb_raw)

	# Threshold
	thresh_raw = metrics.get("beta_threshold", 0.3)
	if isinstance(thresh_raw, list) and thresh_raw:
		try:
			threshold = float(thresh_raw[0])
		except (TypeError, ValueError):
			threshold = 0.3
	else:
		try:
			threshold = float(thresh_raw)
		except (TypeError, ValueError):
			threshold = 0.3

	if output_dir is None:
		out_dir = Path.cwd()
	else:
		out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	service_name = str(service if service else metrics.get("service", "run")).strip().lower()
	generated = []

	# ── Plot 1: Evaluation reward over training ───────────────────────────────
	fig1, ax = plt.subplots(figsize=(10, 5))
	valid = np.isfinite(eval_rew)
	ax.plot(eval_eps[valid], eval_rew[valid], color="teal", linewidth=1.5,
			marker="o", markersize=3, label="Eval mean reward")
	ax.set_xlabel("Training Episode")
	ax.set_ylabel("Mean Reward (deterministic policy)")
	ax.set_title(
		f"WCSAC Evaluation Reward — {service_name.upper()}\n"
		f"Deterministic policy evaluated every {int(eval_eps[1]-eval_eps[0]) if len(eval_eps)>1 else '?'} training episodes"
	)
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=9)
	# Annotate improvement
	if valid.sum() >= 2:
		first_val = float(eval_rew[valid][0])
		last_val  = float(eval_rew[valid][-1])
		best_val  = float(eval_rew[valid].max())
		ax.text(0.98, 0.05,
				f"First: {first_val:.3f}   Best: {best_val:.3f}   Last: {last_val:.3f}",
				transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
				bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="teal", alpha=0.8))
	plt.tight_layout()
	p1 = out_dir / f"wcsac_{service_name}_eval_reward.png"
	fig1.savefig(p1, dpi=150)
	if show: plt.show()
	plt.close(fig1)
	generated.append(p1)
	print(f"plot_evaluation_results: saved {p1}")

	# ── Plot 2: Evaluation beta vs threshold ─────────────────────────────────
	if len(eval_beta) > 0:
		fig2, ax = plt.subplots(figsize=(10, 5))
		valid_b = np.isfinite(eval_beta)
		ax.plot(beta_eps[valid_b], eval_beta[valid_b], color="steelblue",
				linewidth=1.5, marker="o", markersize=3, label="Eval mean beta (all DTIs)")
		ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.5,
				   label=f"Threshold = {threshold:.2f}")
		ax.axhspan(0, threshold, alpha=0.06, color="green")
		ax.axhspan(threshold, 1, alpha=0.06, color="crimson")
		# Fraction of eval checkpoints satisfying threshold
		pct_satisfy = 100 * np.mean(eval_beta[valid_b] <= threshold) if valid_b.any() else 0
		ax.text(0.98, 0.97,
				f"{pct_satisfy:.0f}% of eval checkpoints: mean beta ≤ threshold",
				transform=ax.transAxes, ha="right", va="top", fontsize=9,
				color="green" if pct_satisfy >= 50 else "crimson",
				bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
						  edgecolor="green" if pct_satisfy >= 50 else "crimson", alpha=0.8))
		ax.set_xlabel("Training Episode")
		ax.set_ylabel("Mean Beta across all eval DTI steps")
		ax.set_title(
			f"WCSAC Evaluation Beta vs Threshold — {service_name.upper()}\n"
			f"Deterministic policy — goal: every DTI beta ≤ {threshold:.2f}"
		)
		ax.set_ylim(0, 1)
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=9)
		plt.tight_layout()
		p2 = out_dir / f"wcsac_{service_name}_eval_beta_vs_threshold.png"
		fig2.savefig(p2, dpi=150)
		if show: plt.show()
		plt.close(fig2)
		generated.append(p2)
		print(f"plot_evaluation_results: saved {p2}")

	# ── Plot 3: Evaluation RB allocation ─────────────────────────────────────
	if len(eval_rb) > 0:
		fig3, ax = plt.subplots(figsize=(10, 5))
		valid_r = np.isfinite(eval_rb)
		ax.plot(rb_eps[valid_r], eval_rb[valid_r], color="darkorange",
				linewidth=1.5, marker="o", markersize=3, label="Eval mean RB allocated")
		ax.set_xlabel("Training Episode")
		ax.set_ylabel("Mean RB Allocation (deterministic policy)")
		ax.set_title(
			f"WCSAC Evaluation RB Allocation — {service_name.upper()}\n"
			f"Lower = more efficient (while satisfying QoS)"
		)
		ax.set_ylim(0, 9)
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=9)
		plt.tight_layout()
		p3 = out_dir / f"wcsac_{service_name}_eval_rb.png"
		fig3.savefig(p3, dpi=150)
		if show: plt.show()
		plt.close(fig3)
		generated.append(p3)
		print(f"plot_evaluation_results: saved {p3}")

	return generated


def plot_training_metrics(
	metrics: Dict[str, Any],
	output_dir: Optional[Union[str, Path]] = None,
	service: Optional[str] = None,
	show: bool = False,
) -> List[Path]:
	"""
	Plot advanced WCSAC training metrics and save each figure as PNG.

	Plots:
	- actor_loss
	- critic1_loss
	- critic2_loss
	- q_value_loss
	- alpha
	- risk_alpha (optional for backward compatibility)
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
	found_metrics: List[str] = []
	skipped_metrics: List[str] = []

	# WCSAC metric keys. `risk_alpha` is optional for older history files.
	metric_specs = [
		("actor_loss", "WCSAC Actor Loss", "Loss", f"wcsac_{service_name}_actor_loss.png"),
		("critic1_loss", "WCSAC Critic 1 Loss", "Loss", f"wcsac_{service_name}_critic1_loss.png"),
		("critic2_loss", "WCSAC Critic 2 Loss", "Loss", f"wcsac_{service_name}_critic2_loss.png"),
		("q_value_loss", "WCSAC Q-Value Loss", "Loss", f"wcsac_{service_name}_q_value_loss.png"),
		("alpha", "WCSAC Alpha", "Alpha", f"wcsac_{service_name}_alpha.png"),
		("risk_alpha", "WCSAC Risk Alpha", "Risk Alpha", f"wcsac_{service_name}_risk_alpha.png"),
		("alpha_loss", "WCSAC Alpha Loss", "Loss", f"wcsac_{service_name}_alpha_loss.png"),
		("entropy", "WCSAC Entropy", "Entropy", f"wcsac_{service_name}_entropy.png"),
	]

	for metric_key, title, y_label, filename in metric_specs:
		x, y = _safe_series(metrics.get(metric_key, []))
		if x.size == 0 or y.size == 0:
			skipped_metrics.append(metric_key)
			continue

		save_path = out_dir / filename
		_plot_single_curve(x, y, title, "Episode", y_label, save_path, show)
		found_metrics.append(metric_key)
		generated.append(save_path)

	# Optional diagnostics: useful when metric keys changed or values are empty/NaN.
	print(f"plot_training_metrics: found={found_metrics}")
	if skipped_metrics:
		print(f"plot_training_metrics: skipped={skipped_metrics}")

	return generated


def main() -> None:
	"""CLI entry point for training metrics plotting."""
	parser = argparse.ArgumentParser(description="Plot WCSAC training metrics from JSON")
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
		help="Service name used in output filenames (e.g., voip, cbr)",
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
