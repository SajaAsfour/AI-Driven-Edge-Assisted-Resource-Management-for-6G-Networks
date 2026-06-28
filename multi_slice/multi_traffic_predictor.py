from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from multi_slice.multi_traffic_allocator import AllocationDecision, proportional_allocate_requests
from multi_slice.multi_traffic_config import (
	MultiTrafficPredictionConfig,
	TrafficInputSelection,
	normalize_service_name,
	resolve_final_checkpoint_path,
)
from multi_slice.multi_traffic_plots import save_multi_traffic_plots

if TYPE_CHECKING:
	from WCSAC_RL_Model.env_wrapper import NetworkWCSACEnv


def _json_safe(value: Any) -> Any:
	if isinstance(value, np.ndarray):
		return [_json_safe(v) for v in value.tolist()]
	if isinstance(value, np.generic):
		return _json_safe(value.item())
	if isinstance(value, float):
		if not np.isfinite(value):
			return None
		return float(value)
	if isinstance(value, dict):
		return {str(k): _json_safe(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(v) for v in value]
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, AllocationDecision):
		return asdict(value)
	return value


def _format_service_label(selection: TrafficInputSelection) -> str:
	label = selection.label.strip()
	if label:
		return label
	return selection.service


def _validate_sample_traffic(traffic_by_service: Dict[str, Any]) -> Tuple[int, int]:
	"""Validate per-service traffic matrices from the multi-traffic sample input.

	Returns (num_dtis, num_ttis_per_dti) derived from the matrices themselves.
	"""
	if not traffic_by_service:
		raise ValueError("sample_input['traffic_users_per_tti'] must contain at least one service")

	num_dtis: Optional[int] = None
	num_ttis: Optional[int] = None
	for service, matrix in traffic_by_service.items():
		if not matrix:
			raise ValueError(f"Traffic matrix for service '{service}' must be non-empty")
		if num_dtis is None:
			num_dtis = len(matrix)
		elif len(matrix) != num_dtis:
			raise ValueError(
				f"Service '{service}' has {len(matrix)} DTIs, expected {num_dtis} (all services must match)"
			)
		for row in matrix:
			if not row:
				raise ValueError(f"Traffic matrix for service '{service}' contains an empty DTI row")
			if num_ttis is None:
				num_ttis = len(row)
			elif len(row) != num_ttis:
				raise ValueError(
					f"Service '{service}' has a DTI row with {len(row)} TTIs, expected {num_ttis} "
					"(all rows/services must match)"
				)
			for value in row:
				if isinstance(value, bool) or not isinstance(value, int):
					raise ValueError(f"Traffic value for service '{service}' must be an integer, got {value!r}")
				if value < 0:
					raise ValueError(f"Traffic value for service '{service}' must be >= 0, got {value}")

	return int(num_dtis), int(num_ttis)


def evaluate_allocation(
	env: NetworkWCSACEnv,
	traffic_dti: Sequence[int],
	allocated_rb: int,
) -> Dict[str, Any]:
	traffic_vec = env._normalize_traffic_vector(traffic_dti)
	rb_data = [int(allocated_rb)] * env.n
	env.model.set_service(env.service)
	env.model.set_traffic(traffic_vec)
	env.model.set_resource_blocks(rb_data)
	dti_result = env.model.process_dti(
		traffic_data=traffic_vec,
		rb_data=rb_data,
		c_capacity=env.c,
		rb_used=int(allocated_rb),
		lambda_reward=env.lambda_reward,
	)
	state, reward_current = env.model.to_rl_input(dti_result)
	return {
		"state": state,
		"reward_current": float(reward_current),
		"beta_current": float(dti_result.beta_result.beta_current),
		"beta_cumulative": float(dti_result.beta_result.beta_cumulative),
		"dti_index": int(dti_result.dti_index),
	}


def _get_rb_bounds(env: NetworkWCSACEnv, capacity: int) -> tuple[int, int]:
	rb_min = int(getattr(env, "rb_min", 1))
	rb_max_raw = getattr(env, "rb_max", None)
	rb_max = int(capacity if rb_max_raw is None else rb_max_raw)
	rb_max = min(rb_max, int(capacity))
	if rb_min > rb_max:
		raise ValueError(f"Invalid RB bounds: rb_min={rb_min}, rb_max={rb_max}, capacity={capacity}")
	return rb_min, rb_max


def _evaluate_candidate_beta(
	sweep_env: NetworkWCSACEnv,
	traffic_dti: Sequence[int],
	allocated_rb: int,
) -> Dict[str, Any]:
	traffic_vec = sweep_env._normalize_traffic_vector(traffic_dti)
	rb_value = int(allocated_rb)
	rb_data = [rb_value] * sweep_env.n
	sweep_env.model.reset(sweep_env.service)
	sweep_env.model.set_service(sweep_env.service)
	sweep_env.model.set_traffic(traffic_vec)
	sweep_env.model.set_resource_blocks(rb_data)
	beta_result = sweep_env.model.compute_beta(traffic_vec, rb_data)
	beta_current = float(beta_result.beta_current)
	if hasattr(sweep_env.model, "compute_reward_current"):
		reward_current = float(
			sweep_env.model.compute_reward_current(
				beta_current,
				sweep_env.c,
				rb_value,
				sweep_env.lambda_reward,
			)
		)
	else:
		reward_current = float(-beta_current + sweep_env.lambda_reward * ((sweep_env.c - rb_value) / sweep_env.c))
	return {
		"beta_current": beta_current,
		"reward_current": reward_current,
		"dti_total_failures": float(beta_result.dti_total_failures),
		"dti_total_traffic": int(beta_result.dti_total_traffic),
	}


def _select_threshold_safe_request(
	sweep_env: NetworkWCSACEnv,
	traffic_dti: Sequence[int],
	raw_requested_rb: int,
	beta_threshold: float,
) -> tuple[int, Dict[str, Any], Dict[str, Any], str]:
	"""Mirror WCSAC main.py choice 4: raw prediction -> RB sweep -> threshold-safe request."""
	rb_min, rb_max = _get_rb_bounds(sweep_env, int(getattr(sweep_env, "c", 8)))
	raw_request = int(raw_requested_rb)
	if raw_request < rb_min:
		raw_request = rb_min
	if raw_request > rb_max:
		raw_request = rb_max

	raw_eval = _evaluate_candidate_beta(sweep_env, traffic_dti, raw_request)

	sweep_rows: List[Dict[str, Any]] = []
	for test_rb in range(rb_min, rb_max + 1):
		eval_row = _evaluate_candidate_beta(sweep_env, traffic_dti, test_rb)
		sweep_rows.append(
			{
				"rb": int(test_rb),
				"eval": eval_row,
				"beta_current": float(eval_row["beta_current"]),
			}
		)

	threshold = float(beta_threshold)
	safe_rows = [row for row in sweep_rows if float(row["beta_current"]) <= threshold]
	if safe_rows:
		# Same as WCSAC main.py choice 4: prefer the smallest RB that satisfies beta <= threshold.
		selected = min(safe_rows, key=lambda row: (int(row["rb"]), float(row["beta_current"])))
		reason = "request_beta_within_threshold"
	else:
		# Same fallback as choice 4: choose the RB with the best beta if threshold cannot be satisfied.
		selected = min(sweep_rows, key=lambda row: (float(row["beta_current"]), int(row["rb"])))
		reason = "closest_possible_request_beta"

	selected_rb = int(selected["rb"])
	selected_eval = selected["eval"]
	return selected_rb, selected_eval, raw_eval, reason


@dataclass(slots=True)
class TrafficInputRuntime:
	selection: TrafficInputSelection
	env: Any
	agent: Any
	checkpoint_path: Path
	traffic_matrix: List[List[int]]


@dataclass(slots=True)
class TrafficInputStepLog:
	input_label: str
	service: str
	profile_name: str
	traffic_dti: List[int]
	raw_requested_rb: int
	raw_beta_current: float
	requested_rb: int
	request_beta_current: float
	request_selection_reason: str
	allocated_rb: int
	beta_current: float
	reward: float


@dataclass(slots=True)
class TrafficStepLog:
	dti_index: int
	total_requested_rb: int
	total_allocated_rb: int
	capacity: int
	scaling_applied: bool
	beta_threshold: float
	inputs: List[TrafficInputStepLog]


@dataclass(slots=True)
class MultiTrafficPredictionResult:
	config: Dict[str, Any]
	inputs: List[Dict[str, Any]]
	steps: List[TrafficStepLog]
	output_path: Path
	log_path: Path
	plot_paths: List[Path] = field(default_factory=list)

	def to_dict(self) -> Dict[str, Any]:
		return _json_safe(asdict(self))


class MultiTrafficPredictor:
	"""Run multiple trained WCSAC agents in evaluation-only mode and allocate RBs globally."""

	def __init__(self, config: MultiTrafficPredictionConfig, logger: Optional[logging.Logger] = None) -> None:
		self.config = config.normalized()
		self.logger = logger or logging.getLogger("multi_traffic_prediction")
		self.base_checkpoint_dir = Path(self.config.checkpoint_base_dir)
		self.base_checkpoint_dir.mkdir(parents=True, exist_ok=True)
		self.output_dir = Path(self.config.output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)

		if self.config.capacity <= 0:
			raise ValueError("capacity must be > 0")
		if self.config.beta_threshold < 0:
			raise ValueError("beta_threshold must be >= 0")

	def _load_model_components(self) -> tuple[Any, Any]:
		from WCSAC_RL_Model.agent import WCSACAgent
		from WCSAC_RL_Model.env_wrapper import NetworkWCSACEnv

		return NetworkWCSACEnv, WCSACAgent

	def _build_runtime_input(
		self,
		selection: TrafficInputSelection,
		label: str,
		seed_offset: int,
		traffic_matrix: List[List[int]],
	) -> TrafficInputRuntime:
		service = normalize_service_name(selection.service)
		checkpoint_path = resolve_final_checkpoint_path(self.base_checkpoint_dir, service, self.config.model_name)
		env_class, agent_class = self._load_model_components()

		# Traffic comes entirely from config.sample_input below; the env's own fixed-profile
		# traffic generation is unused (infer_rb_from_traffic/evaluate_allocation take traffic explicitly).
		env = env_class(
			service=service,
			rb_min=1,
			seed=self.config.seed + seed_offset,
			silent=True,
		)

		state_dim = int(np.prod(env.observation_shape))
		action_dim = int(np.prod(env.action_shape))
		agent = agent_class(state_dim=state_dim, action_dim=action_dim)
		agent.load(checkpoint_path)

		return TrafficInputRuntime(
			selection=TrafficInputSelection(service=service, profile_name=selection.profile_name, label=label),
			env=env,
			agent=agent,
			checkpoint_path=checkpoint_path,
			traffic_matrix=traffic_matrix,
		)

	def _build_sweep_env(self, runtime: TrafficInputRuntime, seed_offset: int) -> Any:
		env_class, _ = self._load_model_components()
		return env_class(
			service=runtime.selection.service,
			rb_min=int(getattr(runtime.env, "rb_min", 1)),
			seed=self.config.seed + seed_offset,
			silent=True,
		)

	def run(self) -> MultiTrafficPredictionResult:
		traffic_by_service = self.config.sample_input["traffic_users_per_tti"]
		num_inputs = len(traffic_by_service)
		service_names = list(traffic_by_service.keys())
		num_dtis, num_ttis_per_dti = _validate_sample_traffic(traffic_by_service)

		runtimes: List[TrafficInputRuntime] = []
		sweep_envs: List[Any] = []
		for idx, selection in enumerate(self.config.inputs):
			label = selection.label or f"input_{idx + 1}"
			if selection.service not in traffic_by_service:
				raise ValueError(
					f"No traffic matrix found for service '{selection.service}' in sample_input"
				)
			traffic_matrix = traffic_by_service[selection.service]
			runtime = self._build_runtime_input(selection, label, seed_offset=idx, traffic_matrix=traffic_matrix)
			runtimes.append(runtime)
			sweep_envs.append(self._build_sweep_env(runtime, seed_offset=1000 + idx))

		base_n = runtimes[0].env.n
		for runtime in runtimes[1:]:
			if runtime.env.n != base_n:
				raise ValueError("All inputs must share the same number of TTIs per DTI (n)")

		for runtime in runtimes:
			runtime.env.model.reset(runtime.env.service)

		for runtime in runtimes:
			self.logger.info(
				"Selected %s: service=%s, profile=%s",
				runtime.selection.label,
				runtime.selection.service,
				runtime.selection.profile_name,
			)
		for runtime in runtimes:
			self.logger.info("Checkpoint %s: %s", runtime.selection.label, runtime.checkpoint_path)
		self.logger.info("Global RB capacity: %s", self.config.capacity)
		self.logger.info("Beta threshold: %.6f", self.config.beta_threshold)
		self.logger.info("Number of DTIs: %s", num_dtis)

		step_logs: List[TrafficStepLog] = []

		for dti_index in range(num_dtis):
			display_dti = dti_index + 1
			traffics = [runtime.traffic_matrix[dti_index] for runtime in runtimes]

			self.logger.info(
				"DTI %s | traffic TTIs %s",
				display_dti,
				{runtime.selection.label: traffic for runtime, traffic in zip(runtimes, traffics)},
			)

			raw_requests: List[int] = []
			raw_evals: List[Dict[str, Any]] = []
			requested: List[int] = []
			request_evals: List[Dict[str, Any]] = []
			reasons: List[str] = []

			for runtime, sweep_env, traffic in zip(runtimes, sweep_envs, traffics):
				raw_requested_rb = int(runtime.env.infer_rb_from_traffic(
					agent=runtime.agent,
					traffic_dti=traffic,
					dti_index=dti_index,
				))

				# The multi allocator receives the same threshold-safe requests produced by WCSAC main.py choice 4.
				requested_rb, request_eval, raw_eval, reason = _select_threshold_safe_request(
					sweep_env=sweep_env,
					traffic_dti=traffic,
					raw_requested_rb=raw_requested_rb,
					beta_threshold=self.config.beta_threshold,
				)

				raw_requests.append(raw_requested_rb)
				raw_evals.append(raw_eval)
				requested.append(requested_rb)
				request_evals.append(request_eval)
				reasons.append(reason)

			allocation = proportional_allocate_requests(
				requested,
				capacity=self.config.capacity,
				min_rb=1,
			)

			input_logs: List[TrafficInputStepLog] = []
			for idx, runtime in enumerate(runtimes):
				# Use process_dti only for the final allocation so cumulative beta state is not polluted by request sweeps.
				eval_result = evaluate_allocation(runtime.env, traffics[idx], allocation.allocations[idx])
				input_logs.append(
					TrafficInputStepLog(
						input_label=runtime.selection.label,
						service=runtime.selection.service,
						profile_name=runtime.selection.profile_name,
						traffic_dti=list(traffics[idx]),
						raw_requested_rb=int(raw_requests[idx]),
						raw_beta_current=float(raw_evals[idx]["beta_current"]),
						requested_rb=int(requested[idx]),
						request_beta_current=float(request_evals[idx]["beta_current"]),
						request_selection_reason=reasons[idx],
						allocated_rb=int(allocation.allocations[idx]),
						beta_current=float(eval_result["beta_current"]),
						reward=float(eval_result["reward_current"]),
					)
				)

			step_log = TrafficStepLog(
				dti_index=display_dti,
				total_requested_rb=int(allocation.total_requested),
				total_allocated_rb=int(allocation.total_allocated),
				capacity=int(allocation.capacity),
				scaling_applied=bool(allocation.scaling_applied),
				beta_threshold=float(self.config.beta_threshold),
				inputs=input_logs,
			)
			step_logs.append(step_log)

			self.logger.info(
				"DTI %s | total_requested=%s total_allocated=%s scaled=%s | %s",
				display_dti,
				step_log.total_requested_rb,
				step_log.total_allocated_rb,
				step_log.scaling_applied,
				" | ".join(
					f"{log.input_label}: req={log.requested_rb} alloc={log.allocated_rb} "
					f"beta={log.beta_current:.4f} reward={log.reward:.4f}"
					for log in input_logs
				),
			)

		result = MultiTrafficPredictionResult(
			config={
				"capacity": self.config.capacity,
				"beta_threshold": self.config.beta_threshold,
				"seed": self.config.seed,
				"model_name": self.config.model_name or "wcsac",
				"traffic_source": "multi_traffic_config_sample_input",
				"num_inputs": num_inputs,
				"service_names": service_names,
				"num_dtis": num_dtis,
				"num_ttis_per_dti": num_ttis_per_dti,
			},
			inputs=[
				{
					"label": _format_service_label(runtime.selection),
					"service": runtime.selection.service,
					"profile_name": runtime.selection.profile_name,
					"checkpoint_path": runtime.checkpoint_path,
				}
				for runtime in runtimes
			],
			steps=step_logs,
			output_path=self.output_dir / "wcsac_multi_traffic_prediction_output.json",
			log_path=self.output_dir / "wcsac_multi_traffic_prediction.log",
		)

		result.plot_paths = save_multi_traffic_plots(result)
		result.output_path.write_text(json.dumps(_json_safe(result.to_dict()), indent=2), encoding="utf-8")
		return result
