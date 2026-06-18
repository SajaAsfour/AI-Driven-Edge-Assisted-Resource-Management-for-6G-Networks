from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from SAC_RL_Model.traffic_profiles import build_traffic_matrix_from_profile, get_default_profiles, get_profile_or_raise

from multi_slice.multi_traffic_allocator import AllocationDecision, proportional_allocate_two_requests
from multi_slice.multi_traffic_config import (
	MultiTrafficPredictionConfig,
	TrafficInputSelection,
	normalize_service_name,
	resolve_final_checkpoint_path,
)

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
	profile_values: List[int]
	traffic_matrix: List[List[int]]


@dataclass(slots=True)
class TrafficStepLog:
	dti_index: int
	traffic_dti_1: List[int]
	traffic_dti_2: List[int]
	raw_requested_rb_1: int
	raw_requested_rb_2: int
	raw_beta_current_1: float
	raw_beta_current_2: float
	requested_rb_1: int
	requested_rb_2: int
	request_beta_current_1: float
	request_beta_current_2: float
	request_selection_reason_1: str
	request_selection_reason_2: str
	total_requested_rb: int
	scaling_applied: bool
	allocated_rb_1: int
	allocated_rb_2: int
	beta_threshold: float
	beta_current_1: float
	beta_current_2: float
	reward_1: float
	reward_2: float


@dataclass(slots=True)
class MultiTrafficPredictionResult:
	config: Dict[str, Any]
	inputs: List[Dict[str, Any]]
	steps: List[TrafficStepLog]
	output_path: Path
	log_path: Path

	def to_dict(self) -> Dict[str, Any]:
		return _json_safe(asdict(self))


class MultiTrafficPredictor:
	"""Run two trained WCSAC agents in evaluation-only mode and allocate RBs globally."""

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

	def _build_runtime_input(self, selection: TrafficInputSelection, label: str, seed_offset: int) -> TrafficInputRuntime:
		service = normalize_service_name(selection.service)
		profile_name = str(selection.profile_name).strip().lower()
		profiles = get_default_profiles()
		_, profile_values = get_profile_or_raise(profiles, profile_name)

		checkpoint_path = resolve_final_checkpoint_path(self.base_checkpoint_dir, service, self.config.model_name)
		env_class, agent_class = self._load_model_components()

		env = env_class(
			service=service,
			traffic_profile_mode="fixed",
			fixed_profile_name=profile_name,
			rb_min=1,
			seed=self.config.seed + seed_offset,
			silent=True,
		)

		state_dim = int(np.prod(env.observation_shape))
		action_dim = int(np.prod(env.action_shape))
		agent = agent_class(state_dim=state_dim, action_dim=action_dim)
		agent.load(checkpoint_path)

		rng = random.Random(self.config.seed + seed_offset)
		traffic_matrix = build_traffic_matrix_from_profile(profile_values=profile_values, m=env.m, n=env.n, rng=rng)

		return TrafficInputRuntime(
			selection=TrafficInputSelection(service=service, profile_name=profile_name, label=label),
			env=env,
			agent=agent,
			checkpoint_path=checkpoint_path,
			profile_values=profile_values,
			traffic_matrix=traffic_matrix,
		)

	def _build_sweep_env(self, runtime: TrafficInputRuntime, seed_offset: int) -> Any:
		env_class, _ = self._load_model_components()
		return env_class(
			service=runtime.selection.service,
			traffic_profile_mode="fixed",
			fixed_profile_name=runtime.selection.profile_name,
			rb_min=int(getattr(runtime.env, "rb_min", 1)),
			seed=self.config.seed + seed_offset,
			silent=True,
		)

	def run(self) -> MultiTrafficPredictionResult:
		runtime_1 = self._build_runtime_input(self.config.input_1, "input_1", seed_offset=0)
		runtime_2 = self._build_runtime_input(self.config.input_2, "input_2", seed_offset=1)
		sweep_env_1 = self._build_sweep_env(runtime_1, seed_offset=1000)
		sweep_env_2 = self._build_sweep_env(runtime_2, seed_offset=1001)

		if runtime_1.env.m != runtime_2.env.m or runtime_1.env.n != runtime_2.env.n:
			raise ValueError("Both inputs must share the same network dimensions (n and m)")

		runtime_1.env.model.reset(runtime_1.env.service)
		runtime_2.env.model.reset(runtime_2.env.service)

		self.logger.info("Selected input 1: service=%s, profile=%s", runtime_1.selection.service, runtime_1.selection.profile_name)
		self.logger.info("Selected input 2: service=%s, profile=%s", runtime_2.selection.service, runtime_2.selection.profile_name)
		self.logger.info("Checkpoint input 1: %s", runtime_1.checkpoint_path)
		self.logger.info("Checkpoint input 2: %s", runtime_2.checkpoint_path)
		self.logger.info("Global RB capacity: %s", self.config.capacity)
		self.logger.info("Beta threshold: %.6f", self.config.beta_threshold)

		step_logs: List[TrafficStepLog] = []

		for dti_index in range(runtime_1.env.m):
			display_dti = dti_index + 1
			traffic_1 = runtime_1.traffic_matrix[dti_index]
			traffic_2 = runtime_2.traffic_matrix[dti_index]

			self.logger.info(
				"DTI %s | traffic TTIs input_1=%s | input_2=%s",
				display_dti,
				traffic_1,
				traffic_2,
			)

			raw_requested_1 = int(runtime_1.env.infer_rb_from_traffic(
				agent=runtime_1.agent,
				traffic_dti=traffic_1,
				dti_index=dti_index,
			))
			raw_requested_2 = int(runtime_2.env.infer_rb_from_traffic(
				agent=runtime_2.agent,
				traffic_dti=traffic_2,
				dti_index=dti_index,
			))

			requested_1, request_eval_1, raw_eval_1, reason_1 = _select_threshold_safe_request(
				sweep_env=sweep_env_1,
				traffic_dti=traffic_1,
				raw_requested_rb=raw_requested_1,
				beta_threshold=self.config.beta_threshold,
			)
			requested_2, request_eval_2, raw_eval_2, reason_2 = _select_threshold_safe_request(
				sweep_env=sweep_env_2,
				traffic_dti=traffic_2,
				raw_requested_rb=raw_requested_2,
				beta_threshold=self.config.beta_threshold,
			)

			# The multi allocator now receives the same threshold-safe requests produced by WCSAC main.py choice 4.
			allocation = proportional_allocate_two_requests(
				requested_1,
				requested_2,
				capacity=self.config.capacity,
				min_rb=1,
			)

			# Use process_dti only for the final allocation so cumulative beta state is not polluted by request sweeps.
			eval_1 = evaluate_allocation(runtime_1.env, traffic_1, allocation.allocation_1)
			eval_2 = evaluate_allocation(runtime_2.env, traffic_2, allocation.allocation_2)

			beta_1 = float(eval_1["beta_current"])
			beta_2 = float(eval_2["beta_current"])
			threshold = float(self.config.beta_threshold)

			step_log = TrafficStepLog(
				dti_index=display_dti,
				traffic_dti_1=list(traffic_1),
				traffic_dti_2=list(traffic_2),
				raw_requested_rb_1=int(raw_requested_1),
				raw_requested_rb_2=int(raw_requested_2),
				raw_beta_current_1=float(raw_eval_1["beta_current"]),
				raw_beta_current_2=float(raw_eval_2["beta_current"]),
				requested_rb_1=int(requested_1),
				requested_rb_2=int(requested_2),
				request_beta_current_1=float(request_eval_1["beta_current"]),
				request_beta_current_2=float(request_eval_2["beta_current"]),
				request_selection_reason_1=reason_1,
				request_selection_reason_2=reason_2,
				total_requested_rb=allocation.total_requested,
				scaling_applied=allocation.scaling_applied,
				allocated_rb_1=allocation.allocation_1,
				allocated_rb_2=allocation.allocation_2,
				beta_threshold=threshold,
				beta_current_1=beta_1,
				beta_current_2=beta_2,
				reward_1=float(eval_1["reward_current"]),
				reward_2=float(eval_2["reward_current"]),
			)
			step_logs.append(step_log)

			self.logger.info(
				"DTI %s | request_1=%s request_2=%s total=%s | request_beta_1=%.4f request_beta_2=%.4f threshold=%.4f | alloc_1=%s alloc_2=%s scaled=%s | final_beta_1=%.4f final_beta_2=%.4f | reward_1=%.4f reward_2=%.4f",
				display_dti,
				requested_1,
				requested_2,
				allocation.total_requested,
				step_log.request_beta_current_1,
				step_log.request_beta_current_2,
				threshold,
				allocation.allocation_1,
				allocation.allocation_2,
				allocation.scaling_applied,
				step_log.beta_current_1,
				step_log.beta_current_2,
				step_log.reward_1,
				step_log.reward_2,
			)

		result = MultiTrafficPredictionResult(
			config={
				"capacity": self.config.capacity,
				"beta_threshold": self.config.beta_threshold,
				"seed": self.config.seed,
				"model_name": self.config.model_name or "wcsac",
			},
			inputs=[
				{
					"label": _format_service_label(runtime_1.selection),
					"service": runtime_1.selection.service,
					"profile_name": runtime_1.selection.profile_name,
					"checkpoint_path": runtime_1.checkpoint_path,
					"profile_values": runtime_1.profile_values,
				},
				{
					"label": _format_service_label(runtime_2.selection),
					"service": runtime_2.selection.service,
					"profile_name": runtime_2.selection.profile_name,
					"checkpoint_path": runtime_2.checkpoint_path,
					"profile_values": runtime_2.profile_values,
				},
			],
			steps=step_logs,
			output_path=self.output_dir / "wcsac_multi_traffic_prediction_output.json",
			log_path=self.output_dir / "wcsac_multi_traffic_prediction.log",
		)

		result.output_path.write_text(json.dumps(_json_safe(result.to_dict()), indent=2), encoding="utf-8")
		return result
