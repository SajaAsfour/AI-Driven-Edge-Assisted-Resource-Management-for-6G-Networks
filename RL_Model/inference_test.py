from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
	from .agent import SACAgent
	from .env_wrapper import NetworkSACEnv
except ImportError:
	from agent import SACAgent
	from env_wrapper import NetworkSACEnv


Number = Union[int, float, np.integer, np.floating]
SERVICE_ORDER: Tuple[str, str, str] = ("voip", "cbr", "streaming")


def _validate_number(name: str, value: Any) -> float:
	"""Validate one numeric value and return float."""
	if not isinstance(value, (int, float, np.integer, np.floating)):
		raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
	out = float(value)
	if not np.isfinite(out):
		raise ValueError(f"{name} must be finite, got {value!r}")
	if out < 0.0:
		raise ValueError(f"{name} must be >= 0, got {value!r}")
	return out


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
	"""Load JSON dictionary if file exists, otherwise return empty dictionary."""
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as f:
		loaded = json.load(f)
	if not isinstance(loaded, dict):
		raise ValueError(f"Invalid config JSON at {path}: expected object at top-level")
	return loaded


def _build_agent_from_config(
	state_dim: int,
	action_dim: int,
	agent_cfg: Dict[str, Any],
) -> SACAgent:
	"""Create SACAgent using optional hyperparameters from sidecar config JSON."""
	kwargs: Dict[str, Any] = {
		"state_dim": state_dim,
		"action_dim": action_dim,
	}

	# Missing/None values are skipped so SACAgent defaults apply.
	if agent_cfg.get("hidden_dims") is not None:
		hidden_dims = tuple(int(x) for x in agent_cfg["hidden_dims"])
		kwargs["hidden_dims"] = hidden_dims
	for key in ("gamma", "tau", "actor_lr", "critic_lr", "alpha_lr", "target_entropy", "device"):
		value = agent_cfg.get(key)
		if value is not None:
			kwargs[key] = value

	return SACAgent(**kwargs)


def _load_traffic_input(traffic_input: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
	"""Load traffic input from a dictionary or JSON file path."""
	if isinstance(traffic_input, dict):
		return traffic_input

	if isinstance(traffic_input, (str, Path)):
		path = Path(traffic_input)
		if not path.exists() or not path.is_file():
			raise FileNotFoundError(f"Traffic JSON file not found: {path}")
		with path.open("r", encoding="utf-8") as f:
			loaded = json.load(f)
		if not isinstance(loaded, dict):
			raise ValueError("Traffic JSON root must be an object/dictionary")
		return loaded

	raise ValueError("traffic_input must be a dictionary or path-like JSON file")


def _validate_traffic_sequences(traffic_root: Dict[str, Any]) -> Dict[str, List[List[int]]]:
	"""Validate and normalize traffic_users_per_tti sequences."""
	if "traffic_users_per_tti" not in traffic_root:
		raise ValueError("Missing required key: traffic_users_per_tti")

	per_slice = traffic_root["traffic_users_per_tti"]
	if not isinstance(per_slice, dict):
		raise ValueError("traffic_users_per_tti must be an object/dictionary")

	validated: Dict[str, List[List[int]]] = {}
	for service in SERVICE_ORDER:
		if service not in per_slice:
			raise ValueError(f"Missing service in traffic_users_per_tti: {service}")
		series = per_slice[service]
		if not isinstance(series, list):
			raise ValueError(f"traffic_users_per_tti.{service} must be a list of DTIs")

		dti_rows: List[List[int]] = []
		for dti_idx, dti_traffic in enumerate(series):
			if not isinstance(dti_traffic, list):
				raise ValueError(
					f"traffic_users_per_tti.{service}[{dti_idx}] must be a list of TTI traffic values"
				)
			if len(dti_traffic) == 0:
				raise ValueError(f"traffic_users_per_tti.{service}[{dti_idx}] cannot be empty")

			norm_row: List[int] = []
			for tti_idx, value in enumerate(dti_traffic):
				numeric = _validate_number(
					f"traffic_users_per_tti.{service}[{dti_idx}][{tti_idx}]",
					value,
				)
				norm_row.append(int(round(numeric)))
			dti_rows.append(norm_row)

		validated[service] = dti_rows

	count_set = {len(validated[s]) for s in SERVICE_ORDER}
	if len(count_set) != 1:
		raise ValueError("All slices must have the same number of DTIs in traffic_users_per_tti")

	return validated


def _prepare_service_models(
	checkpoint_dir: Path,
	seed: Optional[int],
) -> Tuple[Dict[str, NetworkSACEnv], Dict[str, SACAgent], int]:
	"""Load all per-slice environments/agents and return total RB capacity C."""
	required = {
		"voip": checkpoint_dir / "sac_voip_final.pt",
		"cbr": checkpoint_dir / "sac_cbr_final.pt",
		"streaming": checkpoint_dir / "sac_streaming_final.pt",
	}
	missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
	if missing:
		raise FileNotFoundError("Missing checkpoint file(s):\n- " + "\n- ".join(missing))

	envs: Dict[str, NetworkSACEnv] = {}
	agents: Dict[str, SACAgent] = {}
	total_capacity: Optional[int] = None

	for service in SERVICE_ORDER:
		ckpt_path = required[service]
		run_cfg = _load_json_if_exists(ckpt_path.with_name(f"{ckpt_path.stem}_config.json"))
		env_cfg = run_cfg.get("environment", {}) if isinstance(run_cfg.get("environment"), dict) else {}
		agent_cfg = run_cfg.get("agent", {}) if isinstance(run_cfg.get("agent"), dict) else {}

		env_kwargs: Dict[str, Any] = {
			"service": service,
			"seed": seed,
		}
		for key in ("config_path", "input_path", "metric_file", "action_low", "action_high", "rb_min", "rb_max"):
			if env_cfg.get(key) is not None:
				env_kwargs[key] = env_cfg[key]

		env = NetworkSACEnv(**env_kwargs)
		state_dim = int(np.prod(env.observation_shape))
		action_dim = int(np.prod(env.action_shape))

		agent = _build_agent_from_config(state_dim, action_dim, agent_cfg)
		agent.load(ckpt_path)

		envs[service] = env
		agents[service] = agent

		if total_capacity is None:
			total_capacity = int(env.c)

	if total_capacity is None:
		raise RuntimeError("Failed to resolve total capacity from environments")

	return envs, agents, total_capacity


def _propose_service_rb(
	service: str,
	traffic_tti: List[int],
	env: NetworkSACEnv,
	agent: SACAgent,
	dti_index: int,
) -> int:
	"""Get one deterministic RB proposal for a single slice in one DTI."""
	normalized_tti = [int(round(_validate_number(f"{service} traffic", x))) for x in traffic_tti]
	if len(normalized_tti) == 0:
		raise ValueError(f"DTI traffic list for {service} cannot be empty")
	return int(
		env.infer_rb_from_traffic(
			traffic_dti=normalized_tti,
			agent=agent,
			dti_index=dti_index,
		)
	)


def _enforce_capacity(
	proposed: List[int],
	capacity: int,
) -> List[int]:
	"""Scale RB proposals so sum <= capacity while preserving proportions."""
	if capacity < 0:
		raise ValueError(f"capacity must be >= 0, got {capacity}")

	nonneg = [max(int(x), 0) for x in proposed]
	total = sum(nonneg)
	if total <= capacity:
		return nonneg
	if capacity == 0:
		return [0] * len(nonneg)

	scale = float(capacity) / float(total)
	scaled = [x * scale for x in nonneg]
	base = [int(np.floor(x)) for x in scaled]

	remaining = capacity - sum(base)
	frac_order = sorted(
		range(len(scaled)),
		key=lambda i: (scaled[i] - base[i]),
		reverse=True,
	)
	for i in frac_order:
		if remaining <= 0:
			break
		base[i] += 1
		remaining -= 1

	return base


def infer_rb_allocation_from_json(
	traffic_input: Union[Dict[str, Any], str, Path],
	checkpoint_dir: Union[str, Path] = "RL_Model/checkpoints",
	seed: Optional[int] = 42,
	) -> Dict[str, List[List[int]]]:
	"""Infer per-DTI RB allocation from structured traffic JSON input.

	Returns:
		{
		  "proposed_rb_per_dti": [
		    [rb_voip, rb_cbr, rb_streaming],
		    ...
		  ],
		  "constrained_rb_per_dti": [
		    [rb_voip, rb_cbr, rb_streaming],
		    ...
		  ]
		}
	"""
	ckpt_dir = Path(checkpoint_dir)
	if not ckpt_dir.exists() or not ckpt_dir.is_dir():
		raise FileNotFoundError(f"checkpoint_dir does not exist or is not a directory: {ckpt_dir}")

	raw_input = _load_traffic_input(traffic_input)
	traffic_by_slice = _validate_traffic_sequences(raw_input)

	envs, agents, capacity = _prepare_service_models(ckpt_dir, seed)
	num_dtis = len(traffic_by_slice["voip"])

	# "proposed" is the raw model output before system-level capacity correction.
	proposed_rb_per_dti: List[List[int]] = []
	# "constrained" is the final allocation after enforcing sum(RBs) <= capacity.
	constrained_rb_per_dti: List[List[int]] = []
	for dti_idx in range(num_dtis):
		proposed: List[int] = []
		for service in SERVICE_ORDER:
			traffic_dti = traffic_by_slice[service][dti_idx]
			rb = _propose_service_rb(service, traffic_dti, envs[service], agents[service], dti_idx)
			proposed.append(rb)
		proposed_rb_per_dti.append(proposed)

		# Enforce rb_voip + rb_cbr + rb_streaming <= C for each DTI.
		constrained = _enforce_capacity(proposed, capacity)
		if sum(constrained) > capacity:
			raise RuntimeError("Capacity enforcement failed: constrained RB sum exceeds capacity")
		constrained_rb_per_dti.append(constrained)

	return {
		"proposed_rb_per_dti": proposed_rb_per_dti,
		"constrained_rb_per_dti": constrained_rb_per_dti,
	}


def infer_rb_allocation(
	voip_traffic: Number,
	cbr_traffic: Number,
	streaming_traffic: Number,
	checkpoint_dir: Union[str, Path] = "RL_Model/checkpoints",
	seed: Optional[int] = 42,
) -> Dict[str, int]:
	"""Backward-compatible scalar helper using one synthetic DTI input."""
	sample_input = {
		"traffic_users_per_tti": {
			"voip": [[int(round(_validate_number("voip_traffic", voip_traffic)))] * 8],
			"cbr": [[int(round(_validate_number("cbr_traffic", cbr_traffic)))] * 8],
			"streaming": [[int(round(_validate_number("streaming_traffic", streaming_traffic)))] * 8],
		}
	}
	result = infer_rb_allocation_from_json(
		traffic_input=sample_input,
		checkpoint_dir=checkpoint_dir,
		seed=seed,
	)
	rb_row = result["constrained_rb_per_dti"][0]
	return {
		"voip": int(rb_row[0]),
		"cbr": int(rb_row[1]),
		"streaming": int(rb_row[2]),
	}


if __name__ == "__main__":
	sample_input = {
		"traffic_users_per_tti": {
			"voip": [
				[5, 5, 5, 5, 5, 5, 5, 5],
				[40, 40, 40, 40, 45, 45, 45, 45],
			],
			"cbr": [
				[5, 5, 5, 5, 5, 5, 5, 5],
				[5, 5, 5, 5, 5, 5, 5, 5],
			],
			"streaming": [
				[40, 40, 40, 40, 45, 45, 45, 45],
				[5, 5, 5, 5, 5, 5, 5, 5],
			],
		},
	}

	result = infer_rb_allocation_from_json(
		traffic_input=sample_input,
		checkpoint_dir="RL_Model/checkpoints",
		seed=42,
	)
	print(result)
