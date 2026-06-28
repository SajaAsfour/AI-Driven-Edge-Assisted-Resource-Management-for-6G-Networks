from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
AVAILABLE_SERVICES = ("voip", "cbr")
AVAILABLE_MODELS = ("wcsac",)

if str(WORKSPACE_ROOT) not in sys.path:
	sys.path.insert(0, str(WORKSPACE_ROOT))


def resolve_default_beta_threshold() -> float:
	"""Single source of truth for beta_threshold: WCSAC_RL_Model/config.py."""
	from WCSAC_RL_Model.config import get_default_config

	return float(get_default_config().agent.beta_threshold)


def resolve_project_root() -> Path:
	return WORKSPACE_ROOT


def normalize_model_name(model_name: str | None) -> str:
	if model_name is None:
		return "wcsac"
	if not isinstance(model_name, str):
		raise ValueError("model_name must be a string or None")
	model_key = model_name.strip().lower()
	if model_key not in AVAILABLE_MODELS:
		raise ValueError(f"model_name must be one of: {', '.join(AVAILABLE_MODELS)}")
	return model_key


def resolve_default_checkpoint_base_dir(model_name: str | None = None) -> Path:
	normalize_model_name(model_name)
	return WORKSPACE_ROOT / "WCSAC_RL_Model" / "checkpoints"


def normalize_service_name(service: str) -> str:
	if not isinstance(service, str):
		raise ValueError("service must be a string")
	service_key = service.strip().lower()
	if service_key not in AVAILABLE_SERVICES:
		raise ValueError(
			f"Unsupported service '{service_key}'. service must be one of: {', '.join(AVAILABLE_SERVICES)}"
		)
	return service_key


def resolve_final_checkpoint_path(base_checkpoint_dir: Path, service: str, model_name: str | None = None) -> Path:
	service_key = normalize_service_name(service)
	checkpoint_prefix = normalize_model_name(model_name)
	checkpoint_path = base_checkpoint_dir / service_key / "random" / f"{checkpoint_prefix}_{service_key}_final.pt"
	if not checkpoint_path.exists():
		raise FileNotFoundError(
			f"Final checkpoint not found for service '{service_key}'. Expected path: {checkpoint_path}"
		)
	return checkpoint_path


@dataclass(slots=True)
class TrafficInputSelection:
	"""User-selected traffic input for one WCSAC agent."""

	service: str
	profile_name: str
	label: str = ""

	def normalized(self) -> "TrafficInputSelection":
		return TrafficInputSelection(
			service=normalize_service_name(self.service),
			profile_name=str(self.profile_name).strip().lower(),
			label=self.label.strip(),
		)


MIN_TRAFFIC_INPUTS = 2


def get_default_multi_traffic_sample_input() -> dict:
	"""Built-in traffic source for multi-traffic prediction mode.

	`traffic_users_per_tti` maps each service name to its traffic matrix
	(rows = DTIs, columns = TTIs per DTI). Number of inputs, service names,
	number of DTIs, and number of TTIs per DTI are all derived from this matrix.
	"""
	return {
		"traffic_users_per_tti": {
			"voip": [
				[5, 10, 5, 10, 5, 10, 5, 10],
				[15, 20, 15, 20, 15, 20, 15, 20],
				[25, 30, 25, 30, 25, 30, 25, 30],
				[35, 40, 35, 40, 35, 40, 35, 40],
				[45, 50, 45, 50, 45, 50, 45, 50],
				[55, 60, 55, 60, 55, 60, 55, 60],
				[65, 70, 65, 70, 65, 70, 65, 70],
				[75, 80, 75, 80, 75, 80, 75, 80],
			],
			"cbr": [
				[5, 10, 5, 10, 5, 10, 5, 10],
				[15, 20, 15, 20, 15, 20, 15, 20],
				[25, 30, 25, 30, 25, 30, 25, 30],
				[35, 40, 35, 40, 35, 40, 35, 40],
				[45, 50, 45, 50, 45, 50, 45, 50],
				[55, 60, 55, 60, 55, 60, 55, 60],
				[65, 70, 65, 70, 65, 70, 65, 70],
				[75, 80, 75, 80, 75, 80, 75, 80],
			],
		}
	}


@dataclass(slots=True)
class MultiTrafficPredictionConfig:
	"""Configuration for an N-input WCSAC multi-traffic prediction run.

	Everything the run needs (number of inputs, service names, traffic matrices,
	number of DTIs, number of TTIs per DTI) is derived from `sample_input`. The
	`inputs` list may be left empty; it is then auto-built from `sample_input`'s
	service keys.
	"""

	inputs: List[TrafficInputSelection] = field(default_factory=list)
	model_name: Optional[str] = "wcsac"
	capacity: int = 8
	beta_threshold: Optional[float] = None
	seed: int = 42
	output_dir: Path = field(default_factory=lambda: WORKSPACE_ROOT / "WCSAC_RL_Model" / "checkpoints" / "multi_traffic")
	checkpoint_base_dir: Optional[Path] = None
	sample_input: dict = field(default_factory=get_default_multi_traffic_sample_input)

	def normalized(self) -> "MultiTrafficPredictionConfig":
		traffic_by_service = self.sample_input.get("traffic_users_per_tti") if self.sample_input else None
		if not traffic_by_service:
			raise ValueError("sample_input must contain a non-empty 'traffic_users_per_tti' mapping")

		inputs = list(self.inputs) if self.inputs else [
			TrafficInputSelection(service=service, profile_name="sample_input", label=f"input_{idx}")
			for idx, service in enumerate(traffic_by_service.keys(), start=1)
		]
		if len(inputs) < MIN_TRAFFIC_INPUTS:
			raise ValueError(
				f"At least {MIN_TRAFFIC_INPUTS} traffic inputs are required, got {len(inputs)}"
			)
		model_key = normalize_model_name(self.model_name)
		base_dir = Path(self.checkpoint_base_dir).expanduser() if self.checkpoint_base_dir is not None else resolve_default_checkpoint_base_dir(model_key)
		if not base_dir.is_absolute():
			base_dir = (WORKSPACE_ROOT / base_dir).resolve()
		beta_threshold = (
			resolve_default_beta_threshold() if self.beta_threshold is None else float(self.beta_threshold)
		)
		return MultiTrafficPredictionConfig(
			inputs=[selection.normalized() for selection in inputs],
			model_name=model_key,
			capacity=int(self.capacity),
			beta_threshold=beta_threshold,
			seed=int(self.seed),
			output_dir=Path(self.output_dir).expanduser(),
			checkpoint_base_dir=base_dir,
			sample_input=self.sample_input,
		)
