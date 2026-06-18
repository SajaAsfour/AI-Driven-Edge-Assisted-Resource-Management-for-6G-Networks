from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
AVAILABLE_SERVICES = ("voip", "cbr")
AVAILABLE_MODELS = ("wcsac",)


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
		raise ValueError(f"service must be one of: {', '.join(AVAILABLE_SERVICES)}")
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


@dataclass(slots=True)
class MultiTrafficPredictionConfig:
	"""Configuration for a two-input WCSAC multi-traffic prediction run."""

	input_1: TrafficInputSelection
	input_2: TrafficInputSelection
	model_name: Optional[str] = "wcsac"
	capacity: int = 8
	seed: int = 42
	output_dir: Path = field(default_factory=lambda: WORKSPACE_ROOT / "WCSAC_RL_Model" / "checkpoints" / "multi_traffic")
	checkpoint_base_dir: Optional[Path] = None

	def normalized(self) -> "MultiTrafficPredictionConfig":
		model_key = normalize_model_name(self.model_name)
		base_dir = Path(self.checkpoint_base_dir).expanduser() if self.checkpoint_base_dir is not None else resolve_default_checkpoint_base_dir(model_key)
		if not base_dir.is_absolute():
			base_dir = (WORKSPACE_ROOT / base_dir).resolve()
		return MultiTrafficPredictionConfig(
			input_1=self.input_1.normalized(),
			input_2=self.input_2.normalized(),
			model_name=model_key,
			capacity=int(self.capacity),
			seed=int(self.seed),
			output_dir=Path(self.output_dir).expanduser(),
			checkpoint_base_dir=base_dir,
		)
