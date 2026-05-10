from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from SAC_RL_Model.config import get_default_config


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
AVAILABLE_SERVICES = ("voip", "cbr", "streaming")


def resolve_project_root() -> Path:
	return WORKSPACE_ROOT


def resolve_default_checkpoint_base_dir() -> Path:
	cfg = get_default_config()
	checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
	if checkpoint_dir.is_absolute():
		return checkpoint_dir
	return WORKSPACE_ROOT / checkpoint_dir


def normalize_service_name(service: str) -> str:
	if not isinstance(service, str):
		raise ValueError("service must be a string")
	service_key = service.strip().lower()
	if service_key not in AVAILABLE_SERVICES:
		raise ValueError(f"service must be one of: {', '.join(AVAILABLE_SERVICES)}")
	return service_key


def resolve_final_checkpoint_path(base_checkpoint_dir: Path, service: str) -> Path:
	service_key = normalize_service_name(service)
	checkpoint_path = base_checkpoint_dir / service_key / "random" / f"sac_{service_key}_final.pt"
	if not checkpoint_path.exists():
		raise FileNotFoundError(
			f"Final checkpoint not found for service '{service_key}'. Expected path: {checkpoint_path}"
		)
	return checkpoint_path


@dataclass(slots=True)
class TrafficInputSelection:
	"""User-selected traffic input for one SAC agent."""

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
	"""Configuration for a two-input multi-traffic prediction run."""

	input_1: TrafficInputSelection
	input_2: TrafficInputSelection
	capacity: int = 8
	seed: int = 42
	output_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "SAC_RL_Model" / "checkpoints" / "multi_traffic")
	checkpoint_base_dir: Optional[Path] = None

	def normalized(self) -> "MultiTrafficPredictionConfig":
		base_dir = Path(self.checkpoint_base_dir).expanduser() if self.checkpoint_base_dir is not None else resolve_default_checkpoint_base_dir()
		if not base_dir.is_absolute():
			base_dir = (WORKSPACE_ROOT / base_dir).resolve()
		return MultiTrafficPredictionConfig(
			input_1=self.input_1.normalized(),
			input_2=self.input_2.normalized(),
			capacity=int(self.capacity),
			seed=int(self.seed),
			output_dir=Path(self.output_dir).expanduser(),
			checkpoint_base_dir=base_dir,
		)
