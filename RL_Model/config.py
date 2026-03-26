from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]


@dataclass(slots=True)
class EnvironmentConfig:
	"""Environment wrapper settings for `NetworkSACEnv`."""

	service: str = "voip"
	seed: Optional[int] = 42
	rb_min: int = 1
	rb_max: Optional[int] = None
	action_low: float = -1.0
	action_high: float = 1.0
	config_path: Optional[PathLike] = None
	input_path: Optional[PathLike] = None
	metric_file: Optional[PathLike] = None


@dataclass(slots=True)
class SACAgentConfig:
	"""SAC model and optimizer hyperparameters for `SACAgent`."""

	state_dim: int = 22
	action_dim: int = 1
	hidden_dims: tuple[int, ...] = (256, 256)
	gamma: float = 0.99
	tau: float = 0.005
	actor_lr: float = 3e-4
	critic_lr: float = 3e-4
	alpha_lr: float = 3e-4
	target_entropy: Optional[float] = None
	device: Optional[str] = None


@dataclass(slots=True)
class ReplayBufferConfig:
	"""Off-policy replay storage settings for `ReplayBuffer`."""

	capacity: int = 100_000
	state_dim: int = 22
	action_dim: int = 1


@dataclass(slots=True)
class TrainingConfig:
	"""Core training-loop settings currently used by `train_sac`."""

	max_episodes: int = 200
	max_steps_per_episode: int = 128
	batch_size: int = 16
	warmup_steps: int = 300
	verbose: bool = True


@dataclass(slots=True)
class EvaluationConfig:
	"""Evaluation behavior during training."""

	evaluation_interval: int = 20
	deterministic: bool = True
	episodes: int = 5
	max_steps_per_episode: int = 128


@dataclass(slots=True)
class CheckpointConfig:
	"""Checkpointing settings for periodic and final saves."""

	checkpoint_dir: PathLike = "RL_Model/checkpoints"
	save_interval: int = 25
	file_prefix: str = "sac"


@dataclass(slots=True)
class SACConfig:
	"""Top-level grouped SAC configuration."""

	environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
	agent: SACAgentConfig = field(default_factory=SACAgentConfig)
	replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
	checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

	def trainer_kwargs(self) -> Dict[str, Any]:
		"""Build kwargs compatible with current `train_sac(...)` signature.

		This keeps integration easy while still allowing richer config sections
		for direct use with `NetworkSACEnv`, `SACAgent`, and `ReplayBuffer`.
		"""
		return {
			"service": self.environment.service,
			"max_episodes": self.training.max_episodes,
			"max_steps_per_episode": self.training.max_steps_per_episode,
			"batch_size": self.training.batch_size,
			"warmup_steps": self.training.warmup_steps,
			"evaluation_interval": self.evaluation.evaluation_interval,
			"save_interval": self.checkpoint.save_interval,
			"replay_capacity": self.replay_buffer.capacity,
			"rb_min": self.environment.rb_min,
			"checkpoint_dir": self.checkpoint.checkpoint_dir,
			"seed": self.environment.seed,
			"verbose": self.training.verbose,
		}


def get_default_config() -> SACConfig:
	"""Return the default project SAC configuration."""
	return SACConfig()
