from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]


def get_default_sample_input() -> Dict[str, Any]:
	return {
		"traffic_users_per_tti": {
			"voip": [
				[5, 5, 5, 5, 5, 5, 5, 5],
				[40, 40, 40, 40, 45, 45, 45, 45],
				[75,80,80,75,80,80,75,75],
			],
			"cbr": [
				[50, 50, 55, 50, 50, 55, 55, 55],
				[5, 5, 5, 5, 5, 5, 5, 5],
				[75,80,80,75,80,80,75,75],

			],
			"streaming": [
				[5, 5, 5, 5, 5, 5, 5, 5],
				[5, 5, 5, 5, 5, 5, 5, 5],
				[5, 5, 5, 5, 5, 5, 5, 5],
			],
		}
	}


@dataclass(slots=True)
class EnvironmentConfig:
	"""Environment wrapper settings for `NetworkWCSACEnv`."""

	service: str = "voip"
	traffic_profile_mode: str = "fixed"
	fixed_profile_name: str = "profile_1"
	log_random_profile_each_step: bool = False
	seed: Optional[int] = 42
	rb_min: int = 1
	rb_max: Optional[int] = None
	action_low: float = -1.0
	action_high: float = 1.0
	config_path: Optional[PathLike] = None
	input_path: Optional[PathLike] = None
	metric_file: Optional[PathLike] = None
	sample_input: Dict[str, Any] = field(default_factory=get_default_sample_input)


@dataclass(slots=True)
class WCSACAgentConfig:
	"""WCSAC model and optimizer hyperparameters for `WCSACAgent`."""

	state_dim: int = 28
	action_dim: int = 1
	hidden_dims: tuple[int, ...] = (256, 256)
	gamma: float = 0.99
	tau: float = 0.005
	actor_lr: float = 3e-4
	critic_lr: float = 3e-4
	alpha_lr: float = 3e-4
	risk_alpha: float = 0.1
	max_grad_norm: Optional[float] = 1.0
	target_entropy: Optional[float] = None
	device: Optional[str] = None


@dataclass(slots=True)
class ReplayBufferConfig:
	"""Off-policy replay storage settings for `ReplayBuffer`."""

	capacity: int = 100_000
	state_dim: int = 28
	action_dim: int = 1


@dataclass(slots=True)
class TrainingConfig:
	"""Core training-loop settings currently used by `train_wcsac`."""

	max_episodes: int = 40000
	max_steps_per_episode: int = 2
	batch_size: int = 16
	warmup_steps: int = 0
	verbose: bool = True


@dataclass(slots=True)
class EvaluationConfig:
	"""Evaluation behavior during training."""

	evaluation_interval: int = 20
	deterministic: bool = True
	episodes: int = 5
	max_steps_per_episode: int = 2

@dataclass(slots=True)
class CheckpointConfig:
	"""Checkpointing settings for periodic and final saves."""

	checkpoint_dir: PathLike = "WCSAC_RL_Model/checkpoints"
	save_interval: int = 1000
	file_prefix: str = "wcsac"


@dataclass(slots=True)
class WCSACConfig:

	environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
	agent: WCSACAgentConfig = field(default_factory=WCSACAgentConfig)
	replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
	checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

	def trainer_kwargs(self) -> Dict[str, Any]:
		
		return {
			"service": self.environment.service,
			"traffic_profile_mode": self.environment.traffic_profile_mode,
			"fixed_profile_name": self.environment.fixed_profile_name,
			"log_random_profile_each_step": self.environment.log_random_profile_each_step,
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


def get_default_config() -> WCSACConfig:
	"""Return the default project WCSAC configuration."""
	return WCSACConfig()
