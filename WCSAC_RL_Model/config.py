from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]


def get_default_sample_input() -> Dict[str, Any]:
	# Each row corresponds to one DTI (m=8 DTIs total, n=8 TTIs per DTI).
	# Each DTI row samples from one distinct UE profile so that all 8 profiles
	# (profile_1=[5,10] through profile_8=[75,80]) are exercised across the
	# episode. Previously all rows used profile_1 values [5,10,...], making
	# this a degenerate single-profile test case.
	_matrix = [
		[ 5, 10,  5, 10,  5, 10,  5, 10],  # DTI 1 — profile_1  [5,  10]
		[15, 20, 15, 20, 15, 20, 15, 20],  # DTI 2 — profile_2  [15, 20]
		[25, 30, 25, 30, 25, 30, 25, 30],  # DTI 3 — profile_3  [25, 30]
		[35, 40, 35, 40, 35, 40, 35, 40],  # DTI 4 — profile_4  [35, 40]
		[45, 50, 45, 50, 45, 50, 45, 50],  # DTI 5 — profile_5  [45, 50]
		[55, 60, 55, 60, 55, 60, 55, 60],  # DTI 6 — profile_6  [55, 60]
		[65, 70, 65, 70, 65, 70, 65, 70],  # DTI 7 — profile_7  [65, 70]
		[75, 80, 75, 80, 75, 80, 75, 80],  # DTI 8 — profile_8  [75, 80]
	]
	return {
		"traffic_users_per_tti": {
			"voip":      [row[:] for row in _matrix],
			"cbr":       [row[:] for row in _matrix],
		}
	}


@dataclass()
class EnvironmentConfig:
	"""Environment wrapper settings for `NetworkWCSACEnv`."""

	service: str = "voip"
	traffic_profile_mode: str = "fixed"
	# Default profile for fixed mode. profile_1=[5,10] UEs.
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


@dataclass()
class WCSACAgentConfig:
	"""WCSAC model and optimizer hyperparameters for `WCSACAgent`."""

	# state_dim = 4 scalar features + k CDF bins = 4 + 16 = 20.
	# The 4 scalars are: progress, beta_current, beta_cumulative, rb_norm.
	# k=16 comes from network_config.json. Formerly mis-set to 22.
	state_dim: int = 20
	action_dim: int = 1
	hidden_dims: tuple[int, ...] = (256, 256)
	gamma: float = 0.99
	tau: float = 0.005
	actor_lr: float = 3e-4
	critic_lr: float = 3e-4
	alpha_lr: float = 2e-4
	# risk_alpha: CVaR tail probability. 0.1 focuses on worst-case 10% of outcomes.
	risk_alpha: float = 0.1
	max_grad_norm: Optional[float] = 1.0
	target_entropy: Optional[float] = None
	# beta_threshold: QoS degradation limit. Matches WCSACAgent default.
	beta_threshold: float = 0.5
	# lagrange_lr: Lagrange multiplier growth rate when constraint is violated.
	# Keep moderate — too high causes lambda to explode and crush the reward signal,
	# making the agent allocate maximum RBs on every step (risk-averse collapse).
	lagrange_lr: float = 1e-4
	# lambda_init: warm-start the Lagrange multiplier so the constraint is active
	# from episode 1, but keep it small so the reward signal still dominates early.
	lambda_init: float = 0.01
	device: Optional[str] = None


@dataclass()
class ReplayBufferConfig:
	"""Off-policy replay storage settings for `ReplayBuffer`."""

	capacity: int = 100_000
	# Must match WCSACAgentConfig.state_dim = 4 + k = 20.
	state_dim: int = 20
	action_dim: int = 1


@dataclass()
class TrainingConfig:
	"""Core training-loop settings currently used by `train_wcsac`."""

	max_episodes: int = 3000
	max_steps_per_episode: int = 200
	# batch_size: was 16, corrected to match trainer.py default of 64.
	batch_size: int = 64
	# warmup_steps: was 0, corrected to match trainer.py default of 1000.
	# Without warmup the agent updates on a tiny unrepresentative buffer
	# from step 1, which harms early learning.
	warmup_steps: int = 1000
	verbose: bool = True


@dataclass()
class EvaluationConfig:
	"""Evaluation behavior during training."""

	evaluation_interval: int = 20
	deterministic: bool = True
	episodes: int = 5
	max_steps_per_episode: int = 200

@dataclass()
class CheckpointConfig:
	"""Checkpointing settings for periodic and final saves."""

	checkpoint_dir: PathLike = "WCSAC_RL_Model/checkpoints"
	save_interval: int = 1000
	file_prefix: str = "wcsac"


@dataclass()
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
