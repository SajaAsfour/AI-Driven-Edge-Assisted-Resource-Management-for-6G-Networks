from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
	from .agent import SACAgent
	from .config import SACConfig
	from .env_wrapper import NetworkSACEnv
	from .replay_buffer import ReplayBuffer


def _to_json_safe(value: Any) -> Any:
	"""Convert values to JSON-safe structures recursively."""
	if isinstance(value, np.generic):
		value = value.item()
	if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
		return None
	if is_dataclass(value):
		return _to_json_safe(asdict(value))
	if isinstance(value, dict):
		return {str(k): _to_json_safe(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_json_safe(v) for v in value]
	if isinstance(value, Path):
		return str(value)
	return value


def save_config(config: Any, path: Union[str, Path]) -> Path:
	"""Save training configuration as JSON next to checkpoints."""
	config_path = Path(path)
	config_path.parent.mkdir(parents=True, exist_ok=True)

	config_dict = _to_json_safe(config)
	with config_path.open("w", encoding="utf-8") as f:
		json.dump(config_dict, f, indent=2, ensure_ascii=False)

	return config_path


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
	"""Load a JSON config file and return it as a dictionary."""
	config_path = Path(path)
	with config_path.open("r", encoding="utf-8") as f:
		loaded = json.load(f)

	if not isinstance(loaded, dict):
		raise ValueError(f"Config JSON must be an object at top-level: {config_path}")
	return loaded


def save_training_metrics(history: Dict[str, Any], path: Union[str, Path]) -> Path:
	"""Save training history metrics as JSON."""
	metrics_path = Path(path)
	metrics_path.parent.mkdir(parents=True, exist_ok=True)

	history_dict = _to_json_safe(history)
	with metrics_path.open("w", encoding="utf-8") as f:
		json.dump(history_dict, f, indent=2, ensure_ascii=False)

	return metrics_path


def _import_rl_components() -> tuple[Any, Any, Any]:
	"""Import RL components lazily to avoid heavy imports at module load."""
	try:
		from .agent import SACAgent
		from .env_wrapper import NetworkSACEnv
		from .replay_buffer import ReplayBuffer
	except ImportError:
		from agent import SACAgent
		from env_wrapper import NetworkSACEnv
		from replay_buffer import ReplayBuffer
	return NetworkSACEnv, SACAgent, ReplayBuffer


def _run_evaluation_episode(env: Any, agent: Any, max_steps: int) -> float:
	"""Run one deterministic evaluation episode and return total reward."""
	state = env.reset()
	episode_reward = 0.0

	for _ in range(max_steps):
		action = agent.select_action(state, evaluate=True)
		next_state, reward, done, _ = env.step(action)
		episode_reward += float(reward)
		state = next_state
		if done:
			break

	return float(episode_reward)


def train_sac(
	config: Optional["SACConfig"] = None,
	) -> Dict[str, Any]:
	"""Train a Soft Actor-Critic agent on the network environment.

	Args:
		config: Optional grouped SAC config object from RL_Model/config.py.
			- If None: create defaults with get_default_config().
			- If provided: use its section values.
			- If a config field is None: fall back to the legacy default value.

	Returns:
		Dictionary with training history and paths to saved checkpoints.
	"""
	# Legacy defaults preserved from original train_sac behavior.
	default_service = "voip"
	default_max_episodes = 200
	default_max_steps_per_episode = 128
	default_batch_size = 64
	default_warmup_steps = 1_000
	default_evaluation_interval = 20
	default_save_interval = 25
	default_replay_capacity = 100_000
	default_rb_min = 1
	default_checkpoint_dir: Union[str, Path] = "RL_Model/checkpoints"
	default_seed: Optional[int] = 42
	default_verbose = True

	try:
		from .config import SACConfig, get_default_config
	except ImportError:
		from config import SACConfig, get_default_config

	if config is None:
		config = get_default_config()
	elif not isinstance(config, SACConfig):
		raise TypeError("config must be an instance of SACConfig")

	# Two-level resolution only:
	# 1) config value
	# 2) fallback default when config value is None
	def _from_config(cfg_value: Any, fallback: Any) -> Any:
		return fallback if cfg_value is None else cfg_value

	service = _from_config(config.environment.service, default_service)
	max_episodes = _from_config(config.training.max_episodes, default_max_episodes)
	max_steps_per_episode = _from_config(
		config.training.max_steps_per_episode,
		default_max_steps_per_episode,
	)
	batch_size = _from_config(config.training.batch_size, default_batch_size)
	warmup_steps = _from_config(config.training.warmup_steps, default_warmup_steps)
	evaluation_interval = _from_config(
		config.evaluation.evaluation_interval,
		default_evaluation_interval,
	)
	save_interval = _from_config(config.checkpoint.save_interval, default_save_interval)
	replay_capacity = _from_config(config.replay_buffer.capacity, default_replay_capacity)
	rb_min = _from_config(config.environment.rb_min, default_rb_min)
	checkpoint_dir = _from_config(config.checkpoint.checkpoint_dir, default_checkpoint_dir)
	seed = _from_config(config.environment.seed, default_seed)
	verbose = _from_config(config.training.verbose, default_verbose)

	if max_episodes <= 0:
		raise ValueError("max_episodes must be > 0")
	if max_steps_per_episode <= 0:
		raise ValueError("max_steps_per_episode must be > 0")
	if batch_size <= 0:
		raise ValueError("batch_size must be > 0")
	if warmup_steps < 0:
		raise ValueError("warmup_steps must be >= 0")
	if evaluation_interval <= 0:
		raise ValueError("evaluation_interval must be > 0")
	if save_interval <= 0:
		raise ValueError("save_interval must be > 0")
	if replay_capacity <= 0:
		raise ValueError("replay_capacity must be > 0")
	if rb_min < 0:
		raise ValueError("rb_min must be >= 0")

	if seed is not None:
		np.random.seed(seed)

	NetworkSACEnv, SACAgent, ReplayBuffer = _import_rl_components()

	env = NetworkSACEnv(service=service, seed=seed, rb_min=rb_min)
	state_dim = int(np.prod(env.observation_shape))
	action_dim = int(np.prod(env.action_shape))

	# Pass agent hyperparameters from config.agent into SACAgent.
	# Any field that is None is intentionally omitted so SACAgent falls back
	# to its own constructor defaults for that parameter.
	agent_kwargs: Dict[str, Any] = {
		"state_dim": state_dim,
		"action_dim": action_dim,
	}
	if config.agent.hidden_dims is not None:
		agent_kwargs["hidden_dims"] = config.agent.hidden_dims
	if config.agent.gamma is not None:
		agent_kwargs["gamma"] = config.agent.gamma
	if config.agent.tau is not None:
		agent_kwargs["tau"] = config.agent.tau
	if config.agent.actor_lr is not None:
		agent_kwargs["actor_lr"] = config.agent.actor_lr
	if config.agent.critic_lr is not None:
		agent_kwargs["critic_lr"] = config.agent.critic_lr
	if config.agent.alpha_lr is not None:
		agent_kwargs["alpha_lr"] = config.agent.alpha_lr
	if config.agent.target_entropy is not None:
		agent_kwargs["target_entropy"] = config.agent.target_entropy
	if config.agent.device is not None:
		agent_kwargs["device"] = config.agent.device

	agent = SACAgent(**agent_kwargs)
	replay_buffer = ReplayBuffer(
		capacity=replay_capacity,
		state_dim=state_dim,
		action_dim=action_dim,
	)

	checkpoint_path = Path(checkpoint_dir)
	checkpoint_path.mkdir(parents=True, exist_ok=True)

	history: Dict[str, list] = {
		"episode_rewards": [],
		"actor_losses": [],
		"critic_losses": [],
		"alpha_values": [],
		"evaluation_rewards": [],
	}
	saved_checkpoints: list[str] = []

	total_steps = 0

	for episode in range(1, max_episodes + 1):
		state = env.reset()
		episode_reward = 0.0

		episode_actor_losses: list[float] = []
		episode_critic_losses: list[float] = []
		episode_alpha_values: list[float] = []

		for _ in range(max_steps_per_episode):
			if total_steps < warmup_steps:
				action = np.random.uniform(
					low=env.action_bounds[0],
					high=env.action_bounds[1],
					size=env.action_shape,
				).astype(np.float32)
			else:
				action = agent.select_action(state, evaluate=False)

			next_state, reward, done, _ = env.step(action)
			replay_buffer.add(state, action, reward, next_state, done)
			episode_reward += float(reward)
			total_steps += 1

			if len(replay_buffer) >= batch_size and total_steps >= warmup_steps:
				update_info = agent.update(replay_buffer, batch_size)
				episode_actor_losses.append(float(update_info["actor_loss"]))
				critic_loss = 0.5 * (
					float(update_info["critic1_loss"]) + float(update_info["critic2_loss"])
				)
				episode_critic_losses.append(critic_loss)
				episode_alpha_values.append(float(update_info["alpha"]))

			state = next_state
			if done:
				break

		mean_actor_loss = (
			float(np.mean(episode_actor_losses))
			if episode_actor_losses
			else float("nan")
		)
		mean_critic_loss = (
			float(np.mean(episode_critic_losses))
			if episode_critic_losses
			else float("nan")
		)
		mean_alpha = (
			float(np.mean(episode_alpha_values))
			if episode_alpha_values
			else float(agent.alpha.item())
		)

		history["episode_rewards"].append(float(episode_reward))
		history["actor_losses"].append(mean_actor_loss)
		history["critic_losses"].append(mean_critic_loss)
		history["alpha_values"].append(mean_alpha)

		if verbose:
			print(
				f"Episode {episode:04d} | "
				f"Reward: {episode_reward:10.4f} | "
				f"Actor Loss: {mean_actor_loss:10.6f} | "
				f"Critic Loss: {mean_critic_loss:10.6f} | "
				f"Alpha: {mean_alpha:8.5f}"
			)

		if episode % evaluation_interval == 0:
			eval_reward = _run_evaluation_episode(
				env=env,
				agent=agent,
				max_steps=max_steps_per_episode,
			)
			history["evaluation_rewards"].append((episode, eval_reward))
			if verbose:
				print(f"  Evaluation @ episode {episode}: reward={eval_reward:.4f}")

		if episode % save_interval == 0:
			episode_ckpt = checkpoint_path / f"sac_{service}_episode_{episode:04d}.pt"
			agent.save(episode_ckpt)
			episode_cfg = episode_ckpt.with_name(f"{episode_ckpt.stem}_config.json")
			save_config(config, episode_cfg)
			saved_checkpoints.append(str(episode_ckpt))
			if verbose:
				print(f"  Saved checkpoint: {episode_ckpt}")

	final_ckpt = checkpoint_path / f"sac_{service}_final.pt"
	agent.save(final_ckpt)
	final_cfg = final_ckpt.with_name(f"{final_ckpt.stem}_config.json")
	save_config(config, final_cfg)
	saved_checkpoints.append(str(final_ckpt))

	if verbose:
		print(f"Training complete. Final checkpoint: {final_ckpt}")

	metrics_path = checkpoint_path / "training_metrics.json"
	save_training_metrics(history, metrics_path)

	return {
		"agent": agent,
		"env": env,
		"replay_buffer": replay_buffer,
		"history": history,
		"saved_checkpoints": saved_checkpoints,
		"total_steps": total_steps,
	}

