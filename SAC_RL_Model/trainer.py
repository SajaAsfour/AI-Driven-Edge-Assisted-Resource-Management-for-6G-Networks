from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

import matplotlib.pyplot as plt
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


def _build_file_logger(name: str, file_path: Path) -> logging.Logger:
	"""Create an isolated file logger with a fresh handler."""
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	logger.propagate = False
	for handler in list(logger.handlers):
		handler.flush()
		handler.close()
		logger.removeHandler(handler)
	file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
	file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
	logger.addHandler(file_handler)
	return logger


def _safe_float_for_log(value: Any) -> Optional[float]:
	"""Return finite float value for logging, otherwise None."""
	try:
		value_float = float(value)
	except (TypeError, ValueError):
		return None
	if not np.isfinite(value_float):
		return None
	return value_float


def _format_optional_float(value: Optional[float], decimals: int = 4) -> str:
	if value is None:
		return "None"
	return f"{float(value):.{decimals}f}"


def _safe_plot_float(value: Any) -> float:
	"""Return finite float value for plots, else NaN."""
	try:
		value_float = float(value)
	except (TypeError, ValueError):
		return float("nan")
	if not np.isfinite(value_float):
		return float("nan")
	return value_float


def _safe_utilization(value: Any) -> float:
	"""Return utilization in [0, 1], else NaN."""
	value_float = _safe_plot_float(value)
	if not np.isfinite(value_float):
		return float("nan")
	return float(np.clip(value_float, 0.0, 1.0))


def _safe_dti_index_for_plot(value: Any, fallback: int) -> int:
	"""Return positive integer DTI index for plotting."""
	try:
		idx = int(value)
	except (TypeError, ValueError):
		idx = int(fallback)
	if idx <= 0:
		idx = int(fallback)
	return idx


def _save_episode_dti_plots(
	episode_data: list[Dict[str, Any]],
	output_dir: Path,
) -> int:
	"""Save per-episode DTI plots for beta, reward, and utilization diagnostics."""
	output_dir.mkdir(parents=True, exist_ok=True)
	generated = 0

	for episode_entry in episode_data:
		episode_idx = int(episode_entry.get("episode", 0))
		raw_dti_indices = list(episode_entry.get("dti_indices", []))
		dti_indices = [
			_safe_dti_index_for_plot(raw_val, fallback=i + 1)
			for i, raw_val in enumerate(raw_dti_indices)
		]
		beta_values = [_safe_plot_float(x) for x in episode_entry.get("beta_values", [])]
		reward_values = [_safe_plot_float(x) for x in episode_entry.get("reward_values", [])]
		utilization_values = [_safe_utilization(x) for x in episode_entry.get("utilization_values", [])]

		if not dti_indices:
			continue

		min_len_beta = min(len(dti_indices), len(beta_values))
		min_len_reward = min(len(dti_indices), len(reward_values))
		min_len_utilization = min(len(dti_indices), len(utilization_values))
		min_len_beta_utilization = min(len(beta_values), len(utilization_values))

		x_vals_beta = dti_indices[:min_len_beta]
		beta_vals = beta_values[:min_len_beta]
		x_vals_reward = dti_indices[:min_len_reward]
		reward_vals = reward_values[:min_len_reward]
		x_vals_utilization = dti_indices[:min_len_utilization]
		util_vals = utilization_values[:min_len_utilization]
		beta_vals_for_scatter = beta_values[:min_len_beta_utilization]
		util_vals_for_scatter = utilization_values[:min_len_beta_utilization]

		beta_path = output_dir / f"beta_vs_dti_episode_{episode_idx:03d}.png"
		reward_path = output_dir / f"reward_vs_dti_episode_{episode_idx:03d}.png"
		utilization_path = output_dir / f"utilization_vs_dti_episode_{episode_idx:03d}.png"
		beta_vs_utilization_path = output_dir / f"beta_vs_utilization_episode_{episode_idx:03d}.png"

		if min_len_beta > 0:
			plt.figure(figsize=(8, 4.5))
			plt.plot(x_vals_beta, beta_vals, marker="o", linewidth=1.5)
			plt.title(f"Beta vs DTI - Episode {episode_idx:03d}")
			plt.xlabel("DTI Index")
			plt.ylabel("beta_current")
			plt.grid(True)
			plt.tight_layout()
			plt.savefig(beta_path, dpi=150)
			plt.close()
			generated += 1

		if min_len_reward > 0:
			plt.figure(figsize=(8, 4.5))
			plt.plot(x_vals_reward, reward_vals, marker="o", linewidth=1.5)
			plt.title(f"Reward vs DTI - Episode {episode_idx:03d}")
			plt.xlabel("DTI Index")
			plt.ylabel("reward_current")
			plt.grid(True)
			plt.tight_layout()
			plt.savefig(reward_path, dpi=150)
			plt.close()
			generated += 1

		if min_len_utilization > 0:
			plt.figure(figsize=(8, 4.5))
			plt.plot(x_vals_utilization, util_vals, marker="o", linewidth=1.5)
			plt.title(f"Utilization vs DTI - Episode {episode_idx:03d}")
			plt.xlabel("DTI Index")
			plt.ylabel("utilization (rb_used / C)")
			plt.grid(True)
			plt.tight_layout()
			plt.savefig(utilization_path, dpi=150)
			plt.close()
			generated += 1

		if min_len_beta_utilization > 0:
			plt.figure(figsize=(8, 4.5))
			plt.scatter(beta_vals_for_scatter, util_vals_for_scatter, s=20, alpha=0.9)
			if min_len_beta_utilization > 1:
				plt.plot(beta_vals_for_scatter, util_vals_for_scatter, linewidth=1.0, alpha=0.35)
			plt.title(f"Beta vs Utilization - Episode {episode_idx:03d}")
			plt.xlabel("beta_current")
			plt.ylabel("utilization (rb_used / C)")
			plt.grid(True)
			plt.tight_layout()
			plt.savefig(beta_vs_utilization_path, dpi=150)
			plt.close()
			generated += 1

	return generated


def _run_evaluation(
	env: Any,
	agent: Any,
	max_steps: int,
	episodes: int,
	evaluation_logger: logging.Logger,
) -> tuple[float, list[Dict[str, Any]]]:
	"""Run deterministic evaluation episodes with full per-step diagnostics logging."""
	if hasattr(env, "set_logger"):
		env.set_logger(evaluation_logger)
	if hasattr(env, "set_logging_context"):
		env.set_logging_context("evaluation")

	episode_rewards: list[float] = []
	episode_dti_series: list[Dict[str, Any]] = []
	evaluation_logger.info("Running deterministic SAC evaluation...")

	for eval_episode_idx in range(1, episodes + 1):
		evaluation_logger.info("=" * 50)
		evaluation_logger.info(f"START EVALUATION EPISODE {eval_episode_idx}")
		evaluation_logger.info("=" * 50)
		if hasattr(env, "set_logger"):
			env.set_logger(evaluation_logger)
		if hasattr(env, "set_logging_context"):
			env.set_logging_context("evaluation")

		state = env.reset()
		episode_reward = 0.0
		episode_dti_indices: list[int] = []
		episode_beta_values: list[float] = []
		episode_reward_values: list[float] = []
		episode_utilization_values: list[float] = []
		episode_rb_used_values: list[float] = []
		for step_idx in range(max_steps):
			action = agent.select_action(state, evaluate=True)
			next_state, reward, done, info = env.step(action)
			episode_reward += float(reward)
			step_info = info if isinstance(info, dict) else {}
			episode_dti_indices.append(
				_safe_dti_index_for_plot(step_info.get("dti_index"), fallback=step_idx + 1)
			)
			episode_beta_values.append(_safe_plot_float(step_info.get("beta_current")))
			episode_reward_values.append(_safe_plot_float(reward))
			episode_utilization_values.append(_safe_utilization(step_info.get("utilization")))
			episode_rb_used_values.append(_safe_plot_float(step_info.get("rb_alloc")))
			state = next_state
			if done:
				break

		episode_rewards.append(float(episode_reward))
		episode_dti_series.append(
			{
				"episode": int(eval_episode_idx),
				"dti_indices": episode_dti_indices,
				"beta_values": episode_beta_values,
				"reward_values": episode_reward_values,
				"utilization_values": episode_utilization_values,
				"rb_used_values": episode_rb_used_values,
			}
		)
		episode_reward_safe = _safe_float_for_log(episode_reward)
		evaluation_logger.info(
			f"Episode {eval_episode_idx:03d} reward: "
			f"{_format_optional_float(episode_reward_safe, decimals=4)}"
		)

	finite_rewards = [float(v) for v in episode_rewards if np.isfinite(v)]
	mean_reward: Optional[float]
	std_reward: Optional[float]
	if finite_rewards:
		mean_reward = float(np.mean(finite_rewards))
		std_reward = float(np.std(finite_rewards))
	else:
		mean_reward = None
		std_reward = None

	evaluation_logger.info(
		"Evaluation complete | mean reward: "
		f"{_format_optional_float(_safe_float_for_log(mean_reward), decimals=4)} | "
		f"std: {_format_optional_float(_safe_float_for_log(std_reward), decimals=4)}"
	)

	if hasattr(env, "set_logging_context"):
		env.set_logging_context("training")

	mean_reward_out = float(mean_reward) if mean_reward is not None else float("nan")
	return mean_reward_out, episode_dti_series


def train_sac(
	config: Optional["SACConfig"] = None,
	) -> Dict[str, Any]:
	"""Train a Soft Actor-Critic agent on the network environment.

	Args:
		config: Optional grouped SAC config object from SAC_RL_Model/config.py.
			- If None: create defaults with get_default_config().
			- If provided: use its section values.
			- If a config field is None: fall back to the legacy default value.

	Returns:
		Dictionary with training history and paths to saved checkpoints.
	"""
	default_service = "voip"
	default_max_episodes = 200
	default_max_steps_per_episode = 128
	default_batch_size = 64
	default_warmup_steps = 1_000
	default_evaluation_interval = 20
	default_save_interval = 25
	default_replay_capacity = 100_000
	default_rb_min = 1
	default_evaluation_episodes = 5
	default_checkpoint_dir: Union[str, Path] = "SAC_RL_Model/checkpoints"
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
	traffic_profile_mode = _from_config(config.environment.traffic_profile_mode, "fixed")
	fixed_profile_name = _from_config(config.environment.fixed_profile_name, "profile_1")
	log_random_profile_each_step = _from_config(
		config.environment.log_random_profile_each_step,
		False,
	)
	evaluation_episodes = _from_config(config.evaluation.episodes, default_evaluation_episodes)
	evaluation_max_steps = _from_config(config.evaluation.max_steps_per_episode, max_steps_per_episode)
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
	if evaluation_episodes <= 0:
		raise ValueError("evaluation_episodes must be > 0")
	if evaluation_max_steps <= 0:
		raise ValueError("evaluation_max_steps must be > 0")
	if save_interval <= 0:
		raise ValueError("save_interval must be > 0")
	if replay_capacity <= 0:
		raise ValueError("replay_capacity must be > 0")
	if rb_min < 0:
		raise ValueError("rb_min must be >= 0")
	if traffic_profile_mode not in {"fixed", "random"}:
		raise ValueError("traffic_profile_mode must be either 'fixed' or 'random'")

	if seed is not None:
		np.random.seed(seed)

	NetworkSACEnv, SACAgent, ReplayBuffer = _import_rl_components()

	env = NetworkSACEnv(
		service=service,
		traffic_profile_mode=traffic_profile_mode,
		fixed_profile_name=fixed_profile_name,
		seed=seed,
		rb_min=rb_min,
		log_random_profile_each_step=log_random_profile_each_step,
	)
	if hasattr(env, "set_logging_context"):
		env.set_logging_context("training")
	state_dim = int(np.prod(env.observation_shape))
	action_dim = int(np.prod(env.action_shape))

	# Pass agent hyperparameters from config.agent into SACAgent.
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

	training_log_path = checkpoint_path / "training.log"
	evaluation_log_path = checkpoint_path / "evaluation.log"
	training_logger = _build_file_logger(f"sac_training_{service}", training_log_path)
	evaluation_logger = _build_file_logger(f"sac_evaluation_{service}", evaluation_log_path)

	if hasattr(env, "set_logger"):
		env.set_logger(training_logger)
	if hasattr(env, "set_logging_context"):
		env.set_logging_context("training")

	history: Dict[str, list] = {
		"episode_rewards": [],
		"actor_loss": [],
		"critic1_loss": [],
		"critic2_loss": [],
		"critic_diff": [],
		"q_value_loss": [],
		"alpha": [],
		"alpha_loss": [],
		"entropy": [],
		"evaluation_rewards": [],
		"training_episode_dti_series": [],
		"latest_evaluation_episode_dti_series": [],
	}
	history["service"] = service
	saved_checkpoints: list[str] = []
	latest_evaluation_episode_dti_series: list[Dict[str, Any]] = []

	total_steps = 0

	for episode in range(1, max_episodes + 1):
		state = env.reset()
		episode_reward = 0.0
		episode_dti_indices: list[int] = []
		episode_beta_values: list[float] = []
		episode_reward_values: list[float] = []
		episode_utilization_values: list[float] = []
		episode_rb_used_values: list[float] = []

		episode_actor_losses: list[float] = []
		episode_critic1_losses: list[float] = []
		episode_critic2_losses: list[float] = []
		episode_critic_diffs: list[float] = []
		episode_q_value_losses: list[float] = []
		episode_alpha_values: list[float] = []
		episode_alpha_losses: list[float] = []
		episode_entropy_values: list[float] = []

		for step_idx in range(max_steps_per_episode):
			if total_steps < warmup_steps:
				action = np.random.uniform(
					low=env.action_bounds[0],
					high=env.action_bounds[1],
					size=env.action_shape,
				).astype(np.float32)
			else:
				action = agent.select_action(state, evaluate=False)

			next_state, reward, done, info = env.step(action)
			if verbose:
				profile_mode = info.get("profile_mode")
				profile_name = info.get("profile_name")
				profile_values = info.get("profile_values")
				dti_index = info.get("dti_index")
				rb_alloc = _safe_float_for_log(info.get("rb_alloc"))
				capacity = _safe_float_for_log(info.get("capacity", info.get("C")))
				utilization = _safe_float_for_log(info.get("utilization"))
				if utilization is None:
					utilization_text = "None"
				elif rb_alloc is not None and capacity is not None:
					utilization_text = f"rb_used / C = {rb_alloc:.0f} / {capacity:.0f} = {utilization:.4f}"
				else:
					utilization_text = f"{utilization:.4f}"
				training_logger.info(
					f"Step {step_idx + 1:03d} | DTI {int(dti_index) + 1:03d} | "
					f"Profile Mode: {profile_mode} | Profile: {profile_name} -> {profile_values} | "
					f"Utilization: {utilization_text}"
				)
			episode_dti_indices.append(
				_safe_dti_index_for_plot(info.get("dti_index"), fallback=step_idx + 1)
			)
			episode_beta_values.append(_safe_plot_float(info.get("beta_current")))
			episode_reward_values.append(_safe_plot_float(reward))
			episode_utilization_values.append(_safe_utilization(info.get("utilization")))
			episode_rb_used_values.append(_safe_plot_float(info.get("rb_alloc")))
			replay_buffer.add(state, action, reward, next_state, done)
			episode_reward += float(reward)
			total_steps += 1

			if len(replay_buffer) >= batch_size and total_steps >= warmup_steps:
				update_info = agent.update(replay_buffer, batch_size)

				actor_loss = float(update_info.get("actor_loss", float("nan")))
				critic1_loss = float(update_info.get("critic1_loss", float("nan")))
				critic2_loss = float(update_info.get("critic2_loss", float("nan")))
				critic_diff = abs(critic1_loss - critic2_loss)
				q_value_loss = 0.5 * (critic1_loss + critic2_loss)
				alpha_value = float(update_info.get("alpha", float("nan")))
				alpha_loss = float(update_info.get("alpha_loss", float("nan")))

				entropy_value: float
				if "entropy" in update_info and update_info.get("entropy") is not None:
					entropy_value = float(update_info["entropy"])
				elif "log_prob_mean" in update_info and update_info.get("log_prob_mean") is not None:
					entropy_value = -float(update_info["log_prob_mean"])
				else:
					entropy_value = float("nan")

				episode_actor_losses.append(actor_loss)
				episode_critic1_losses.append(critic1_loss)
				episode_critic2_losses.append(critic2_loss)
				episode_critic_diffs.append(critic_diff)
				episode_q_value_losses.append(q_value_loss)
				episode_alpha_values.append(alpha_value)
				episode_alpha_losses.append(alpha_loss)
				episode_entropy_values.append(entropy_value)

			state = next_state
			if done:
				break

		def _nanmean_or_nan(values: list[float]) -> float:
			if not values:
				return float("nan")
			arr = np.asarray(values, dtype=np.float64)
			return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

		mean_actor_loss = _nanmean_or_nan(episode_actor_losses)
		mean_critic1_loss = _nanmean_or_nan(episode_critic1_losses)
		mean_critic2_loss = _nanmean_or_nan(episode_critic2_losses)
		mean_critic_diff = _nanmean_or_nan(episode_critic_diffs)
		mean_q_value_loss = _nanmean_or_nan(episode_q_value_losses)
		mean_alpha = _nanmean_or_nan(episode_alpha_values)
		if not np.isfinite(mean_alpha):
			mean_alpha = float(agent.alpha.item())
		mean_alpha_loss = _nanmean_or_nan(episode_alpha_losses)
		mean_entropy = _nanmean_or_nan(episode_entropy_values)

		history["episode_rewards"].append(float(episode_reward))
		history["actor_loss"].append(mean_actor_loss)
		history["critic1_loss"].append(mean_critic1_loss)
		history["critic2_loss"].append(mean_critic2_loss)
		history["critic_diff"].append(mean_critic_diff)
		history["q_value_loss"].append(mean_q_value_loss)
		history["alpha"].append(mean_alpha)
		history["alpha_loss"].append(mean_alpha_loss)
		history["entropy"].append(mean_entropy)
		history["training_episode_dti_series"].append(
			{
				"episode": int(episode),
				"dti_indices": episode_dti_indices,
				"beta_values": episode_beta_values,
				"reward_values": episode_reward_values,
				"utilization_values": episode_utilization_values,
				"rb_used_values": episode_rb_used_values,
			}
		)

		if verbose:
			training_logger.info(
				f"Episode {episode:04d} | "
				f"Reward: {episode_reward:10.4f} | "
				f"Actor Loss: {mean_actor_loss:10.6f} | "
				f"Critic1 Loss: {mean_critic1_loss:10.6f} | "
				f"Critic2 Loss: {mean_critic2_loss:10.6f} | "
				f"Q Loss: {mean_q_value_loss:10.6f} | "
				f"Critic Diff: {mean_critic_diff:10.6f} | "
				f"Alpha: {mean_alpha:8.5f}"
			)

		if episode % evaluation_interval == 0:
			eval_reward, eval_episode_series = _run_evaluation(
				env=env,
				agent=agent,
				max_steps=evaluation_max_steps,
				episodes=evaluation_episodes,
				evaluation_logger=evaluation_logger,
			)
			
			latest_evaluation_episode_dti_series = eval_episode_series
			history["latest_evaluation_episode_dti_series"] = list(latest_evaluation_episode_dti_series)
			if hasattr(env, "set_logger"):
				env.set_logger(training_logger)
			if hasattr(env, "set_logging_context"):
				env.set_logging_context("training")
			history["evaluation_rewards"].append((episode, eval_reward))
			if verbose:
				evaluation_logger.info(
					f"Evaluation trigger at training episode {episode}: "
					f"mean reward={_format_optional_float(_safe_float_for_log(eval_reward), decimals=4)}"
				)

		if episode % save_interval == 0:
			episode_ckpt = checkpoint_path / f"sac_{service}_episode_{episode:04d}.pt"
			agent.save(episode_ckpt)
			episode_cfg = episode_ckpt.with_name(f"{episode_ckpt.stem}_config.json")
			save_config(config, episode_cfg)
			saved_checkpoints.append(str(episode_ckpt))
			if verbose:
				training_logger.info(f"Saved checkpoint: {episode_ckpt}")

	final_ckpt = checkpoint_path / f"sac_{service}_final.pt"
	agent.save(final_ckpt)
	final_cfg = final_ckpt.with_name(f"{final_ckpt.stem}_config.json")
	save_config(config, final_cfg)
	saved_checkpoints.append(str(final_ckpt))

	if verbose:
		training_logger.info(f"Training complete. Final checkpoint: {final_ckpt}")

	metrics_path = checkpoint_path / "training_metrics.json"
	save_training_metrics(history, metrics_path)

	evaluation_dti_plot_dir = checkpoint_path / "evaluation_dti_plots" / service
	evaluation_plots_count = _save_episode_dti_plots(latest_evaluation_episode_dti_series, evaluation_dti_plot_dir)
	if verbose:
		training_logger.info(
			f"Saved per-episode DTI evaluation plots: {evaluation_plots_count} files at {evaluation_dti_plot_dir}"
		)


	try:
		try:
			from SAC_RL_Model.plot_metrics import plot_training_metrics
		except ImportError:
			from .plot_metrics import plot_training_metrics

		generated_plots = plot_training_metrics(
			metrics=history,
			output_dir=checkpoint_path,
			service=service,
			show=False,
		)
		if verbose:
			training_logger.info(
				f"Saved training metrics: {metrics_path} | "
				f"Generated plots: {len(generated_plots)}"
			)
	except Exception as plot_error:
		training_logger.warning(f"Metrics plotting failed: {plot_error}")

	for logger in (training_logger, evaluation_logger):
		for h in list(logger.handlers):
			h.flush()
			h.close()
			logger.removeHandler(h)

	return {
		"agent": agent,
		"env": env,
		"replay_buffer": replay_buffer,
		"history": history,
		"saved_checkpoints": saved_checkpoints,
		"total_steps": total_steps,
	}

