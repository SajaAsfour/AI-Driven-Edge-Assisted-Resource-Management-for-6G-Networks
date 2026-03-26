from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
	from .agent import SACAgent
	from .env_wrapper import NetworkSACEnv
	from .replay_buffer import ReplayBuffer


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
	service: str = "voip",
	max_episodes: int = 200,
	max_steps_per_episode: int = 128,
	batch_size: int = 64,
	warmup_steps: int = 1_000,
	evaluation_interval: int = 20,
	save_interval: int = 25,
	replay_capacity: int = 100_000,
	rb_min: int = 1,
	checkpoint_dir: Union[str, Path] = "RL_Model/checkpoints",
	seed: Optional[int] = 42,
	verbose: bool = True,
	) -> Dict[str, Any]:
	"""Train a Soft Actor-Critic agent on the network environment.

	Args:
		service: Target service name (voip, cbr, streaming).
		max_episodes: Number of episodes to train.
		max_steps_per_episode: Max steps before forcing episode end.
		batch_size: Replay minibatch size for SAC updates.
		warmup_steps: Steps collected before gradient updates start.
		evaluation_interval: Run deterministic evaluation every N episodes.
		save_interval: Save model checkpoint every N episodes.
		replay_capacity: Replay buffer size.
		rb_min: Minimum RB allocation allowed by the environment action mapping.
		checkpoint_dir: Directory for periodic checkpoints.
		seed: Optional random seed.
		verbose: Print progress to console.

	Returns:
		Dictionary with training history and paths to saved checkpoints.
	"""
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

	agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
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
			saved_checkpoints.append(str(episode_ckpt))
			if verbose:
				print(f"  Saved checkpoint: {episode_ckpt}")

	final_ckpt = checkpoint_path / f"sac_{service}_final.pt"
	agent.save(final_ckpt)
	saved_checkpoints.append(str(final_ckpt))

	if verbose:
		print(f"Training complete. Final checkpoint: {final_ckpt}")

	return {
		"agent": agent,
		"env": env,
		"replay_buffer": replay_buffer,
		"history": history,
		"saved_checkpoints": saved_checkpoints,
		"total_steps": total_steps,
	}

