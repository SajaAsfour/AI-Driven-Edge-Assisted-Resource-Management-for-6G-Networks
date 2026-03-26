from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

try:
	from .networks import Critic, GaussianActor
except ImportError:
	from networks import Critic, GaussianActor


class SACAgent:
	"""Soft Actor-Critic (SAC) agent for continuous action spaces.

	This implementation follows the standard SAC formulation with:
	- Twin Q critics and target critics
	- Tanh-squashed Gaussian policy (reparameterization trick)
	- Automatic entropy temperature tuning via learnable log_alpha
	"""

	def __init__(
		self,
		state_dim: int = 22,
		action_dim: int = 1,
		hidden_dims: tuple[int, ...] = (256, 256),
		gamma: float = 0.99,
		tau: float = 0.005,
		actor_lr: float = 3e-4,
		critic_lr: float = 3e-4,
		alpha_lr: float = 3e-4,
		target_entropy: Optional[float] = None,
		device: Optional[Union[str, torch.device]] = None,
	) -> None:
		if not isinstance(state_dim, int) or state_dim <= 0:
			raise ValueError(f"state_dim must be positive int, got {state_dim!r}")
		if not isinstance(action_dim, int) or action_dim <= 0:
			raise ValueError(f"action_dim must be positive int, got {action_dim!r}")
		if gamma <= 0.0 or gamma > 1.0:
			raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")
		if tau <= 0.0 or tau > 1.0:
			raise ValueError(f"tau must be in (0, 1], got {tau!r}")

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.gamma = float(gamma)
		self.tau = float(tau)

		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device)

		self.actor = GaussianActor(
			state_dim=state_dim,
			action_dim=action_dim,
			hidden_dims=hidden_dims,
		).to(self.device)

		self.critic1 = Critic(
			state_dim=state_dim,
			action_dim=action_dim,
			hidden_dims=hidden_dims,
		).to(self.device)
		self.critic2 = Critic(
			state_dim=state_dim,
			action_dim=action_dim,
			hidden_dims=hidden_dims,
		).to(self.device)

		self.target_critic1 = Critic(
			state_dim=state_dim,
			action_dim=action_dim,
			hidden_dims=hidden_dims,
		).to(self.device)
		self.target_critic2 = Critic(
			state_dim=state_dim,
			action_dim=action_dim,
			hidden_dims=hidden_dims,
		).to(self.device)

		self.target_critic1.load_state_dict(self.critic1.state_dict())
		self.target_critic2.load_state_dict(self.critic2.state_dict())

		for p in self.target_critic1.parameters():
			p.requires_grad = False
		for p in self.target_critic2.parameters():
			p.requires_grad = False

		self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
		self.critic1_optimizer = Adam(self.critic1.parameters(), lr=critic_lr)
		self.critic2_optimizer = Adam(self.critic2.parameters(), lr=critic_lr)

		# log_alpha is optimized directly to keep alpha positive via exp(log_alpha).
		self.log_alpha = nn.Parameter(torch.tensor(0.0, device=self.device))
		self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

		self.target_entropy = float(target_entropy) if target_entropy is not None else float(-action_dim)

		self.alpha = self.log_alpha.exp().detach()

	def select_action(self, state: Union[np.ndarray, list, tuple], evaluate: bool = False) -> np.ndarray:
		"""Select action from current policy.

		Args:
			state: Flat state vector shape [state_dim].
			evaluate: If True, use deterministic action (policy mean).

		Returns:
			Action as NumPy array with shape [action_dim].
		"""
		state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
		if state_arr.shape[0] != self.state_dim:
			raise ValueError(
				f"state must flatten to shape ({self.state_dim},), got {state_arr.shape}"
			)

		state_tensor = torch.as_tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
		with torch.no_grad():
			action_tensor, _, _ = self.actor(
				state_tensor,
				deterministic=evaluate,
				with_logprob=False,
			)

		action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
		return action

	def update(self, replay_buffer, batch_size: int) -> Dict[str, float]:
		"""Run one SAC training update from a sampled replay batch.

		Update order:
		1) Critic update (Bellman backup using target critics)
		2) Actor update (maximize Q + entropy)
		3) Temperature update (automatic entropy tuning)
		4) Soft-update target critics

		Args:
			replay_buffer: Buffer exposing sample(batch_size) -> dict of NumPy arrays.
			batch_size: Number of transitions sampled per update.

		Returns:
			Dictionary of scalar training metrics.
		"""
		batch = replay_buffer.sample(batch_size)

		states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
		actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
		rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
		next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
		dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

		with torch.no_grad():
			next_actions, next_log_prob, _ = self.actor(
				next_states,
				deterministic=False,
				with_logprob=True,
			)
			target_q1_next = self.target_critic1(next_states, next_actions)
			target_q2_next = self.target_critic2(next_states, next_actions)
			target_q_next = torch.min(target_q1_next, target_q2_next)

			alpha = self.log_alpha.exp()
			target_v = target_q_next - alpha * next_log_prob
			target_q = rewards + (1.0 - dones) * self.gamma * target_v

		current_q1 = self.critic1(states, actions)
		current_q2 = self.critic2(states, actions)
		critic1_loss = F.mse_loss(current_q1, target_q)
		critic2_loss = F.mse_loss(current_q2, target_q)

		self.critic1_optimizer.zero_grad(set_to_none=True)
		critic1_loss.backward()
		self.critic1_optimizer.step()

		self.critic2_optimizer.zero_grad(set_to_none=True)
		critic2_loss.backward()
		self.critic2_optimizer.step()

		new_actions, log_prob, _ = self.actor(
			states,
			deterministic=False,
			with_logprob=True,
		)
		q1_new = self.critic1(states, new_actions)
		q2_new = self.critic2(states, new_actions)
		min_q_new = torch.min(q1_new, q2_new)

		alpha_detached = self.log_alpha.exp().detach()
		actor_loss = (alpha_detached * log_prob - min_q_new).mean()

		self.actor_optimizer.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_optimizer.step()

		alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

		self.alpha_optimizer.zero_grad(set_to_none=True)
		alpha_loss.backward()
		self.alpha_optimizer.step()

		self.alpha = self.log_alpha.exp().detach()

		self.soft_update(self.target_critic1, self.critic1, self.tau)
		self.soft_update(self.target_critic2, self.critic2, self.tau)

		return {
			"critic1_loss": float(critic1_loss.item()),
			"critic2_loss": float(critic2_loss.item()),
			"actor_loss": float(actor_loss.item()),
			"alpha_loss": float(alpha_loss.item()),
			"alpha": float(self.alpha.item()),
			"q1_mean": float(current_q1.mean().item()),
			"q2_mean": float(current_q2.mean().item()),
			"log_prob_mean": float(log_prob.mean().item()),
		}

	@staticmethod
	def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
		"""Polyak averaging: target = (1 - tau) * target + tau * source."""
		if tau <= 0.0 or tau > 1.0:
			raise ValueError(f"tau must be in (0, 1], got {tau!r}")
		with torch.no_grad():
			for target_param, source_param in zip(target.parameters(), source.parameters()):
				target_param.data.mul_(1.0 - tau)
				target_param.data.add_(tau * source_param.data)

	def save(self, path: Union[str, Path]) -> None:
		"""Save model and optimizer states to disk."""
		save_path = Path(path)
		save_path.parent.mkdir(parents=True, exist_ok=True)

		checkpoint = {
			"state_dim": self.state_dim,
			"action_dim": self.action_dim,
			"gamma": self.gamma,
			"tau": self.tau,
			"target_entropy": self.target_entropy,
			"actor": self.actor.state_dict(),
			"critic1": self.critic1.state_dict(),
			"critic2": self.critic2.state_dict(),
			"target_critic1": self.target_critic1.state_dict(),
			"target_critic2": self.target_critic2.state_dict(),
			"actor_optimizer": self.actor_optimizer.state_dict(),
			"critic1_optimizer": self.critic1_optimizer.state_dict(),
			"critic2_optimizer": self.critic2_optimizer.state_dict(),
			"log_alpha": self.log_alpha.detach().cpu(),
			"alpha_optimizer": self.alpha_optimizer.state_dict(),
		}
		torch.save(checkpoint, save_path)

	def load(self, path: Union[str, Path]) -> None:
		"""Load model and optimizer states from disk."""
		load_path = Path(path)
		if not load_path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {load_path}")

		checkpoint = torch.load(load_path, map_location=self.device)

		self.actor.load_state_dict(checkpoint["actor"])
		self.critic1.load_state_dict(checkpoint["critic1"])
		self.critic2.load_state_dict(checkpoint["critic2"])
		self.target_critic1.load_state_dict(checkpoint["target_critic1"])
		self.target_critic2.load_state_dict(checkpoint["target_critic2"])

		self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
		self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
		self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
		self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

		loaded_log_alpha = checkpoint["log_alpha"].to(self.device)
		with torch.no_grad():
			self.log_alpha.copy_(loaded_log_alpha)
		self.alpha = self.log_alpha.exp().detach()

		if "target_entropy" in checkpoint:
			self.target_entropy = float(checkpoint["target_entropy"])
