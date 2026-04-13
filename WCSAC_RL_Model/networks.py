from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Normal


EPS: float = 1e-6
LOG_STD_MIN: float = -20.0
LOG_STD_MAX: float = 2.0


ActivationLike = Union[
	str,
	nn.Module,
	type,
	Callable[[], nn.Module],
]


def _resolve_activation(activation: ActivationLike) -> nn.Module:
	"""Resolve user-provided activation spec into an `nn.Module` instance."""
	if isinstance(activation, nn.Module):
		return activation

	if isinstance(activation, str):
		key = activation.strip().lower()
		mapping = {
			"relu": nn.ReLU,
			"tanh": nn.Tanh,
			"gelu": nn.GELU,
			"elu": nn.ELU,
			"leakyrelu": nn.LeakyReLU,
			"silu": nn.SiLU,
			"swish": nn.SiLU,
			"identity": nn.Identity,
		}
		if key not in mapping:
			raise ValueError(f"Unsupported activation string: {activation!r}")
		return mapping[key]()

	if isinstance(activation, type) and issubclass(activation, nn.Module):
		return activation()

	if callable(activation):
		module = activation()
		if not isinstance(module, nn.Module):
			raise ValueError("Callable activation must return an nn.Module")
		return module

	raise ValueError(f"Unsupported activation type: {type(activation).__name__}")


def build_mlp(
	input_dim: int,
	output_dim: int,
	hidden_dims: Sequence[int],
	activation: ActivationLike = "relu",
	output_activation: Optional[ActivationLike] = None,
) -> nn.Sequential:
	"""
	Build a fully-connected MLP.

	Args:
		input_dim: Input feature dimension.
		output_dim: Output feature dimension.
		hidden_dims: Hidden layer dimensions (e.g. [256, 256]).
		activation: Hidden-layer activation configuration.
		output_activation: Optional final activation.

	Returns:
		`nn.Sequential` MLP.
	"""
	if not isinstance(input_dim, int) or input_dim <= 0:
		raise ValueError(f"input_dim must be a positive int, got {input_dim!r}")
	if not isinstance(output_dim, int) or output_dim <= 0:
		raise ValueError(f"output_dim must be a positive int, got {output_dim!r}")
	if hidden_dims is None:
		raise ValueError("hidden_dims cannot be None")

	hidden_dims = list(hidden_dims)
	for idx, h in enumerate(hidden_dims):
		if not isinstance(h, int) or h <= 0:
			raise ValueError(f"hidden_dims[{idx}] must be a positive int, got {h!r}")

	layers: List[nn.Module] = []
	prev_dim = input_dim

	for h in hidden_dims:
		layers.append(nn.Linear(prev_dim, h))
		layers.append(_resolve_activation(activation))
		prev_dim = h

	layers.append(nn.Linear(prev_dim, output_dim))

	if output_activation is not None:
		layers.append(_resolve_activation(output_activation))

	return nn.Sequential(*layers)


class GaussianActor(nn.Module):
	"""
	Gaussian policy network used by Soft Actor-Critic.

	Given a state, it predicts Gaussian parameters (`mean`, `log_std`), samples
	with reparameterization (`rsample`), then applies `tanh` squashing.
	"""

	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		hidden_dims: Sequence[int] = (256, 256),
		activation: ActivationLike = "relu",
		log_std_min: float = LOG_STD_MIN,
		log_std_max: float = LOG_STD_MAX,
	) -> None:
		super().__init__()

		if not isinstance(state_dim, int) or state_dim <= 0:
			raise ValueError(f"state_dim must be a positive int, got {state_dim!r}")
		if not isinstance(action_dim, int) or action_dim <= 0:
			raise ValueError(f"action_dim must be a positive int, got {action_dim!r}")
		if log_std_min >= log_std_max:
			raise ValueError("log_std_min must be strictly less than log_std_max")

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.log_std_min = float(log_std_min)
		self.log_std_max = float(log_std_max)

		self.backbone = build_mlp(
			input_dim=state_dim,
			output_dim=hidden_dims[-1] if len(hidden_dims) > 0 else state_dim,
			hidden_dims=hidden_dims[:-1] if len(hidden_dims) > 1 else [],
			activation=activation,
		) if len(hidden_dims) > 0 else nn.Identity()

		last_dim = hidden_dims[-1] if len(hidden_dims) > 0 else state_dim
		self.mean_head = nn.Linear(last_dim, action_dim)
		self.log_std_head = nn.Linear(last_dim, action_dim)

	def _validate_state(self, state: torch.Tensor) -> torch.Tensor:
		if not isinstance(state, torch.Tensor):
			raise ValueError("state must be a torch.Tensor")
		if state.ndim == 1:
			state = state.unsqueeze(0)
		if state.ndim != 2:
			raise ValueError(f"state must be rank-2 [batch, state_dim], got shape {tuple(state.shape)}")
		if state.shape[-1] != self.state_dim:
			raise ValueError(
				f"Expected state_dim={self.state_dim}, got last dimension {state.shape[-1]}"
			)
		if not torch.isfinite(state).all():
			raise ValueError("state contains non-finite values")
		return state

	def distribution_parameters(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Compute Gaussian parameters `(mean, log_std)` for a batch of states."""
		state = self._validate_state(state)
		features = self.backbone(state)
		mean = self.mean_head(features)
		log_std = self.log_std_head(features)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		return mean, log_std

	def forward(
		self,
		state: torch.Tensor,
		deterministic: bool = False,
		with_logprob: bool = True,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
		"""
		Run policy forward pass.

		Args:
			state: State tensor `[batch, state_dim]` or `[state_dim]`.
			deterministic: If True, use mean action (no noise).
			with_logprob: If True, return corrected log-probability.

		Returns:
			action: Squashed action in `[-1, 1]`, shape `[batch, action_dim]`.
			log_prob: Corrected log-probability, shape `[batch, 1]` (or None).
			mean_action: Squashed mean action, shape `[batch, action_dim]`.
		"""
		mean, log_std = self.distribution_parameters(state)
		std = torch.exp(log_std)

		dist = Normal(mean, std)
		pre_tanh = mean if deterministic else dist.rsample()
		action = torch.tanh(pre_tanh)
		mean_action = torch.tanh(mean)

		log_prob: Optional[torch.Tensor]
		if with_logprob:
			# SAC tanh correction term:
			# log pi(a|s) = log N(u; mean, std) - sum(log(1 - tanh(u)^2))
			raw_log_prob = dist.log_prob(pre_tanh)
			correction = torch.log(1.0 - action.pow(2) + EPS)
			log_prob = (raw_log_prob - correction).sum(dim=-1, keepdim=True)

			if not torch.isfinite(log_prob).all():
				raise RuntimeError("Non-finite log_prob encountered in actor forward pass")
		else:
			log_prob = None

		return action, log_prob, mean_action


class Critic(nn.Module):
	"""
	Q-network for SAC.

	Input: concatenated `(state, action)`.
	Output: scalar Q-value.
	"""

	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		hidden_dims: Sequence[int] = (256, 256),
		activation: ActivationLike = "relu",
	) -> None:
		super().__init__()

		if not isinstance(state_dim, int) or state_dim <= 0:
			raise ValueError(f"state_dim must be a positive int, got {state_dim!r}")
		if not isinstance(action_dim, int) or action_dim <= 0:
			raise ValueError(f"action_dim must be a positive int, got {action_dim!r}")

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.q_net = build_mlp(
			input_dim=state_dim + action_dim,
			output_dim=1,
			hidden_dims=hidden_dims,
			activation=activation,
		)

	def _validate_inputs(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if not isinstance(state, torch.Tensor) or not isinstance(action, torch.Tensor):
			raise ValueError("state and action must be torch.Tensor")

		if state.ndim == 1:
			state = state.unsqueeze(0)
		if action.ndim == 1:
			action = action.unsqueeze(0)

		if state.ndim != 2 or action.ndim != 2:
			raise ValueError(
				f"state and action must be rank-2; got {tuple(state.shape)} and {tuple(action.shape)}"
			)
		if state.shape[-1] != self.state_dim:
			raise ValueError(f"Expected state last dim {self.state_dim}, got {state.shape[-1]}")
		if action.shape[-1] != self.action_dim:
			raise ValueError(f"Expected action last dim {self.action_dim}, got {action.shape[-1]}")
		if state.shape[0] != action.shape[0]:
			raise ValueError("state and action batch sizes must match")
		if not torch.isfinite(state).all() or not torch.isfinite(action).all():
			raise ValueError("state/action contains non-finite values")
		return state, action

	def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""Compute scalar Q-value for each `(state, action)` pair."""
		state, action = self._validate_inputs(state, action)
		x = torch.cat([state, action], dim=-1)
		q_value = self.q_net(x)

		if q_value.ndim != 2 or q_value.shape[-1] != 1:
			raise RuntimeError(f"Critic output must have shape [batch, 1], got {tuple(q_value.shape)}")
		if not torch.isfinite(q_value).all():
			raise RuntimeError("Non-finite Q-value encountered in critic forward pass")
		return q_value
