"""Replay buffer implementation for WCSAC (off-policy RL).

This module provides a NumPy-backed circular replay buffer that stores
transitions in the form:
	(state, action, reward, next_state, done)
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class ReplayBuffer:
	"""Fixed-size circular replay buffer for continuous-control WCSAC.

	Parameters
	----------
	capacity : int
		Maximum number of transitions to store.
	state_dim : int
		Dimension of flattened state vector.
	action_dim : int
		Dimension of continuous action vector.
	"""

	def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
		if not isinstance(capacity, int) or capacity <= 0:
			raise ValueError(f"capacity must be a positive int, got {capacity!r}")
		if not isinstance(state_dim, int) or state_dim <= 0:
			raise ValueError(f"state_dim must be a positive int, got {state_dim!r}")
		if not isinstance(action_dim, int) or action_dim <= 0:
			raise ValueError(f"action_dim must be a positive int, got {action_dim!r}")

		self.capacity = capacity
		self.state_dim = state_dim
		self.action_dim = action_dim

		self._states = np.zeros((capacity, state_dim), dtype=np.float32)
		self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
		self._rewards = np.zeros((capacity, 1), dtype=np.float32)
		# cost signal for constrained RL (e.g., beta_current or exceedance)
		self._costs = np.zeros((capacity, 1), dtype=np.float32)
		self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
		self._dones = np.zeros((capacity, 1), dtype=np.float32)

		self._ptr = 0
		self._size = 0

	def __len__(self) -> int:
		"""Return the current number of valid transitions in the buffer."""
		return self._size

	def _as_flat_vector(self, name: str, value: np.ndarray | list | tuple, dim: int) -> np.ndarray:
		arr = np.asarray(value, dtype=np.float32).reshape(-1)
		if arr.shape[0] != dim:
			raise ValueError(
				f"{name} must have shape ({dim},) after flattening, got {arr.shape}"
			)
		if not np.all(np.isfinite(arr)):
			raise ValueError(f"{name} contains non-finite values")
		return arr

	def add(
		self,
		state: np.ndarray | list | tuple,
		action: np.ndarray | list | tuple,
		reward: float,
		cost: float,
		next_state: np.ndarray | list | tuple,
		done: bool | float | int,
	) -> None:
		"""Add one transition to the replay buffer.

		Parameters
		----------
		state : array-like
			Current state, expected flat length = state_dim.
		action : array-like
			Continuous action, expected flat length = action_dim.
		reward : float
			Scalar reward for the transition.
		next_state : array-like
			Next state, expected flat length = state_dim.
		done : bool | float | int
			Episode termination flag. Stored as float32 (0.0 or 1.0).
		"""
		state_arr = self._as_flat_vector("state", state, self.state_dim)
		action_arr = self._as_flat_vector("action", action, self.action_dim)
		next_state_arr = self._as_flat_vector("next_state", next_state, self.state_dim)

		if not isinstance(reward, (int, float, np.floating, np.integer)):
			raise ValueError(f"reward must be numeric, got {type(reward).__name__}")
		reward_val = float(reward)
		if not np.isfinite(reward_val):
			raise ValueError(f"reward must be finite, got {reward!r}")

		if not isinstance(cost, (int, float, np.floating, np.integer)):
			raise ValueError(f"cost must be numeric, got {type(cost).__name__}")
		cost_val = float(cost)
		if not np.isfinite(cost_val):
			raise ValueError(f"cost must be finite, got {cost!r}")

		if isinstance(done, (bool, np.bool_)):
			done_val = float(done)
		elif isinstance(done, (int, float, np.integer, np.floating)):
			done_val = float(done)
			if done_val not in (0.0, 1.0):
				raise ValueError(f"done numeric value must be 0 or 1, got {done!r}")
		else:
			raise ValueError(f"done must be bool or numeric 0/1, got {type(done).__name__}")

		idx = self._ptr
		self._states[idx] = state_arr
		self._actions[idx] = action_arr
		self._rewards[idx, 0] = reward_val
		self._costs[idx, 0] = cost_val
		self._next_states[idx] = next_state_arr
		self._dones[idx, 0] = done_val

		self._ptr = (self._ptr + 1) % self.capacity
		self._size = min(self._size + 1, self.capacity)

	def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
		"""Sample a random minibatch of transitions.

		Parameters
		----------
		batch_size : int
			Number of transitions to sample.

		Returns
		-------
		Dict[str, np.ndarray]
			Batch dictionary with keys:
			- "states": shape (B, state_dim)
			- "actions": shape (B, action_dim)
			- "rewards": shape (B, 1)
			- "next_states": shape (B, state_dim)
			- "dones": shape (B, 1)
		"""
		if not isinstance(batch_size, int) or batch_size <= 0:
			raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
		if self._size == 0:
			raise ValueError("Cannot sample from an empty replay buffer")
		if batch_size > self._size:
			raise ValueError(
				f"batch_size ({batch_size}) cannot exceed current buffer size ({self._size})"
			)

		indices = np.random.choice(self._size, size=batch_size, replace=False)

		return {
			"states": self._states[indices],
			"actions": self._actions[indices],
			"rewards": self._rewards[indices],
			"costs": self._costs[indices],
			"next_states": self._next_states[indices],
			"dones": self._dones[indices],
		}

