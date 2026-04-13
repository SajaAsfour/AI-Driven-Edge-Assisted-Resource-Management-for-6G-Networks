from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import sys
import logging

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from WCSAC_Main import load_configuration, load_metric_matrices
from Network_Model.src.NetworkModel import NetworkModel
from WCSAC_RL_Model.traffic_profiles import (
	build_dti_from_profile,
	build_traffic_matrix_from_profile,
	get_default_profiles,
	get_profile_or_raise,
	validate_dti_values_in_profile,
)


Number = Union[int, float]


class NetworkWCSACEnv:
	"""
	This wrapper is intentionally lightweight:
	- one service at a time (`voip`, `cbr`, or `streaming`)
	- single continuous WCSAC action mapped to an integer RB allocation

	Notes on integration with current simulation:
	- Traffic demand per DTI is profile-driven (fixed profile or per-step random profile).
	- The wrapper does NOT re-implement network logic.
	  It calls `NetworkModel.process_dti(...)` and uses the returned reward.
	- The action controls RB allocation for the selected service in current DTI,
	  replicated across TTIs in that DTI (same pattern your current code uses).
	"""
	SERVICE_FILE_NAMES: Dict[str, str] = {
		"voip": "D2min_VoIP_summary.json",
		"cbr": "D30sec_CBR_summary.json",
		"streaming": "D90sec_VideoStream_summary.json",
	}

	SERVICE_ALIASES: Dict[str, str] = {
		"voip": "voip",
		"cbr": "cbr",
		"streaming": "streaming",
		"videostream": "streaming",
		"video_stream": "streaming",
		"video-stream": "streaming",
	}

	def __init__(
		self,
		service: str = "voip",
		traffic_profile_mode: str = "fixed",
		fixed_profile_name: str = "profile_1",
		config_path: Optional[Union[str, Path]] = None,
		input_path: Optional[Union[str, Path]] = None,
		metric_file: Optional[Union[str, Path]] = None,
		action_low: float = -1.0,
		action_high: float = 1.0,
		rb_min: int = 0,
		rb_max: Optional[int] = None,
		seed: Optional[int] = None,
		log_random_profile_each_step: bool = False,
		silent: bool = False,
	) -> None:
		"""
		Initialize the RL wrapper.

		Args:
			service: Target service (`voip`, `cbr`, `streaming`, `videostream`).
			config_path: Optional path to network config JSON.
			input_path: Optional path to network input JSON.
			metric_file: Optional path to per-service metric summary JSON.
			action_low: Minimum WCSAC action value before mapping.
			action_high: Maximum WCSAC action value before mapping.
			rb_min: Minimum RB decision for one DTI.
			rb_max: Maximum RB decision for one DTI. Defaults to config `c`.
			seed: Optional numpy RNG seed.
		"""
		if seed is not None:
			np.random.seed(seed)
		self._rng = np.random.default_rng(seed)

		self.service: str = self._normalize_service(service)

		config_dir = PROJECT_ROOT / "Network_Model" / "data" / "configuration"
		input_dir = PROJECT_ROOT / "Network_Model" / "data" / "input"

		cfg_path = Path(config_path) if config_path is not None else config_dir / "network_config.json"
		inp_path = Path(input_path) if input_path is not None else input_dir / "network_input.json"

		self.config: Dict[str, Any] = load_configuration(cfg_path, inp_path)

		self.n: int = int(self.config["n"])
		self.m: int = int(self.config["m"])
		self.k: int = int(self.config["k"])
		self.c: float = float(self.config["c"])
		self.lambda_reward: float = float(self.config["lambda_reward"])

		self.traffic_profile_mode: str = self._normalize_profile_mode(traffic_profile_mode)
		self.fixed_profile_name: str = str(fixed_profile_name)
		self.log_random_profile_each_step: bool = bool(log_random_profile_each_step)
		self.silent: bool = bool(silent)
		self.ue_profiles: Dict[str, list[int]] = get_default_profiles()
		if not self.ue_profiles:
			raise ValueError("UE profiles cannot be empty")

		self._profile_names: list[str] = list(self.ue_profiles.keys())
		self._traffic_by_dti: Optional[list[list[int]]] = None
		self._current_profile_name: Optional[str] = None
		self._current_profile_values: Optional[list[int]] = None

		self.action_low = float(action_low)
		self.action_high = float(action_high)
		if not np.isfinite(self.action_low) or not np.isfinite(self.action_high):
			raise ValueError("action_low and action_high must be finite")
		if self.action_high <= self.action_low:
			raise ValueError("action_high must be greater than action_low")

		self.rb_min = int(rb_min)
		if self.rb_min < 0:
			raise ValueError("rb_min must be >= 0")

		default_rb_max = int(self.c)
		self.rb_max = int(default_rb_max if rb_max is None else rb_max)
		if self.rb_max < self.rb_min:
			raise ValueError("rb_max must be >= rb_min")

		self.model = NetworkModel(
			n=self.n,
			m=self.m,
			k=self.k,
			traffic_elements=self.config["traffic_elements"],
			q_thresholds_voip=self.config["q_thresholds_voip"],
			q_thresholds_cbr=self.config["q_thresholds_cbr"],
			q_thresholds_streaming=self.config["q_thresholds_streaming"],
		)

		metric_path = (
			Path(metric_file)
			if metric_file is not None
			else config_dir / self.SERVICE_FILE_NAMES[self.service]
		)
		metric_data = load_metric_matrices(self.service, metric_path)
		self.model.set_metric_matrices(self.service, metric_data)
		self.model.set_service(self.service)

		if self.traffic_profile_mode == "fixed":
			profile_name, profile_values = get_profile_or_raise(self.ue_profiles, self.fixed_profile_name)
			self._current_profile_name = profile_name
			self._current_profile_values = list(profile_values)
			self._traffic_by_dti = build_traffic_matrix_from_profile(
				profile_values=self._current_profile_values,
				m=self.m,
				n=self.n,
			)
			if not self.silent:
				print(f"Using fixed UE profile: {profile_name} -> {self._current_profile_values}")
		else:
			self._traffic_by_dti = None
			if not self.silent:
				print("Using dynamic random UE profiles (changes every step)")

		self._dti_cursor: int = 0
		self._done: bool = False
		self._last_beta_current: float = 0.0
		self._last_beta_cumulative: float = 0.0
		self._last_reward: float = 0.0
		self._last_rb_norm: float = 0.0
		self._last_cdf_y: np.ndarray = np.zeros(self.k, dtype=np.float32)
		self._current_traffic_dti: np.ndarray | list[int] = [0] * self.n
		self._is_current_dti_prepared: bool = False
		self.logging_context: str = "training"
		self._logger: Optional[logging.Logger] = None

	def reset(self) -> np.ndarray:
		"""
		Reset episode state and return initial flat state vector.

		Returns:
			Flat NumPy vector suitable as neural-network input.
		"""
		self.model.set_service(self.service)
		self.model.reset(self.service)

		self._dti_cursor = 0
		self._done = False
		self._last_beta_current = 0.0
		self._last_beta_cumulative = 0.0
		self._last_reward = 0.0
		self._last_rb_norm = 0.0
		self._last_cdf_y = np.zeros(self.k, dtype=np.float32)
		self._current_traffic_dti = [0] * self.n
		self._is_current_dti_prepared = False
		if self.traffic_profile_mode == "fixed":
			profile_name, profile_values = get_profile_or_raise(self.ue_profiles, self.fixed_profile_name)
			self._current_profile_name = profile_name
			self._current_profile_values = list(profile_values)
		else:
			self._current_profile_name = None
			self._current_profile_values = None
		return self.prepare_current_dti()

	def set_logging_context(self, context: str) -> None:
		"""Set step diagnostics logging context (training or evaluation)."""
		context_str = str(context).strip().lower()
		if context_str not in {"training", "evaluation"}:
			raise ValueError("context must be either 'training' or 'evaluation'")
		self.logging_context = context_str

	def set_logger(self, logger: Optional[logging.Logger]) -> None:
		"""Attach logger used for per-step diagnostics in step()."""
		if logger is not None and not isinstance(logger, logging.Logger):
			raise TypeError("logger must be an instance of logging.Logger or None")
		self._logger = logger

	def _safe_numeric_for_log(self, value: Any) -> Optional[float]:
		"""Return finite float value for logging, otherwise None."""
		try:
			value_float = float(value)
		except (TypeError, ValueError):
			return None
		if not np.isfinite(value_float):
			return None
		return value_float

	@staticmethod
	def _safe_utilization(value: Any) -> float:
		"""Return utilization clipped to [0, 1], else NaN for invalid values."""
		try:
			value_float = float(value)
		except (TypeError, ValueError):
			return float("nan")
		if not np.isfinite(value_float):
			return float("nan")
		return float(np.clip(value_float, 0.0, 1.0))

	@staticmethod
	def _format_log_value(
		value: Optional[float],
		decimals: int = 4,
		integer_if_whole: bool = False,
	) -> str:
		if value is None:
			return "None"
		if integer_if_whole and float(value).is_integer():
			return str(int(value))
		return f"{float(value):.{decimals}f}"

	def _log_step_reward_diagnostics(
		self,
		dti_index: int,
		rb_used: Number,
		utilization: Number,
		beta_current: Number,
		reward_current: Number,
		done: bool,
	) -> None:
		"""Emit detailed reward diagnostics for every environment step."""
		logger = self._logger
		if logger is None:
			logger = logging.getLogger()

		beta_val = self._safe_numeric_for_log(beta_current)
		rb_used_val = self._safe_numeric_for_log(rb_used)
		utilization_val = self._safe_numeric_for_log(utilization)
		c_val = self._safe_numeric_for_log(self.c)
		lambda_val = self._safe_numeric_for_log(self.lambda_reward)
		reward_val = self._safe_numeric_for_log(reward_current)

		resource_term_val: Optional[float] = None
		if (
			lambda_val is not None
			and c_val is not None
			and rb_used_val is not None
			and c_val != 0.0
		):
			resource_term_val = self._safe_numeric_for_log(
				lambda_val * ((c_val - rb_used_val) / c_val)
			)

		logger.info("----------------------------------------")
		logger.info(f"Context: {self.logging_context}")
		logger.info(f"DTI Index: {int(dti_index)}")
		logger.info(f"Service: {self.service}")
		logger.info("")
		logger.info(f"beta_current = {self._format_log_value(beta_val, decimals=4)}")
		logger.info(f"rb_used = {self._format_log_value(rb_used_val, decimals=4, integer_if_whole=True)}")
		logger.info(f"C = {self._format_log_value(c_val, decimals=4, integer_if_whole=True)}")
		logger.info(
			"utilization = rb_used / C = "
			f"{self._format_log_value(rb_used_val, decimals=4, integer_if_whole=True)} / "
			f"{self._format_log_value(c_val, decimals=4, integer_if_whole=True)} = "
			f"{self._format_log_value(utilization_val, decimals=4)}"
		)
		logger.info(f"lambda_reward = {self._format_log_value(lambda_val, decimals=4)}")
		logger.info("")
		logger.info(
			"resource_term = "
			f"{self._format_log_value(lambda_val, decimals=4)} * "
			f"(({self._format_log_value(c_val, decimals=4, integer_if_whole=True)} - "
			f"{self._format_log_value(rb_used_val, decimals=4, integer_if_whole=True)}) / "
			f"{self._format_log_value(c_val, decimals=4, integer_if_whole=True)}) = "
			f"{self._format_log_value(resource_term_val, decimals=4)}"
		)
		logger.info("")
		logger.info("reward_current = -beta_current + lambda_reward * ((C - rb_used) / C)")
		logger.info(
			"reward_current = "
			f"-{self._format_log_value(beta_val, decimals=4)} + "
			f"{self._format_log_value(resource_term_val, decimals=4)}"
		)
		logger.info(f"reward_current = {self._format_log_value(reward_val, decimals=4)}")
		logger.info("")
		logger.info("Step Summary:")
		logger.info(f"- beta_current: {self._format_log_value(beta_val, decimals=4)}")
		logger.info(f"- rb_used: {self._format_log_value(rb_used_val, decimals=4, integer_if_whole=True)}")
		logger.info(f"- utilization: {self._format_log_value(utilization_val, decimals=4)}")
		logger.info(f"- reward_current: {self._format_log_value(reward_val, decimals=4)}")
		logger.info(f"- done: {bool(done)}")
		logger.info("----------------------------------------")

	def _log_dti_traffic_details(
		self,
		dti_index: int,
		profile_name: str,
		profile_values: list[int],
		traffic_dti: list[int],
	) -> None:
		"""Log generated per-TTI traffic values for current DTI."""
		if self._logger is not None:
			logger = self._logger
			logger.info("-" * 50)
			logger.info(
				f"DTI {int(dti_index) + 1} | Profile: {profile_name} -> {list(profile_values)}"
			)
			logger.info(f"Traffic TTIs: {list(traffic_dti)}")
			logger.info("-" * 50)
		else:
			print("-" * 50)
			print(f"DTI {int(dti_index) + 1} | Profile: {profile_name} -> {list(profile_values)}")
			print(f"Traffic TTIs: {list(traffic_dti)}")
			print("-" * 50)

	def prepare_current_dti(self) -> np.ndarray:
		"""
		Prepare traffic/profile for the current DTI cursor and return full state.

		This makes current DTI traffic available before action selection.
		"""
		if self._done or self._dti_cursor >= self.m:
			raise RuntimeError("Cannot prepare current DTI because episode is done.")

		if self.traffic_profile_mode == "fixed":
			if self._traffic_by_dti is None:
				raise ValueError("Fixed profile traffic matrix was not initialized")
			traffic_dti = self._traffic_by_dti[self._dti_cursor]
			if self._current_profile_name is None or self._current_profile_values is None:
				profile_name, profile_values = get_profile_or_raise(self.ue_profiles, self.fixed_profile_name)
				self._current_profile_name = profile_name
				self._current_profile_values = list(profile_values)
		else:
			profile_name = str(self._rng.choice(self._profile_names))
			profile_values = [int(v) for v in self.ue_profiles[profile_name]]
			traffic_dti = build_dti_from_profile(profile_values=profile_values, n=self.n)
			self._current_profile_name = profile_name
			self._current_profile_values = profile_values
			if self.log_random_profile_each_step:
				if self._logger is not None:
					self._logger.info(
						f"Step {self._dti_cursor + 1} -> {profile_name} -> {profile_values}"
					)
				else:
					print(
						f"Step {self._dti_cursor + 1} -> {profile_name} -> {profile_values}"
					)

		if self._current_profile_values is None or self._current_profile_name is None:
			raise ValueError("Profile data is missing for current DTI preparation")

		self._log_dti_traffic_details(
			dti_index=self._dti_cursor,
			profile_name=self._current_profile_name,
			profile_values=list(self._current_profile_values),
			traffic_dti=list(traffic_dti),
		)
		validate_dti_values_in_profile(
			dti=traffic_dti,
			profile_values=self._current_profile_values,
			n=self.n,
		)

		self._current_traffic_dti = list(traffic_dti)
		self._is_current_dti_prepared = True
		return self.get_state()

	def step(self, action: Union[Number, np.ndarray, list, tuple]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
		"""
		Execute one DTI transition.

		Args:
			action: Continuous WCSAC action. It is mapped to integer RB allocation.

		Returns:
			next_state: Flat NumPy state vector.
			reward: Immediate reward for current DTI.
			done: Episode completion flag.
			info: Extra diagnostics (beta values, chosen RB, indices).
		"""
		if self._done:
			raise RuntimeError("Episode is done. Call reset() before step().")
		if not self._is_current_dti_prepared:
			raise RuntimeError("Current DTI traffic is not prepared. Call reset() before step().")

		traffic_dti = np.asarray(self._current_traffic_dti, dtype=np.float32).reshape(-1).tolist()
		if len(traffic_dti) != self.n:
			raise ValueError(
				f"Prepared current traffic must have length n={self.n}, got {len(traffic_dti)}"
			)
		traffic_dti = [int(np.round(x)) for x in traffic_dti]
		if self._current_profile_values is None or self._current_profile_name is None:
			raise ValueError("Profile data is missing for current step")
		current_profile_name = str(self._current_profile_name)
		current_profile_values = list(self._current_profile_values)

		rb_alloc = self._action_to_rb(action)
		try:
			utilization_raw = float(rb_alloc) / float(self.c)
		except (TypeError, ValueError, ZeroDivisionError):
			utilization_raw = float("nan")
		utilization = self._safe_utilization(utilization_raw)
		rb_data = [rb_alloc] * self.n

		self.model.set_traffic(traffic_dti)
		self.model.set_resource_blocks(rb_data)

		dti_result = self.model.process_dti(
			traffic_data=traffic_dti,
			rb_data=rb_data,
			c_capacity=self.c,
			rb_used=rb_alloc,
			lambda_reward=self.lambda_reward,
		)

		(beta_current, cdf_matrix), reward_current = self.model.to_rl_input(dti_result)
		self._last_beta_current = float(beta_current)
		self._last_beta_cumulative = float(dti_result.beta_result.beta_cumulative)
		self._last_reward = float(reward_current)
		self._last_rb_norm = self._normalize_rb(rb_alloc)
		self._last_cdf_y = np.asarray(cdf_matrix[:, 1], dtype=np.float32)

		self._dti_cursor += 1
		self._done = self._dti_cursor >= self.m
		self._is_current_dti_prepared = False
		self._log_step_reward_diagnostics(
			dti_index=int(dti_result.dti_index),
			rb_used=rb_alloc,
			utilization=utilization,
			beta_current=dti_result.beta_result.beta_current,
			reward_current=dti_result.reward_current,
			done=self._done,
		)

		if not self._done:
			next_state = self.prepare_current_dti()
		else:
			next_state = self.get_state()
		info = {
			"service": self.service,
			"logging_context": self.logging_context,
			"dti_index": int(dti_result.dti_index),
			"cursor": int(self._dti_cursor),
			"rb_alloc": int(rb_alloc),
			"capacity": float(self.c),
			"C": float(self.c),
			"utilization": float(utilization),
			"profile_mode": self.traffic_profile_mode,
			"profile_name": current_profile_name,
			"profile_values": current_profile_values,
			"traffic_dti": list(traffic_dti),
			"beta_current": float(dti_result.beta_result.beta_current),
			"beta_cumulative": float(dti_result.beta_result.beta_cumulative),
			"reward_current": float(dti_result.reward_current),
		}
		return next_state, float(dti_result.reward_current), self._done, info

	def get_state(self) -> np.ndarray:
		"""
		Build a flat state vector from current wrapper/model status.

		State layout:
			[progress,
			 last_beta_current,
			 last_beta_cumulative,
			 last_rb_norm,
			 cdf_y[0], ..., cdf_y[k-1],
			 traffic_dti_norm[0], ..., traffic_dti_norm[n-1]]

		"""
		progress = float(self._dti_cursor) / float(max(self.m, 1))

		cdf_y = np.clip(self._last_cdf_y.astype(np.float32, copy=False), 0.0, 1.0)

		max_traffic = float(max(self.config["traffic_elements"]))
		if max_traffic <= 0:
			traffic_state = np.zeros(self.n, dtype=np.float32)
		else:
			traffic_state = np.asarray(self._current_traffic_dti, dtype=np.float32).reshape(-1)
			if traffic_state.size < self.n:
				traffic_state = np.pad(traffic_state, (0, self.n - traffic_state.size), mode="constant")
			elif traffic_state.size > self.n:
				traffic_state = traffic_state[: self.n]
			traffic_state = np.clip(traffic_state, 0.0, max_traffic) / max_traffic

		state = np.concatenate(
			[
				np.array(
					[
						progress,
						float(np.clip(self._last_beta_current, 0.0, 1.0)),
						float(np.clip(self._last_beta_cumulative, 0.0, 1.0)),
						float(np.clip(self._last_rb_norm, 0.0, 1.0)),
					],
					dtype=np.float32,
				),
				cdf_y,
				traffic_state,
			]
		)
		if not np.all(np.isfinite(state)):
			raise ValueError("state contains non-finite values")
		return state

    #for testing
	def build_state_from_traffic(
		self,
		traffic_dti: Union[list, tuple, np.ndarray],
		dti_index: int = 0,
	) -> np.ndarray:
		"""Build a traffic-conditioned inference state for a specific DTI.

		This method avoids relying on reset-only history by deriving beta/CDF
		features from one deterministic model probe using the provided traffic.
		"""
		traffic_vec = self._normalize_traffic_vector(traffic_dti)

		progress = float(np.clip(float(dti_index) / float(max(self.m, 1)), 0.0, 1.0))

		probe_rb = int(np.round(0.5 * (self.rb_min + self.rb_max)))
		rb_data = [probe_rb] * self.n

		self.model.set_service(self.service)
		self.model.reset(self.service)
		self.model.set_traffic(traffic_vec)
		self.model.set_resource_blocks(rb_data)
		dti_result = self.model.process_dti(
			traffic_data=traffic_vec,
			rb_data=rb_data,
			c_capacity=self.c,
			rb_used=probe_rb,
			lambda_reward=self.lambda_reward,
		)

		(beta_current, cdf_matrix), _ = self.model.to_rl_input(dti_result)
		beta_cumulative = float(dti_result.beta_result.beta_cumulative)
		cdf_y = np.asarray(cdf_matrix[:, 1], dtype=np.float32)
		cdf_y = np.clip(cdf_y, 0.0, 1.0)

		max_traffic = float(max(self.config["traffic_elements"]))
		if max_traffic <= 0:
			traffic_state = np.zeros(self.n, dtype=np.float32)
		else:
			traffic_state = np.asarray(traffic_vec, dtype=np.float32) / max_traffic
			traffic_state = np.clip(traffic_state, 0.0, 1.0)

		state = np.concatenate(
			[
				np.array(
					[
						progress,
						float(np.clip(beta_current, 0.0, 1.0)),
						float(np.clip(beta_cumulative, 0.0, 1.0)),
						float(np.clip(self._normalize_rb(probe_rb), 0.0, 1.0)),
					],
					dtype=np.float32,
				),
				cdf_y.astype(np.float32, copy=False),
				traffic_state.astype(np.float32, copy=False),
			]
		)
		if not np.all(np.isfinite(state)):
			raise ValueError("inference state contains non-finite values")
		return state

	def infer_rb_from_traffic(
		self,
		traffic_dti: Union[list, tuple, np.ndarray],
		agent: Any,
		dti_index: int = 0,
	) -> int:
		"""Infer deterministic RB directly from provided DTI traffic."""
		state = self.build_state_from_traffic(traffic_dti=traffic_dti, dti_index=dti_index)
		action = agent.select_action(state, evaluate=True)
		return int(self._action_to_rb(action))

	@property
	def observation_shape(self) -> Tuple[int]:
		"""Return observation shape for network input layer construction."""
		return (4 + self.k + self.n,)

	@property
	def action_shape(self) -> Tuple[int]:
		"""Return action shape expected by WCSAC actor."""
		return (1,)

	@property
	def action_bounds(self) -> Tuple[float, float]:
		"""Return external action bounds accepted by this environment."""
		return (self.action_low, self.action_high)

	def _normalize_rb(self, rb: int) -> float:
		"""Normalize integer RB allocation into [0, 1]."""
		span = float(max(self.rb_max - self.rb_min, 1))
		return float(np.clip((float(rb) - self.rb_min) / span, 0.0, 1.0))

	def _action_to_rb(self, action: Union[Number, np.ndarray, list, tuple]) -> int:
		"""
		Convert a WCSAC continuous action to a valid integer RB allocation.

		Mapping:
			1) extract scalar action value
			2) normalize into [-1, 1]
			3) linearly map into [rb_min, rb_max]
			4) round and clamp to integer bounds
		"""
		value = self._action_to_scalar(action)
		a_norm = self._normalize_action_value(value)
		ratio = 0.5 * (a_norm + 1.0)
		rb_float = self.rb_min + ratio * (self.rb_max - self.rb_min)
		rb_int = int(np.round(rb_float))
		rb_int = int(np.clip(rb_int, self.rb_min, self.rb_max))
		return rb_int

	def _normalize_action_value(self, action_scalar: float) -> float:
		"""Normalize action from [action_low, action_high] into [-1, 1] with clipping."""
		clipped = float(np.clip(action_scalar, self.action_low, self.action_high))
		ratio = (clipped - self.action_low) / (self.action_high - self.action_low)
		a_norm = (2.0 * ratio) - 1.0
		return float(np.clip(a_norm, -1.0, 1.0))

	@staticmethod
	def _action_to_scalar(action: Union[Number, np.ndarray, list, tuple]) -> float:
		"""Extract a finite scalar value from action input."""
		if isinstance(action, (int, float, np.floating, np.integer)):
			value = float(action)
		elif isinstance(action, (list, tuple, np.ndarray)):
			arr = np.asarray(action, dtype=np.float32).reshape(-1)
			if arr.size == 0:
				raise ValueError("action cannot be empty")
			value = float(arr[0])
		else:
			raise ValueError(f"Unsupported action type: {type(action).__name__}")

		if not np.isfinite(value):
			raise ValueError(f"action must be finite, got {value}")
		return value

	def _normalize_traffic_vector(self, traffic_dti: Union[list, tuple, np.ndarray]) -> list:
		"""Convert input traffic to integer list with length exactly n."""
		arr = np.asarray(traffic_dti, dtype=np.float32).reshape(-1)
		if arr.size == 0:
			raise ValueError("traffic_dti cannot be empty")

		if not np.all(np.isfinite(arr)):
			raise ValueError("traffic_dti must contain finite values")
		if np.any(arr < 0.0):
			raise ValueError("traffic_dti values must be >= 0")

		vals = [int(np.round(x)) for x in arr.tolist()]
		if len(vals) == self.n:
			return vals

		if len(vals) < self.n:
			repeated = (vals * ((self.n + len(vals) - 1) // len(vals)))[: self.n]
			return repeated

		return vals[: self.n]

	@classmethod
	def _normalize_service(cls, service: str) -> str:
		"""Normalize service alias to one of: voip, cbr, streaming."""
		if not isinstance(service, str):
			raise ValueError("service must be a string")
		key = service.strip().lower()
		if key not in cls.SERVICE_ALIASES:
			raise ValueError(
				"service must be one of: voip, cbr, streaming, videostream"
			)
		return cls.SERVICE_ALIASES[key]

	@staticmethod
	def _normalize_profile_mode(mode: str) -> str:
		"""Normalize traffic profile mode to one of: fixed, random."""
		if not isinstance(mode, str):
			raise ValueError("traffic_profile_mode must be a string")
		key = mode.strip().lower()
		if key not in {"fixed", "random"}:
			raise ValueError("traffic_profile_mode must be one of: fixed, random")
		return key
