from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MULTI_SLICE_ROOT = PROJECT_ROOT / "multi_slice"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def configure_logger(log_path: Path) -> logging.Logger:
	logger = logging.getLogger("multi_traffic_prediction")
	logger.setLevel(logging.INFO)
	logger.propagate = False

	for handler in list(logger.handlers):
		logger.removeHandler(handler)

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	log_path.parent.mkdir(parents=True, exist_ok=True)
	file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	return logger


def _prompt_choice(prompt: str, options: list[str]) -> str:
	while True:
		print(prompt)
		for idx, option in enumerate(options, start=1):
			print(f"  {idx}. {option}")
		choice = input(f"Select 1-{len(options)}: ").strip()
		if choice.isdigit():
			idx = int(choice)
			if 1 <= idx <= len(options):
				return options[idx - 1]
		print("Invalid selection. Please try again.")


def _prompt_profile(prompt_label: str) -> str:
	from WCSAC_RL_Model.traffic_profiles import get_default_profiles

	profiles = list(get_default_profiles().keys())
	return _prompt_choice(prompt_label, profiles)


def _prompt_int(prompt: str, minimum: int) -> int:
	while True:
		raw = input(prompt).strip()
		if raw.isdigit():
			value = int(raw)
			if value >= minimum:
				return value
		print(f"Invalid selection. Please enter an integer >= {minimum}.")


def _get_default_beta_threshold() -> float:
	try:
		from WCSAC_RL_Model.config import get_default_config

		cfg = get_default_config()
		beta_threshold_value = getattr(cfg.agent, "beta_threshold", None)
		return float(0.1 if beta_threshold_value is None else beta_threshold_value)
	except Exception:
		return 0.1


MAX_TRAFFIC_INPUTS = 8


def main() -> None:
	from multi_slice.multi_traffic_config import AVAILABLE_SERVICES, MIN_TRAFFIC_INPUTS, MultiTrafficPredictionConfig, TrafficInputSelection
	from multi_slice.multi_traffic_predictor import MultiTrafficPredictor

	while True:
		print("====================================")
		print("Enter your choice (1-2):")
		print("  1. WCSAC")
		print("  2. Exit")
		print("====================================")

		model_choice = input("Select 1-2: ").strip()
		if model_choice == "1":
			model_name = "wcsac"
			model_label = "WCSAC"
		elif model_choice == "2":
			print("Exiting.")
			return
		else:
			print("Invalid selection. Please try again.")
			continue

		num_inputs = _prompt_int(
			f"How many traffic inputs do you want to use? ({MIN_TRAFFIC_INPUTS}-{MAX_TRAFFIC_INPUTS}): ",
			minimum=MIN_TRAFFIC_INPUTS,
		)
		if num_inputs > MAX_TRAFFIC_INPUTS:
			print(f"Invalid selection. Maximum is {MAX_TRAFFIC_INPUTS} because capacity = 8.")
			continue

		inputs: list[TrafficInputSelection] = []
		for i in range(1, num_inputs + 1):
			service = _prompt_choice(f"Select service for input {i}", list(AVAILABLE_SERVICES))
			profile = _prompt_profile(f"Select profile for input {i}")
			inputs.append(TrafficInputSelection(service=service, profile_name=profile, label=f"input_{i}"))

		print("\nSelected inputs:")
		for selection in inputs:
			print(f"  {selection.label}: service={selection.service}, profile={selection.profile_name}")
		print()

		output_dir = MULTI_SLICE_ROOT / "multi_traffic"
		log_path = output_dir / f"{model_name}_multi_traffic_prediction.log"
		logger = configure_logger(log_path)
		beta_threshold = _get_default_beta_threshold()

		config = MultiTrafficPredictionConfig(
			inputs=inputs,
			model_name=model_name,
			capacity=8,
			beta_threshold=beta_threshold,
			seed=42,
			output_dir=output_dir,
		)

		try:
			print(f"Starting multi-traffic prediction mode for {model_label}")
			for selection in inputs:
				print(f"{selection.label}: service={selection.service}, profile={selection.profile_name}")
			print(f"Beta threshold: {beta_threshold}")

			runner = MultiTrafficPredictor(config=config, logger=logger)
			result = runner.run()
		except FileNotFoundError as exc:
			print(f"ERROR: {exc}")
			continue
		except Exception as exc:
			print(f"ERROR: Multi-traffic prediction failed: {exc}")
			continue

		print("Finished multi-traffic prediction")
		if result.steps:
			last_step = result.steps[-1]
			print("Last allocated RBs:")
			for input_log in last_step.inputs:
				print(f"  {input_log.input_label} = {input_log.allocated_rb}")
		print("\nMulti-traffic prediction completed.\n")


if __name__ == "__main__":
	main()