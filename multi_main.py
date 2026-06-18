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
	from SAC_RL_Model.traffic_profiles import get_default_profiles

	profiles = list(get_default_profiles().keys())
	return _prompt_choice(prompt_label, profiles)


def _get_default_beta_threshold() -> float:
	try:
		from WCSAC_RL_Model.config import get_default_config

		cfg = get_default_config()
		beta_threshold_value = getattr(cfg.agent, "beta_threshold", None)
		return float(0.1 if beta_threshold_value is None else beta_threshold_value)
	except Exception:
		return 0.1


def main() -> None:
	from multi_slice.multi_traffic_config import AVAILABLE_SERVICES, MultiTrafficPredictionConfig, TrafficInputSelection
	from multi_slice.multi_traffic_predictor import MultiTrafficPredictor

	while True:
		print("====================================")
		print("Select RL Model:")
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

		service_1 = _prompt_choice("Select service for input 1", list(AVAILABLE_SERVICES))
		profile_1 = _prompt_profile("Select profile for input 1")
		service_2 = _prompt_choice("Select service for input 2", list(AVAILABLE_SERVICES))
		profile_2 = _prompt_profile("Select profile for input 2")

		output_dir = MULTI_SLICE_ROOT / "multi_traffic"
		log_path = output_dir / f"{model_name}_multi_traffic_prediction.log"
		logger = configure_logger(log_path)
		beta_threshold = _get_default_beta_threshold()

		config = MultiTrafficPredictionConfig(
			input_1=TrafficInputSelection(service=service_1, profile_name=profile_1, label="input_1"),
			input_2=TrafficInputSelection(service=service_2, profile_name=profile_2, label="input_2"),
			model_name=model_name,
			capacity=8,
			beta_threshold=beta_threshold,
			seed=42,
			output_dir=output_dir,
		)

		try:
			print(f"Starting multi-traffic prediction mode for {model_label}")
			print(f"Input 1: service={service_1}, profile={profile_1}")
			print(f"Input 2: service={service_2}, profile={profile_2}")
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
			print(
				f"Last allocated RBs: input 1 = {last_step.allocated_rb_1}, input 2 = {last_step.allocated_rb_2}"
			)
		print("\nMulti-traffic prediction completed.\n")


if __name__ == "__main__":
	main()