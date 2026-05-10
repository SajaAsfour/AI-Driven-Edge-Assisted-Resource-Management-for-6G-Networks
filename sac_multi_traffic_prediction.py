from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MULTI_SLICE_ROOT = PROJECT_ROOT / "multi_slice_SAC"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from multi_slice_SAC.multi_traffic_config import AVAILABLE_SERVICES, MultiTrafficPredictionConfig, TrafficInputSelection
from multi_slice_SAC.multi_traffic_predictor import MultiTrafficPredictor
from SAC_RL_Model.traffic_profiles import get_default_profiles


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
	profiles = list(get_default_profiles().keys())
	return _prompt_choice(prompt_label, profiles)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Two-input multi-traffic SAC prediction and proportional RB allocation.")
	parser.add_argument("--service-1", choices=AVAILABLE_SERVICES, help="Traffic service for input 1")
	parser.add_argument("--profile-1", help="Traffic profile for input 1 (e.g. profile_1)")
	parser.add_argument("--service-2", choices=AVAILABLE_SERVICES, help="Traffic service for input 2")
	parser.add_argument("--profile-2", help="Traffic profile for input 2 (e.g. profile_6)")
	parser.add_argument("--capacity", type=int, default=8, help="Global RB capacity shared by both inputs")
	parser.add_argument("--seed", type=int, default=42, help="Seed used for traffic generation")
	parser.add_argument("--output-dir", default=str(MULTI_SLICE_ROOT / "multi_traffic"), help="Directory for logs and JSON output")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = Path(args.output_dir).expanduser()
	log_path = output_dir / "multi_traffic_prediction.log"
	logger = configure_logger(log_path)

	service_1 = args.service_1 or _prompt_choice("Select service for input 1", list(AVAILABLE_SERVICES))
	profile_1 = args.profile_1 or _prompt_profile("Select profile for input 1")
	service_2 = args.service_2 or _prompt_choice("Select service for input 2", list(AVAILABLE_SERVICES))
	profile_2 = args.profile_2 or _prompt_profile("Select profile for input 2")

	config = MultiTrafficPredictionConfig(
		input_1=TrafficInputSelection(service=service_1, profile_name=profile_1, label="input_1"),
		input_2=TrafficInputSelection(service=service_2, profile_name=profile_2, label="input_2"),
		capacity=int(args.capacity),
		seed=int(args.seed),
		output_dir=output_dir,
	)

	try:
		print("Starting multi-traffic prediction mode")
		print(f"Input 1: service={service_1}, profile={profile_1}")
		print(f"Input 2: service={service_2}, profile={profile_2}")

		runner = MultiTrafficPredictor(config=config, logger=logger)
		result = runner.run()
	except FileNotFoundError as exc:
		print(f"ERROR: {exc}")
		return
	except Exception as exc:
		print(f"ERROR: Multi-traffic prediction failed: {exc}")
		return

	print("Finished multi-traffic prediction")
	if result.steps:
		last_step = result.steps[-1]
		print(
			f"Last allocated RBs: input 1 = {last_step.allocated_rb_1}, input 2 = {last_step.allocated_rb_2}"
		)
	print("\nMulti-traffic prediction completed.")


if __name__ == "__main__":
	main()
