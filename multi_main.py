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


def main() -> None:
	from multi_slice.multi_traffic_config import MultiTrafficPredictionConfig
	from multi_slice.multi_traffic_predictor import MultiTrafficPredictor

	output_dir = MULTI_SLICE_ROOT / "multi_traffic"
	config = MultiTrafficPredictionConfig(output_dir=output_dir)
	model_label = (config.model_name or "wcsac").upper()
	log_path = output_dir / f"{config.model_name or 'wcsac'}_multi_traffic_prediction.log"
	logger = configure_logger(log_path)

	runner = MultiTrafficPredictor(config=config, logger=logger)
	print(f"Starting multi-traffic prediction mode for {model_label}")
	print(f"Beta threshold: {runner.config.beta_threshold}")
	result = runner.run()

	print("Finished multi-traffic prediction")
	print(f"Number of inputs: {result.config['num_inputs']}")
	print(f"Service names: {result.config['service_names']}")
	print(f"Number of DTIs: {result.config['num_dtis']}")
	print(f"Number of TTIs per DTI: {result.config['num_ttis_per_dti']}")
	print(f"Beta threshold: {result.config['beta_threshold']}")
	print(f"Capacity: {result.config['capacity']}")
	print(f"JSON output: {result.output_path}")
	print("Plot images:")
	for plot_path in result.plot_paths:
		print(f"  {plot_path}")
	if result.steps:
		last_step = result.steps[-1]
		print("Last allocated RBs:")
		for input_log in last_step.inputs:
			print(f"  {input_log.input_label} = {input_log.allocated_rb}")
	print("\nMulti-traffic prediction completed.\n")


if __name__ == "__main__":
	main()
