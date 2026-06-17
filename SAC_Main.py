from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = Path(__file__).parent.resolve()
NETWORK_MODEL_DIR = BASE_DIR / "Network_Model"
if str(NETWORK_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_MODEL_DIR))

from Network_Model.src.NetworkModel import NetworkModel
from SAC_RL_Model.traffic_profiles import get_default_profiles, get_profile_or_raise
Number = Union[int, float]


def _safe_plot_float(value: Any) -> float:
    """Return finite float value for plots, else NaN."""
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(value_float):
        return float("nan")
    return value_float


def _safe_dti_index_for_plot(value: Any, fallback: int) -> int:
    """Return a positive integer DTI index for plotting."""
    try:
        idx = int(value)
    except (TypeError, ValueError):
        idx = int(fallback)
    if idx <= 0:
        idx = int(fallback)
    return idx


def _rb_to_action_scalar(env: Any, rb_value: int) -> float:
    """Map integer RB to action scalar using environment action bounds."""
    span = float(max(int(env.rb_max) - int(env.rb_min), 1))
    ratio = (float(rb_value) - float(env.rb_min)) / span
    a_norm = (2.0 * ratio) - 1.0
    action_scalar = float(env.action_low) + ((a_norm + 1.0) * 0.5) * (
        float(env.action_high) - float(env.action_low)
    )
    return float(action_scalar)


def _log_q_value_inspection_for_state(
    evaluation_logger: logging.Logger,
    env: Any,
    agent: Any,
    state: Any,
    policy_action: np.ndarray,
) -> None:
    """Log Q1/Q2/Qmin for all RB candidates at the current evaluation state."""
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_arr.size <= 0:
        evaluation_logger.info("Q-value inspection skipped: empty state")
        return

    state_tensor = torch.as_tensor(
        state_arr,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    evaluation_logger.info("## Q-value inspection for current state")
    with torch.no_grad():
        for rb in range(int(env.rb_min), int(env.rb_max) + 1):
            action_scalar = _rb_to_action_scalar(env, rb)
            action_vec = np.zeros(int(agent.action_dim), dtype=np.float32)
            action_vec[0] = np.float32(action_scalar)
            action_tensor = torch.as_tensor(
                action_vec,
                dtype=torch.float32,
                device=agent.device,
            ).unsqueeze(0)

            q1 = float(agent.critic1(state_tensor, action_tensor).item())
            q2 = float(agent.critic2(state_tensor, action_tensor).item())
            q_min = float(min(q1, q2))
            evaluation_logger.info(
                f"RB = {rb} | Q1 = {q1:.6f} | Q2 = {q2:.6f} | Q_min = {q_min:.6f}"
            )

    selected_rb = int(env._action_to_rb(policy_action))
    evaluation_logger.info(f"Selected RB by policy: {selected_rb}")
    evaluation_logger.info("------------------------------------------")


def save_episode_dti_plots(episode_data: List[Dict[str, Any]], output_dir: Path) -> int:
    """Save per-episode Beta-vs-DTI and Reward-vs-DTI plots as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    for entry in episode_data:
        episode_idx = int(entry.get("episode", 0))
        dti_indices = [int(x) for x in entry.get("dti_indices", [])]
        beta_values = [float(x) for x in entry.get("beta_values", [])]
        reward_values = [float(x) for x in entry.get("reward_values", [])]

        if not dti_indices:
            continue

        min_len = min(len(dti_indices), len(beta_values), len(reward_values))
        if min_len <= 0:
            continue

        x_vals = dti_indices[:min_len]
        beta_vals = beta_values[:min_len]
        reward_vals = reward_values[:min_len]

        beta_file = output_dir / f"beta_vs_dti_episode_{episode_idx:03d}.png"
        reward_file = output_dir / f"reward_vs_dti_episode_{episode_idx:03d}.png"

        plt.figure(figsize=(8, 4.5))
        plt.plot(x_vals, beta_vals, marker="o", linewidth=1.5)
        plt.title(f"Beta vs DTI - Episode {episode_idx:03d}")
        plt.xlabel("DTI Index")
        plt.ylabel("beta_current")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(beta_file, dpi=150)
        plt.close()
        generated += 1

        plt.figure(figsize=(8, 4.5))
        plt.plot(x_vals, reward_vals, marker="o", linewidth=1.5)
        plt.title(f"Reward vs DTI - Episode {episode_idx:03d}")
        plt.xlabel("DTI Index")
        plt.ylabel("reward_current")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(reward_file, dpi=150)
        plt.close()
        generated += 1

    return generated


def save_prediction_beta_plot(
    dti_indices: List[int],
    beta_values: List[float],
    output_dir: Path,
    service: str,
    model_type: str
) -> Path:
    """
    Save Beta vs DTI plot for prediction results.
    
    Args:
        dti_indices: List of DTI indices
        beta_values: List of beta_current values
        output_dir: Directory to save the plot
        service: Service name (voip, cbr)
        model_type: Model type (wcsac, sac)
        
    Returns:
        Path to the saved plot file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure lengths match
    min_len = min(len(dti_indices), len(beta_values))
    if min_len <= 0:
        raise ValueError("No DTI indices or beta values provided")
    
    x_vals = dti_indices[:min_len]
    beta_vals = beta_values[:min_len]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x_vals, beta_vals, marker="o", linewidth=2, markersize=8, color="steelblue")
    
    # Title with service and model type
    title = f"Beta vs DTI - Prediction - {service.upper()} - {model_type.upper()}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    ax.set_xlabel("DTI Index", fontsize=12)
    ax.set_ylabel("beta_current", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add text label inside the plot showing service and model
    label_text = f"Service: {service.upper()}\nModel: {model_type.upper()}"
    ax.text(
        0.02, 0.98, label_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    
    fig.tight_layout()
    
    # Save the plot
    filename = f"prediction_beta_vs_dti_{service}_{model_type}.png"
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    
    return plot_path



def setup_logging(log_file: str = "debug.log"):
    """
    Configure logging to write to file only (no console output for results).
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)


# DATA LOADING FUNCTIONS

def require_int(name: str, value: Any) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be int, got {type(value).__name__}: {value!r}")
    return value


def require_number(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}: {value!r}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def require_list(name: str, value: Any) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}")
    return value


def require_dict(name: str, value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict/object, got {type(value).__name__}")
    return value


def validate_thresholds(service: str, got: Dict[str, Any], required_metrics: List[str]) -> Dict[str, Number]:
    missing = [m for m in required_metrics if m not in got]
    extra = [k for k in got.keys() if k not in required_metrics]
    
    if missing:
        raise ValueError(f"q_thresholds.{service} missing metrics: {missing}")
    if extra:
        raise ValueError(f"q_thresholds.{service} has unexpected metrics: {extra}")

    out: Dict[str, Number] = {}
    for metric in required_metrics:
        v = got[metric]
        if not isinstance(v, (int, float)):
            raise ValueError(
                f"q_thresholds.{service}.{metric} must be a number, got {type(v).__name__}"
            )
        out[metric] = v
    return out


def validate_users_matrix(service: str, mat: Any, m: int, n: int) -> List[List[int]]:
    if not isinstance(mat, list):
        raise ValueError(f"traffic_users_per_tti.{service} must be a 2D list")
    if len(mat) != m:
        raise ValueError(f"traffic_users_per_tti.{service} must have {m} rows, got {len(mat)}")

    out: List[List[int]] = []
    for row_i, row in enumerate(mat):
        if not isinstance(row, list):
            raise ValueError(f"traffic_users_per_tti.{service}[{row_i}] must be a list")
        if len(row) != n:
            raise ValueError(f"traffic_users_per_tti.{service}[{row_i}] must have {n} columns")

        row_out: List[int] = []
        for col_i, x in enumerate(row):
            if not isinstance(x, int):
                raise ValueError(f"traffic_users_per_tti.{service}[{row_i}][{col_i}] must be int")
            if x < 0:
                raise ValueError(f"traffic_users_per_tti.{service}[{row_i}][{col_i}] must be >= 0")
            row_out.append(x)
        out.append(row_out)
    return out


def validate_resource_blocks(mat: Any, m: int) -> List[List[int]]:
    if not isinstance(mat, list):
        raise ValueError("resource_blocks_per_dti must be a 2D list")

    if len(mat) != m:
        raise ValueError(f"resource_blocks_per_dti must have {m} rows, got {len(mat)}")

    out: List[List[int]] = []
    for dti, row in enumerate(mat):
        if not isinstance(row, list):
            raise ValueError(f"resource_blocks_per_dti[{dti}] must be a list")
        if len(row) < 2:
            raise ValueError(f"resource_blocks_per_dti[{dti}] must have at least 2 values")

        rb_row: List[int] = []
        for svc_name, val in zip(["VoIP", "CBR"], row):
            if not isinstance(val, int):
                raise ValueError(f"resource_blocks_per_dti[{dti}] {svc_name} must be int")
            if val < 0:
                raise ValueError(f"resource_blocks_per_dti[{dti}] {svc_name} must be >= 0")
            rb_row.append(val)

        out.append(rb_row)

    return out


def compute_rb_per_tti_from_dti(resource_blocks_per_dti: List[List[int]], 
                                 m: int, n: int) -> Dict[str, List[List[int]]]:
    voip_rb_tti = []
    cbr_rb_tti = []
    
    for dti_idx in range(m):
        voip_rb = resource_blocks_per_dti[dti_idx][0]
        cbr_rb = resource_blocks_per_dti[dti_idx][1]
        
        voip_rb_tti.append([voip_rb] * n)
        cbr_rb_tti.append([cbr_rb] * n)
    
    return {
        'voip': voip_rb_tti,
        'cbr': cbr_rb_tti,
    }


def load_configuration(config_path: Path, input_path: Path) -> Dict[str, Any]:
    VOIP_METRICS = [
        "voIPFrameLoss",
        "voIPReceivedThroughput",
        "voIPPlayoutLoss",
    ]
    CBR_METRICS = [
        "cbrReceivedThroughput",
    ]
    STREAMING_METRICS = [
        "rtVideoStreamingSegmentLoss",
    ]
    
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    input_data = json.loads(input_path.read_text(encoding="utf-8"))
    
    n = require_int("n", config_data.get("n"))
    m = require_int("m", config_data.get("m"))
    k = require_int("k", config_data.get("k"))
    c = require_int("c", config_data.get("c"))
    lambda_reward = require_number("lambda_reward", config_data.get("lambda_reward"))
    
    if n <= 0 or m <= 0 or k <= 0:
        raise ValueError("n, m, k must be positive integers")
    if c < 0:
        raise ValueError("c must be >= 0")

    traffic_elements = require_list("traffic_elements", config_data.get("traffic_elements"))
    if len(traffic_elements) != k:
        raise ValueError(f"traffic_elements length must equal k={k}")

    q_thresholds = require_dict("q_thresholds", config_data.get("q_thresholds"))
    voip_raw = require_dict("q_thresholds.voip", q_thresholds.get("voip"))
    cbr_raw = require_dict("q_thresholds.cbr", q_thresholds.get("cbr"))
    streaming_raw = require_dict("q_thresholds.streaming", q_thresholds.get("streaming"))

    q_voip = validate_thresholds("voip", voip_raw, VOIP_METRICS)
    q_cbr = validate_thresholds("cbr", cbr_raw, CBR_METRICS)
    q_streaming = validate_thresholds("streaming", streaming_raw, STREAMING_METRICS)
    
    resource_blocks_per_dti = validate_resource_blocks(
        input_data.get("resource_blocks_per_dti"), m
    )

    rb_per_tti = compute_rb_per_tti_from_dti(resource_blocks_per_dti, m, n)

    users = require_dict("traffic_users_per_tti", input_data.get("traffic_users_per_tti"))
    voip_users = validate_users_matrix("voip", users.get("voip"), m=m, n=n)
    cbr_users = validate_users_matrix("cbr", users.get("cbr"), m=m, n=n)

    return {
        'n': n, 'm': m, 'k': k, 'c': c, 'lambda_reward': lambda_reward,
        'traffic_elements': traffic_elements,
        'q_thresholds_voip': q_voip,
        'q_thresholds_cbr': q_cbr,
        'q_thresholds_streaming': q_streaming,
        'resource_blocks_per_dti': resource_blocks_per_dti,
        'resource_blocks_per_tti': rb_per_tti,
        'traffic_users_per_tti': {
            'voip': voip_users,
            'cbr': cbr_users,
        }
    }


def load_metric_matrices(json_file: Path) -> Dict[str, Any]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    meta = data['meta']
    
    ue_start = meta['ue_range']['start']
    ue_end = meta['ue_range']['end']
    ue_step = meta['ue_range']['step']
    ue_values = list(range(ue_start, ue_end + 1, ue_step))
    
    rb_start = meta['rb_range']['start']
    rb_end = meta['rb_range']['end']
    rb_step = meta['rb_range']['step']
    rbs_values = list(range(rb_start, rb_end + 1, rb_step))
    
    mean_matrices = {}
    std_matrices = {}
    
    for metric_name in meta['metrics']:
        mean_matrices[metric_name] = {}
        std_matrices[metric_name] = {}
        
        for ue in ue_values:
            mean_matrices[metric_name][str(ue)] = []
            std_matrices[metric_name][str(ue)] = []
            
            for rb in rbs_values:
                rb_key = f'rb_{rb}'
                ue_key = f'ue_{ue}'
                
                try:
                    metric_data = data['data'][rb_key][ue_key][metric_name]
                    mean_matrices[metric_name][str(ue)].append(metric_data['mean'])
                    std_matrices[metric_name][str(ue)].append(metric_data['std'])
                except KeyError:
                    mean_matrices[metric_name][str(ue)].append(0.0)
                    std_matrices[metric_name][str(ue)].append(0.0)
    
    return {
        'metadata': {
            'ue_values': ue_values,
            'rbs_values': rbs_values,
            'metrics': meta['metrics'],
            'units': meta['units']
        },
        'mean_matrices': mean_matrices,
        'std_matrices': std_matrices
    }


# MENU DISPLAY

def display_menu():
    print("\n" + "="*50)
    print("NETWORK MODEL :)")
    print("="*50)
    print("Please select a service:")
    print("  1. VoIP")
    print("  2. CBR")
    print("  3. Exit")
    print("="*50)


def get_user_choice() -> str:
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice == '1':
            return 'voip'
        elif choice == '2':
            return 'cbr'
        elif choice == '3':
            return 'exit'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


# DTI LOGGING

def log_dti_result(logger: logging.Logger, service: str, dti_index: int, 
                   cdf_result, beta_result, reward_current, traffic_dti, rb_dti,
                   c_capacity: Number, rb_used_current: Number, lambda_reward: Number):
    """Log a single DTI result to file in the required format."""
    logger.info(f"Service: {service}")
    logger.info(f"DTI Index: {dti_index}")
    logger.info(f"Traffic: {traffic_dti}")
    logger.info(f"RB: {rb_dti}")
    
    logger.info("CDF Values:")
    for traffic_val, cdf_prob in zip(cdf_result.cdf_x, cdf_result.cdf_y):
        logger.info(f"  Traffic {int(traffic_val)}: {float(cdf_prob):.4f}")

    cdf_matrix = np.column_stack((cdf_result.cdf_x, cdf_result.cdf_y))
    state = (float(beta_result.beta_current), cdf_matrix)
    result = (state, float(reward_current))

    logger.info("")
    logger.info("RL STRUCTURED OUTPUT (Current DTI):")
    logger.info("  state = (beta_current, CDF_matrix)")
    logger.info(
        f"  beta_current = dti_total_failures / dti_total_traffic = "
        f"{beta_result.dti_total_failures}/{beta_result.dti_total_traffic} = {beta_result.beta_current:.6f}"
    )
    logger.info("  CDF_matrix = [traffic_level, cdf_probability]")
    for row in cdf_matrix:
        logger.info(f"    [{int(row[0])}, {float(row[1]):.6f}]")

    resource_term = float(lambda_reward) * ((float(c_capacity) - float(rb_used_current)) / float(c_capacity))
    logger.info(
        f"  reward_current = -beta_current + lambda_reward * ((C - rb_used_current) / C)"
    )
    logger.info(
        f"  reward_current = -{beta_result.beta_current:.6f} + {float(lambda_reward):.6f} * "
        f"(({float(c_capacity):.6f} - {float(rb_used_current):.6f}) / {float(c_capacity):.6f})"
    )
    logger.info(f"  resource_term = {resource_term:.6f}")
    logger.info(f"  result = (state, reward_current)")
    logger.info(f"  result.reward_current = {float(result[1]):.6f}")

    logger.info(f"Beta Value: {beta_result.beta_cumulative:.4f}")
    logger.info(f"Reward (Current DTI): {reward_current:.4f}")
    logger.info("-" * 24)
    logger.info("")


def display_main_execution_menu() -> None:
    print("\n" + "=" * 60)
    print("MAIN MENU")
    print("=" * 60)
    print("  1. Run netowrk model simulation")
    print("  2. Train SAC")
    print("  3. Evaluate trained SAC")
    print("  4. Predict RBs from custom traffic input")
    print("  5. Exit")
    print("=" * 60)


def get_main_execution_choice() -> str:
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nInput cancelled. Exiting...")
            return "5"
        if choice in {"1", "2", "3", "4", "5"}:
            return choice
        print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


def choose_service_for_rl() -> Optional[str]:
    print("\nSelect RL service:")
    print("  1. VoIP")
    print("  2. CBR")
    print("  3. Back to Main Menu")
    while True:
        choice = input("Enter service choice (1-3): ").strip()
        if choice == "1":
            return "voip"
        if choice == "2":
            return "cbr"
        if choice == "3":
            return None
        print("Invalid choice, please try again.")


def choose_service_for_evaluation() -> Optional[str]:
    print("\nSelect RL service:")
    print("  1. VoIP")
    print("  2. CBR")
    print("  3. Back to Main Menu")
    while True:
        choice = input("Enter service choice (1-3): ").strip()
        if choice == "1":
            return "voip"
        if choice == "2":
            return "cbr"
        if choice == "3":
            return None
        print("Invalid choice. Please enter 1, 2, or 3.")


def choose_traffic_profile_mode_for_rl() -> str:
    print("\nTraffic profile mode:")
    print("  1. Fixed profile")
    print("  2. Random profile")
    while True:
        choice = input("Enter profile mode choice (1-2): ").strip()
        if choice == "1":
            return "fixed"
        if choice == "2":
            return "random"
        print("Invalid choice. Please enter 1 or 2.")


def choose_fixed_profile_name() -> str:
    profiles = get_default_profiles()
    print("\nAvailable fixed UE profiles:")
    ordered = list(profiles.items())
    for idx, (name, values) in enumerate(ordered, start=1):
        print(f"  {idx}. {name} -> {values}")

    while True:
        raw = input(f"Select profile (1-{len(ordered)}): ").strip()
        if not raw.isdigit():
            print("Invalid selection. Please enter a number.")
            continue
        idx = int(raw)
        if idx < 1 or idx > len(ordered):
            print(f"Invalid selection. Please enter a number from 1 to {len(ordered)}.")
            continue
        profile_name = ordered[idx - 1][0]
        _, values = get_profile_or_raise(profiles, profile_name)
        print(f"Selected fixed profile: {profile_name} -> {values}")
        return profile_name


def choose_profile_settings_for_rl() -> Tuple[str, Optional[str]]:
    mode = choose_traffic_profile_mode_for_rl()
    if mode == "fixed":
        fixed_profile = choose_fixed_profile_name()
        return mode, fixed_profile
    print("Random profile mode selected")
    return mode, None


def resolve_checkpoint_dir(base_dir: Path, service: str, profile_mode: str) -> Path:
    return base_dir / service / profile_mode


def resolve_random_final_checkpoint(base_checkpoint_dir: Path, service: str) -> Path:
    ckpt_path = resolve_checkpoint_dir(base_checkpoint_dir, service, "random") / f"sac_{service}_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Random final checkpoint not found for service '{service}': {ckpt_path}"
        )
    return ckpt_path


def choose_evaluation_mode(train_mode: str) -> str:
    print("\nEvaluation mode:")
    print("  1. Use original training profile settings")
    if train_mode == "random":
        print("  2. Override to fixed profile evaluation")
    else:
        print("  2. Override to random profile evaluation")
    while True:
        choice = input("Enter evaluation mode (1-2): ").strip()
        if choice == "1":
            return "original"
        if choice == "2":
            return "override"
        print("Invalid choice. Please enter 1 or 2.")


def choose_rb_allocation_mode_for_evaluation() -> bool:
    print("\nSelect RB allocation mode:")
    print("  1. Use AI policy (default)")
    print("  2. Use hard-coded RB value (debug mode)")
    while True:
        choice = input("Enter RB allocation mode (1-2): ").strip()
        if choice == "1":
            return False
        if choice == "2":
            return True
        print("Invalid choice. Please enter 1 or 2.")


def choose_hardcoded_rb_value(env: Any) -> int:
    raw_value = input("Enter fixed RB value to use for all DTIs: ").strip()
    try:
        rb_value = int(raw_value)
    except ValueError as e:
        raise ValueError(f"Invalid RB value '{raw_value}': must be an integer") from e

    rb_min = getattr(env, "rb_min", None)
    rb_max = getattr(env, "rb_max", None)
    if rb_min is None or rb_max is None:
        raise ValueError("Environment RB bounds are not configured (rb_min/rb_max)")
    if rb_value < rb_min or rb_value > rb_max:
        raise ValueError(f"Invalid RB value {rb_value}: must be within [{rb_min}, {rb_max}]")

    return rb_value


def _checkpoint_sort_key(path: Path) -> Tuple[int, int, str]:
    name = path.name
    match = re.search(r"_episode_(\d+)\.pt$", name)
    if match:
        return (0, int(match.group(1)), name.lower())
    if name.endswith("_final.pt"):
        return (2, 0, name.lower())
    return (1, 0, name.lower())


def list_checkpoints_for_service_mode(
    base_checkpoint_dir: Path,
    service: str,
    profile_mode: str,
) -> List[Path]:
    checkpoint_dir = resolve_checkpoint_dir(base_checkpoint_dir, service, profile_mode)
    return sorted(checkpoint_dir.glob("*.pt"), key=_checkpoint_sort_key)


def choose_checkpoint_for_evaluation(checkpoint_dir: Path) -> Optional[Path]:
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=_checkpoint_sort_key)
    if not candidates:
        return None

    print("\nSelect checkpoint to evaluate:")
    for idx, path in enumerate(candidates, start=1):
        print(f"  {idx}. {path.name}")

    while True:
        raw = input(f"Enter checkpoint choice (1-{len(candidates)}): ").strip()
        if not raw.isdigit():
            print("Invalid choice. Please enter a number.")
            continue
        idx = int(raw)
        if idx < 1 or idx > len(candidates):
            print(f"Invalid choice. Please enter a number from 1 to {len(candidates)}.")
            continue
        return candidates[idx - 1]


def _infer_service_from_checkpoint_name(checkpoint_path: Path) -> Optional[str]:
    match = re.match(r"^sac_(voip|cbr)_(?:episode_\d+|final)\.pt$", checkpoint_path.name)
    if not match:
        return None
    return str(match.group(1)).strip().lower()


def run_networkmodel(config: Dict[str, Any], config_dir: Path, logger: logging.Logger) -> None:
    model = NetworkModel(
        n=config['n'],
        m=config['m'],
        k=config['k'],
        traffic_elements=config['traffic_elements'],
        q_thresholds_voip=config['q_thresholds_voip'],
        q_thresholds_cbr=config['q_thresholds_cbr'],
        q_thresholds_streaming=config['q_thresholds_streaming']
    )

    service_files = {
        'voip': config_dir / 'D2min_VoIP_summary.json',
        'cbr': config_dir / 'D30sec_CBR_summary.json'
    }

    service_names = {
        'voip': 'VoIP',
        'cbr': 'CBR'
    }

    service_rb_index = {
        'voip': 0,
        'cbr': 1,
    }

    while True:
        display_menu()
        service = get_user_choice()

        if service == 'exit':
            break

        json_file = service_files[service]
        try:
            metric_data = load_metric_matrices(json_file)
            model.set_metric_matrices(service, metric_data)
        except FileNotFoundError:
            print(f"ERROR: File not found: {json_file.name}")
            continue
        except Exception as e:
            print(f"ERROR: Failed to load metrics: {e}")
            continue

        model.set_service(service)
        model.reset(service)

        m = config['m']
        traffic_data_all_dtis = config['traffic_users_per_tti'][service]
        rb_data_all_dtis = config['resource_blocks_per_tti'][service]

        for dti in range(m):
            print(f"Processing DTI {dti + 1}/{m}...")

            traffic_dti = traffic_data_all_dtis[dti]
            rb_dti = rb_data_all_dtis[dti]

            total_rb_dti = sum(config['resource_blocks_per_dti'][dti])
            if total_rb_dti > config['c']:
                print(
                    f"\n[ERROR] DTI {dti + 1}: total RB ({total_rb_dti}) must be <= c ({config['c']})"
                )
                print("Skipping to next DTI...\n")
                continue

            model.set_traffic(traffic_dti)
            model.set_resource_blocks(rb_dti)

            rb_used_current = config['resource_blocks_per_dti'][dti][service_rb_index[service]]

            try:
                dti_result = model.process_dti(
                    traffic_dti,
                    rb_dti,
                    c_capacity=config['c'],
                    rb_used=rb_used_current,
                    lambda_reward=config['lambda_reward'],
                )
            except ValueError as e:
                print(f"\n[ERROR] DTI {dti + 1}: {e}")
                print(f"Skipping to next DTI...\n")
                continue

            state, reward_current = model.to_rl_input(dti_result)
            beta_current, cdf_matrix = state

            cdf = dti_result.cdf_result
            beta = dti_result.beta_result

            print("\n" + "=" * 60)
            print(f"SERVICE: {service_names[service].upper()} | DTI {dti + 1}/{m}")
            print("=" * 60)
            print(f"Traffic in this DTI (per TTI): {traffic_dti}")
            print(f"Resource Blocks (per TTI): {rb_dti}")
            print("\nCDF VALUES:")
            print("  Traffic Level    |    CDF Probability")
            print("  ----------------|-------------------")
            for traffic_val, prob in zip(cdf.cdf_x, cdf.cdf_y):
                print(f"  {int(traffic_val):>13}    |    {prob:.4f}")

            print(f"\nBETA RESULTS:")
            print(f"  Beta (Current DTI only): {beta.beta_current:.4f}")
            print(f"  Beta (Cumulative):       {beta.beta_cumulative:.4f}")
            print(f"  Failures (Current DTI):  {beta.dti_total_failures}")
            print(f"  Total Users (Current DTI): {beta.dti_total_traffic}")
            print(f"  Cumulative Failures:     {beta.cumulative_failures}")
            print(f"  Cumulative Users:        {beta.cumulative_traffic}")
            resource_term = config['lambda_reward'] * ((config['c'] - rb_used_current) / config['c'])
            print("\nREWARD COMPUTATION (Current DTI):")
            print("  reward_current = -beta_current + lambda_reward * ((C - rb_used) / C)")
            print(
                f"  reward_current = -{beta.beta_current:.4f} + "
                f"{config['lambda_reward']:.4f} * (({config['c']} - {rb_used_current}) / {config['c']})"
            )
            print(f"  resource_term = {resource_term:.4f}")
            print(f"  Reward (Current DTI):    {dti_result.reward_current:.4f}")
            print("\nRL INPUT FORMAT (Current DTI):")
            print("  state = (beta_current, CDF_matrix)")
            print(f"  beta_current = {beta_current:.4f}")
            print(f"  CDF_matrix shape = {cdf_matrix.shape}")
            print(f"  result = (state, reward_current) -> reward_current = {reward_current:.4f}")
            print("=" * 60 + "\n")

            log_dti_result(
                logger,
                service,
                dti,
                dti_result.cdf_result,
                dti_result.beta_result,
                dti_result.reward_current,
                traffic_dti,
                rb_dti,
                c_capacity=config['c'],
                rb_used_current=rb_used_current,
                lambda_reward=config['lambda_reward']
            )

            if dti < m - 1:
                input("Press ENTER to continue to next DTI...")


def run_sac_training_mode() -> None:

    try:
        from SAC_RL_Model.trainer import train_sac
        from SAC_RL_Model.config import get_default_config
    except Exception as e:
        print(f"ERROR: RL training modules could not be imported: {e}")
        return

    
    service = choose_service_for_rl()
    if service is None:
        print("Returning to main menu...")
        return
    profile_mode, fixed_profile_name = choose_profile_settings_for_rl()

    cfg = get_default_config()
    base_checkpoint_dir_cfg = Path(cfg.checkpoint.checkpoint_dir)
    base_checkpoint_dir = (
        base_checkpoint_dir_cfg
        if base_checkpoint_dir_cfg.is_absolute()
        else BASE_DIR / base_checkpoint_dir_cfg
    )
    checkpoint_dir = resolve_checkpoint_dir(base_checkpoint_dir, service, profile_mode)

    cfg.environment.service = service
    cfg.environment.traffic_profile_mode = profile_mode
    cfg.environment.fixed_profile_name = fixed_profile_name or cfg.environment.fixed_profile_name
    cfg.checkpoint.checkpoint_dir = checkpoint_dir
    cfg.training.verbose = True

    print("\nStarting SAC training (config-driven)...")
    print(f"Checkpoint output directory: {checkpoint_dir}")

    try:
        train_result = train_sac(config=cfg)
    except Exception as e:
        print(f"ERROR: SAC training failed: {e}")
        return

    checkpoints = train_result.get("saved_checkpoints", [])
    print("\nSAC training completed successfully.")
    if checkpoints:
        print(f"Latest checkpoint: {checkpoints[-1]}")


def run_sac_evaluation_mode() -> None:
    try:
        from SAC_RL_Model.trainer import load_config
        from SAC_RL_Model.config import get_default_config
        from SAC_RL_Model.env_wrapper import NetworkSACEnv
        from SAC_RL_Model.agent import SACAgent
    except Exception as e:
        print(f"ERROR: RL evaluation modules could not be imported: {e}")
        return

    cfg = get_default_config()

    checkpoint_dir_cfg = Path(cfg.checkpoint.checkpoint_dir)
    base_checkpoint_dir = checkpoint_dir_cfg if checkpoint_dir_cfg.is_absolute() else BASE_DIR / checkpoint_dir_cfg

    while True:
        selected_service_for_folder = choose_service_for_evaluation()
        if selected_service_for_folder is None:
            print("\nLeaving SAC evaluation mode. Returning to main menu...")
            break

        selected_mode_for_folder = choose_traffic_profile_mode_for_rl()
        checkpoint_dir = resolve_checkpoint_dir(base_checkpoint_dir, selected_service_for_folder, selected_mode_for_folder)

        available_checkpoints = list_checkpoints_for_service_mode(
            base_checkpoint_dir=base_checkpoint_dir,
            service=selected_service_for_folder,
            profile_mode=selected_mode_for_folder,
        )

        ckpt_path = choose_checkpoint_for_evaluation(checkpoint_dir)
        if ckpt_path is None:
            print(
                "ERROR: "
                f"No checkpoint files found for service={selected_service_for_folder} "
                f"and mode={selected_mode_for_folder} in {checkpoint_dir}"
            )
            print("Returning to evaluation menu...")
            continue
        if ckpt_path not in available_checkpoints:
            print(f"ERROR: Selected checkpoint is not valid for folder: {checkpoint_dir}")
            print("Returning to evaluation menu...")
            continue

        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint file not found: {ckpt_path}")
            print("Returning to evaluation menu...")
            continue

        config_path = ckpt_path.with_name(f"{ckpt_path.stem}_config.json")
        loaded_config: Dict[str, Any] = {}
        if config_path.exists():
            try:
                loaded_config = load_config(config_path)
                print(f"Loaded config from checkpoint: {config_path}")
            except Exception as e:
                print(f"WARNING: Failed to load checkpoint config, using fallbacks: {e}")
                loaded_config = {}

        env_cfg_raw = loaded_config.get("environment", {}) if isinstance(loaded_config, dict) else {}
        env_cfg = env_cfg_raw if isinstance(env_cfg_raw, dict) else {}
        eval_cfg_raw = loaded_config.get("evaluation", {}) if isinstance(loaded_config, dict) else {}
        eval_cfg = eval_cfg_raw if isinstance(eval_cfg_raw, dict) else {}

        inferred_service = _infer_service_from_checkpoint_name(ckpt_path) or selected_service_for_folder

        train_service = str(env_cfg.get("service", inferred_service)).strip().lower()
        train_mode = str(env_cfg.get("traffic_profile_mode", selected_mode_for_folder)).strip().lower()
        if train_mode not in {"fixed", "random"}:
            train_mode = selected_mode_for_folder
        train_profile = str(env_cfg.get("fixed_profile_name", "profile_1"))
        train_seed = env_cfg.get("seed", cfg.environment.seed)
        train_rb_min = env_cfg.get("rb_min", cfg.environment.rb_min)

        eval_mode = choose_evaluation_mode(train_mode)

        if eval_mode == "original":
            selected_service = train_service
            selected_mode = train_mode
            selected_profile = train_profile
            print("\nEvaluating using training profile settings:")
            print(f"service = {selected_service}")
            if selected_mode == "random":
                print("mode = random (dynamic per step)")
            else:
                print("mode = fixed")
                print(f"profile = {selected_profile}")
        else:
            selected_service = train_service
            if train_mode == "random":
                selected_mode = "fixed"
                selected_profile = choose_fixed_profile_name()
            else:
                selected_mode = "random"
                selected_profile = None
            print("\nEvaluating with overridden profile settings (service fixed from checkpoint):")
            print(f"service = {selected_service}")
            if selected_mode == "random":
                print("mode = random (dynamic per step)")
            else:
                print("mode = fixed")
                print(f"profile = {selected_profile}")

        eval_episodes = int(cfg.evaluation.episodes)
        eval_steps = int(cfg.evaluation.max_steps_per_episode)
        debug_print_q_values = bool(getattr(cfg.evaluation, "debug_print_q_values", False))
        if "debug_print_q_values" in eval_cfg:
            debug_print_q_values = bool(eval_cfg.get("debug_print_q_values"))
        if eval_episodes <= 0:
            eval_episodes = 1
        if eval_steps <= 0:
            eval_steps = 1

        evaluation_logger = logging.getLogger("sac_evaluation_standalone")
        evaluation_logger.setLevel(logging.INFO)
        evaluation_logger.propagate = False
        for handler in list(evaluation_logger.handlers):
            handler.flush()
            handler.close()
            evaluation_logger.removeHandler(handler)

        try:
            checkpoint_dir_path = ckpt_path.parent

            env_kwargs = {
                "service": selected_service,
                "traffic_profile_mode": selected_mode,
                "seed": train_seed,
            }
            if selected_mode == "fixed":
                env_kwargs["fixed_profile_name"] = selected_profile
            if train_rb_min is not None:
                env_kwargs["rb_min"] = train_rb_min

            env = NetworkSACEnv(**env_kwargs)
            if hasattr(env, "set_logger"):
                env.set_logger(evaluation_logger)
            if hasattr(env, "set_logging_context"):
                env.set_logging_context("evaluation")

            use_hardcoded_rb = choose_rb_allocation_mode_for_evaluation()
            hardcoded_rb_value: Optional[int] = None
            log_path_for_run: Path
            if use_hardcoded_rb:
                hardcoded_rb_value = choose_hardcoded_rb_value(env)
                log_path_for_run = (
                    checkpoint_dir_path
                    / f"evaluation_hardcoded_rb_{int(hardcoded_rb_value)}.log"
                )
            else:
                log_path_for_run = checkpoint_dir_path / "evaluation.log"

            eval_file_handler = logging.FileHandler(
                log_path_for_run,
                mode='w',
                encoding='utf-8',
            )
            eval_file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            evaluation_logger.addHandler(eval_file_handler)

            if use_hardcoded_rb:
                evaluation_logger.info(
                    f"[DEBUG MODE] Writing hard-coded RB evaluation output to: {log_path_for_run}"
                )
                print(
                    f"Hard-coded RB output log: {log_path_for_run}"
                )

            state_dim = int(np.prod(env.observation_shape))
            action_dim = int(np.prod(env.action_shape))

            agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
            agent.load(ckpt_path)

            rewards: List[float] = []
            evaluation_episode_dti_series: List[Dict[str, Any]] = []
            evaluation_logger.info("Running deterministic SAC evaluation...")
            print("\nRunning deterministic SAC evaluation...")
            for episode in range(1, eval_episodes + 1):
                evaluation_logger.info("=" * 50)
                evaluation_logger.info(f"START EVALUATION EPISODE {episode}")
                evaluation_logger.info("=" * 50)
                if hasattr(env, "set_logger"):
                    env.set_logger(evaluation_logger)
                if hasattr(env, "set_logging_context"):
                    env.set_logging_context("evaluation")
                state = env.reset()
                ep_reward = 0.0
                ep_dti_indices: List[int] = []
                ep_beta_values: List[float] = []
                ep_reward_values: List[float] = []
                for step_idx in range(eval_steps):
                    policy_action_for_debug: Optional[np.ndarray] = None
                    if use_hardcoded_rb:
                        if debug_print_q_values:
                            policy_action_for_debug = agent.select_action(state, evaluate=True)
                        rb_value = int(hardcoded_rb_value)

                        # Convert RB value to normalized action in [-1, 1]
                        action = np.array([
                            2.0 * (rb_value - env.rb_min)
                            / (env.rb_max - env.rb_min)
                            - 1.0
                        ], dtype=np.float32)

                        evaluation_logger.info(
                            f"[DEBUG MODE] Using hard-coded RB allocation: {rb_value}"
                        )
                    else:
                        action = agent.select_action(state, evaluate=True)
                        policy_action_for_debug = action

                    if debug_print_q_values:
                        _log_q_value_inspection_for_state(
                            evaluation_logger=evaluation_logger,
                            env=env,
                            agent=agent,
                            state=state,
                            policy_action=policy_action_for_debug,
                        )

                    next_state, reward, done, info = env.step(action)
                    ep_reward += float(reward)
                    ep_dti_indices.append(_safe_dti_index_for_plot(info.get("dti_index"), fallback=step_idx + 1))
                    ep_beta_values.append(_safe_plot_float(info.get("beta_current")))
                    ep_reward_values.append(_safe_plot_float(reward))
                    state = next_state
                    if done:
                        break
                evaluation_episode_dti_series.append(
                    {
                        "episode": int(episode),
                        "dti_indices": ep_dti_indices,
                        "beta_values": ep_beta_values,
                        "reward_values": ep_reward_values,
                    }
                )
                rewards.append(ep_reward)
                ep_reward_log_value = float(ep_reward) if np.isfinite(ep_reward) else None
                if ep_reward_log_value is None:
                    evaluation_logger.info(f"Episode {episode:03d} reward: None")
                else:
                    evaluation_logger.info(f"Episode {episode:03d} reward: {ep_reward_log_value:.4f}")
                print(f"Episode {episode:03d} reward: {ep_reward:.4f}")

            finite_rewards = [float(r) for r in rewards if np.isfinite(r)]
            mean_reward = float(np.mean(finite_rewards)) if finite_rewards else 0.0
            std_reward = float(np.std(finite_rewards)) if finite_rewards else 0.0
            if finite_rewards:
                evaluation_logger.info(f"Evaluation complete | mean reward: {mean_reward:.4f} | std: {std_reward:.4f}")
            else:
                evaluation_logger.info("Evaluation complete | mean reward: None | std: None")

            evaluation_dti_plot_dir = checkpoint_dir_path / "evaluation_dti_plots" / selected_service
            generated_eval_plots = save_episode_dti_plots(
                episode_data=evaluation_episode_dti_series,
                output_dir=evaluation_dti_plot_dir,
            )
            evaluation_logger.info(
                f"Saved per-episode DTI evaluation plots: {generated_eval_plots} files at {evaluation_dti_plot_dir}"
            )

            print(f"\nEvaluation complete | mean reward: {mean_reward:.4f} | std: {std_reward:.4f}")
            print("Evaluation finished. Returning to evaluation menu...")
        except Exception as e:
            print(f"ERROR: Evaluation failed: {e}")
            print("Returning to evaluation menu...")
        finally:
            for handler in list(evaluation_logger.handlers):
                handler.flush()
                handler.close()
                evaluation_logger.removeHandler(handler)


def predict_resource_blocks_from_input(
    sample_input: Dict[str, Any],
    base_checkpoint_dir: Union[str, Path] = "RL_Model/checkpoints",
    seed: Optional[int] = 42,
    network_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Predict per-DTI RB allocations for each service using trained SAC checkpoints.

    This function reuses existing `NetworkSACEnv` and `SACAgent` inference logic
    (`env.infer_rb_from_traffic(...)`) without rewriting policy behavior.
    
    Args:
        sample_input: Traffic input dictionary
        base_checkpoint_dir: Directory containing checkpoints
        seed: Random seed
        network_config: Network configuration for beta calculation
    """
    try:
        from SAC_RL_Model.trainer import load_config
        from SAC_RL_Model.env_wrapper import NetworkSACEnv
        from SAC_RL_Model.agent import SACAgent
    except Exception as e:
        raise RuntimeError(f"RL inference modules could not be imported: {e}") from e

    if not isinstance(sample_input, dict):
        raise ValueError("sample_input must be a dictionary")

    if "traffic_users_per_tti" not in sample_input:
        raise ValueError("sample_input must contain top-level key 'traffic_users_per_tti'")

    traffic_all = sample_input["traffic_users_per_tti"]
    if not isinstance(traffic_all, dict):
        raise ValueError("'traffic_users_per_tti' must be a dictionary")

    required_services = ["voip", "cbr"]
    missing_services = [s for s in required_services if s not in traffic_all]
    if missing_services:
        raise ValueError(f"Missing service keys in traffic_users_per_tti: {missing_services}")

    base_checkpoint_dir_path = Path(base_checkpoint_dir)
    if not base_checkpoint_dir_path.is_absolute():
        base_checkpoint_dir_path = BASE_DIR / base_checkpoint_dir_path

    predicted_per_dti: Dict[str, List[int]] = {}
    predicted_per_tti: Dict[str, List[List[int]]] = {}
    beta_per_dti: Dict[str, List[float]] = {}
    reward_per_dti: Dict[str, List[float]] = {}

    for service in required_services:
        ckpt_path = resolve_random_final_checkpoint(base_checkpoint_dir_path, service)
        print(f"Loading random final checkpoint for {service}: {ckpt_path}")

        env_service = service
        env_seed = seed
        env_rb_min = None

        config_path = ckpt_path.with_name(f"{ckpt_path.stem}_config.json")
        if config_path.exists():
            loaded_config = load_config(config_path)
            env_cfg = loaded_config.get("environment", {})
            if isinstance(env_cfg, dict):
                env_service = env_cfg.get("service", env_service)
                env_seed = env_cfg.get("seed", env_seed)
                env_rb_min = env_cfg.get("rb_min", env_rb_min)

        env_kwargs: Dict[str, Any] = {"service": env_service, "seed": env_seed, "silent": True}
        if env_rb_min is not None:
            env_kwargs["rb_min"] = env_rb_min

        env = NetworkSACEnv(**env_kwargs)
        state_dim = int(np.prod(env.observation_shape))
        action_dim = int(np.prod(env.action_shape))

        agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load(ckpt_path)

        service_traffic = traffic_all[service]
        if not isinstance(service_traffic, list):
            raise ValueError(f"traffic_users_per_tti.{service} must be a list of DTI traffic vectors")

        service_rb_per_dti: List[int] = []
        service_rb_per_tti: List[List[int]] = []
        service_beta_per_dti: List[float] = []
        service_reward_per_dti: List[float] = []

        for dti_index, traffic_dti in enumerate(service_traffic):
            if not isinstance(traffic_dti, list):
                raise ValueError(f"traffic_users_per_tti.{service}[{dti_index}] must be a list")
            if len(traffic_dti) != env.n:
                raise ValueError(
                    f"traffic_users_per_tti.{service}[{dti_index}] must have length n={env.n}, got {len(traffic_dti)}"
                )

            rb_pred = int(env.infer_rb_from_traffic(traffic_dti=traffic_dti, agent=agent, dti_index=dti_index))
            if rb_pred < env.rb_min or (env.rb_max is not None and rb_pred > env.rb_max):
                raise ValueError(
                    f"Predicted RB out of bounds for {service} DTI {dti_index}: {rb_pred} not in [{env.rb_min}, {env.rb_max}]"
                )

            service_rb_per_dti.append(rb_pred)
            service_rb_per_tti.append([rb_pred] * env.n)

            # Calculate beta_current if network_config is provided
            beta_val = float("nan")
            reward_val = float("nan")
            if network_config is not None:
                try:
                    # Create RB data from predicted RB (replicate for each TTI)
                    rb_dti = [rb_pred] * env.n
                    
                    # Get network model
                    model = NetworkModel(
                        n=network_config['n'],
                        m=network_config['m'],
                        k=network_config['k'],
                        traffic_elements=network_config['traffic_elements'],
                        q_thresholds_voip=network_config['q_thresholds_voip'],
                        q_thresholds_cbr=network_config['q_thresholds_cbr'],
                        q_thresholds_streaming=network_config['q_thresholds_streaming']
                    )
                    
                    # Load metric matrices for the service
                    config_dir = BASE_DIR / "Network_Model" / "data" / "configuration"
                    service_files = {
                        'voip': config_dir / 'D2min_VoIP_summary.json',
                        'cbr': config_dir / 'D30sec_CBR_summary.json'
                    }
                    json_file = service_files.get(service)
                    if json_file and json_file.exists():
                        metric_data = load_metric_matrices(json_file)
                        model.set_metric_matrices(service, metric_data)
                        model.set_service(service)
                        model.reset(service)
                        
                        # Compute beta
                        beta_result = model.compute_beta(traffic_dti, rb_dti)
                        beta_val = float(beta_result.beta_current)
                        
                        # Compute reward
                        c_capacity = network_config['c']
                        service_rb_index = {"voip": 0, "cbr": 1}[service]
                        rb_used = network_config['resource_blocks_per_dti'][dti_index][service_rb_index]
                        lambda_reward = network_config['lambda_reward']
                        
                        reward_val = model.compute_reward_current(
                            beta_val, c_capacity, rb_used, lambda_reward
                        )
                        reward_val = float(reward_val)
                except Exception as e:
                    print(f"Warning: Failed to calculate beta for {service} DTI {dti_index}: {e}")
            
            service_beta_per_dti.append(beta_val)
            service_reward_per_dti.append(reward_val)

        predicted_per_dti[service] = service_rb_per_dti
        predicted_per_tti[service] = service_rb_per_tti
        beta_per_dti[service] = service_beta_per_dti
        reward_per_dti[service] = service_reward_per_dti

    return {
        "predicted_resource_blocks_per_dti": predicted_per_dti,
        "predicted_resource_blocks_per_tti": predicted_per_tti,
        "beta_values_per_dti": beta_per_dti,
        "reward_values_per_dti": reward_per_dti,
    }


def _json_safe(value: Any) -> Any:
    """Recursively convert objects to JSON-serializable and finite-safe values."""
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_sac_custom_inference_mode() -> None:
    try:
        from SAC_RL_Model.config import get_default_config
    except Exception as e:
        print(f"ERROR: RL config modules could not be imported: {e}")
        return

    cfg = get_default_config()
    sample_input = cfg.environment.sample_input
    checkpoint_dir_cfg = Path(cfg.checkpoint.checkpoint_dir)
    base_checkpoint_dir = checkpoint_dir_cfg if checkpoint_dir_cfg.is_absolute() else BASE_DIR / checkpoint_dir_cfg

    # Load network configuration for beta calculation
    network_config = None
    try:
        CONFIG_DIR = BASE_DIR / "Network_Model" / "data" / "configuration"
        INPUT_DIR = BASE_DIR / "Network_Model" / "data" / "input"
        config_file = CONFIG_DIR / "network_config.json"
        input_file = INPUT_DIR / "network_input.json"
        
        if config_file.exists() and input_file.exists():
            network_config = load_configuration(config_file, input_file)
            print("Loaded network configuration for beta calculation")
        else:
            print("WARNING: Network configuration files not found; beta calculation will be skipped")
    except Exception as e:
        print(f"WARNING: Failed to load network configuration: {e}")

    try:
        output = predict_resource_blocks_from_input(
            sample_input=sample_input,
            base_checkpoint_dir=base_checkpoint_dir,
            seed=42,
            network_config=network_config,
        )
    except Exception as e:
        print(f"ERROR: Custom SAC inference failed: {e}")
        return

    print("\n" + "=" * 60)
    print("CUSTOM SAC INFERENCE OUTPUT")
    print("=" * 60)
    for service, rb_list in output["predicted_resource_blocks_per_dti"].items():
        print(f"\nService: {service}")
        for dti_index, rb in enumerate(rb_list):
            print(f"  DTI {dti_index} -> Predicted RB: {int(rb)}")

    output_path = base_checkpoint_dir / "custom_inference_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(output), indent=2), encoding="utf-8")
    print("\nSaved custom inference output to:")
    print(output_path)
    # Generate and save sample-input plots (only for custom/sample input simulation mode)
    try:
        traffic_users = sample_input.get("traffic_users_per_tti", {})
        services = ["voip", "cbr"]
        default_profiles = list(get_default_profiles().values())
        model_type = "sac"

        def infer_profile_number_from_dti(dti_values: Any) -> float:
            if not isinstance(dti_values, (list, tuple)) or not dti_values:
                return float("nan")

            try:
                dti_set = {int(v) for v in dti_values}
            except (TypeError, ValueError):
                return float("nan")

            for profile_idx, profile_values in enumerate(default_profiles, start=1):
                if dti_set.issubset({int(v) for v in profile_values}):
                    return float(profile_idx)

            dti_mean = float(np.mean([float(v) for v in dti_values]))
            profile_means = [float(np.mean([float(v) for v in profile_values])) for profile_values in default_profiles]
            closest_idx = int(np.argmin([abs(dti_mean - pm) for pm in profile_means])) + 1
            return float(closest_idx)

        allocation = output.get("predicted_resource_blocks_per_dti", {})
        beta_values = output.get("beta_values_per_dti", {})
        saved_plot_paths: List[Path] = []

        for svc in services:
            svc_data = traffic_users.get(svc, [])
            rb_list = allocation.get(svc, [])
            svc_beta_values = beta_values.get(svc, [])
            
            if not svc_data or not rb_list:
                continue

            plot_dir = base_checkpoint_dir / svc / "sample_input_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Create combined plot with 3 subplots: Traffic Profile, RB Allocation, and Beta
            fig, (ax_left, ax_center, ax_right) = plt.subplots(1, 3, figsize=(22, 6))

            x_vals = list(range(1, min(len(svc_data), len(rb_list)) + 1))
            profile_numbers = [infer_profile_number_from_dti(dti) for dti in svc_data[: len(x_vals)]]
            rb_values = [int(x) for x in rb_list[: len(x_vals)]]

            # Left plot: Traffic Profile
            ax_left.plot(x_vals, profile_numbers, marker="o", linewidth=1.5, label=svc.capitalize())
            ax_left.set_title(f"Traffic Profile Number per DTI - {svc.capitalize()}")
            ax_left.set_xlabel("Number of DTI")
            ax_left.set_ylabel("Profile Number")
            ax_left.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
            ax_left.set_ylim(1, 8)
            ax_left.set_xticks(range(1, 9))
            ax_left.set_xlim(1, 8)
            ax_left.grid(True)
            ax_left.legend()
            
            # Add service and model label to left plot
            label_text = f"Service: {svc.upper()}\nModel: {model_type.upper()}"
            ax_left.text(
                0.98, 0.05, label_text,
                transform=ax_left.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )

            # Center plot: RB Allocation
            ax_center.plot(x_vals, rb_values, marker="o", linewidth=1.5, label=svc.capitalize(), color="darkorange")
            ax_center.set_title(f"Agent RB Allocation per DTI - {svc.capitalize()}")
            ax_center.set_xlabel("Number of DTI")
            ax_center.set_ylabel("Allocated RBs")
            ax_center.set_xticks(range(1, 9))
            ax_center.set_xlim(1, 8)
            ax_center.grid(True)
            ax_center.legend()
            
            # Add service and model label to center plot
            ax_center.text(
                0.98, 0.05, label_text,
                transform=ax_center.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )

            # Right plot: Beta values
            beta_x_vals = list(range(1, len(svc_beta_values) + 1))
            has_valid_beta = svc_beta_values and not all(np.isnan(v) for v in svc_beta_values)
            
            if has_valid_beta:
                ax_right.plot(beta_x_vals, svc_beta_values, marker="o", linewidth=1.5, label=svc.capitalize(), color="steelblue")
                ax_right.set_title(f"Beta Current per DTI - {svc.capitalize()}")
                ax_right.set_xlabel("Number of DTI")
                ax_right.set_ylabel("beta_current")
                ax_right.set_xticks(range(1, 9))
                ax_right.set_xlim(1, 8)
                ax_right.grid(True)
                ax_right.legend()
            else:
                ax_right.text(0.5, 0.5, "No valid beta data", 
                             ha="center", va="center", transform=ax_right.transAxes, fontsize=12)
                ax_right.set_title(f"Beta Current per DTI - {svc.capitalize()}")
                ax_right.set_xlabel("Number of DTI")
                ax_right.set_ylabel("beta_current")
            
            # Add service and model label to right plot
            ax_right.text(
                0.98, 0.05, label_text,
                transform=ax_right.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )

            fig.tight_layout()
            service_file = plot_dir / f"{svc}_combined_traffic_rb_beta_plots.png"
            fig.savefig(service_file, dpi=150)
            plt.close(fig)
            saved_plot_paths.append(service_file)

        print("\nSaved combined prediction plots (Traffic, RB, Beta) to:")
        for plot_path in saved_plot_paths:
            print(plot_path)

    except Exception as e:
        print(f"WARNING: Failed to generate sample-input plots: {e}")



if __name__ == "__main__":
    np.random.seed(42)

    CONFIG_DIR = NETWORK_MODEL_DIR / "data" / "configuration"
    INPUT_DIR = NETWORK_MODEL_DIR / "data" / "input"
    OUTPUT_DIR = NETWORK_MODEL_DIR / "data" / "output"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    LOG_FILE = OUTPUT_DIR / "debug.log"
    logger = setup_logging(str(LOG_FILE))
    
    config_file = CONFIG_DIR / "network_config.json"
    input_file = INPUT_DIR / "network_input.json"
    
    if not config_file.exists():
        print(f"ERROR: Configuration file not found: {config_file}")
        exit(1)
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        exit(1)
    
    try:
        config = load_configuration(config_file, input_file)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        exit(1)

    while True:
        display_main_execution_menu()
        mode_choice = get_main_execution_choice()

        if mode_choice == "1":
            run_networkmodel(config=config, config_dir=CONFIG_DIR, logger=logger)
        elif mode_choice == "2":
            run_sac_training_mode()
        elif mode_choice == "3":
            run_sac_evaluation_mode()
        elif mode_choice == "4":
            run_sac_custom_inference_mode()
        else:
            break
    
