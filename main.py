from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np

# Ensure Network_Model is importable when running main.py from repo root.
BASE_DIR = Path(__file__).parent.resolve()
NETWORK_MODEL_DIR = BASE_DIR / "Network_Model"
if str(NETWORK_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_MODEL_DIR))

from Network_Model.src.NetworkModel import NetworkModel
Number = Union[int, float]


# LOGGING CONFIGURATION

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
        if len(row) != 3:
            raise ValueError(f"resource_blocks_per_dti[{dti}] must have 3 values")

        rb_row: List[int] = []
        for svc_name, val in zip(["VoIP", "CBR", "Streaming"], row):
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
    streaming_rb_tti = []
    
    for dti_idx in range(m):
        voip_rb = resource_blocks_per_dti[dti_idx][0]
        cbr_rb = resource_blocks_per_dti[dti_idx][1]
        streaming_rb = resource_blocks_per_dti[dti_idx][2]
        
        voip_rb_tti.append([voip_rb] * n)
        cbr_rb_tti.append([cbr_rb] * n)
        streaming_rb_tti.append([streaming_rb] * n)
    
    return {
        'voip': voip_rb_tti,
        'cbr': cbr_rb_tti,
        'streaming': streaming_rb_tti
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
        "rtVideoStreamingEnd2endDelaySegment", "rtVideoStreamingSegmentLoss",
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
    streaming_users = validate_users_matrix("streaming", users.get("streaming"), m=m, n=n)

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
            'streaming': streaming_users,
        }
    }


def load_metric_matrices(service: str, json_file: Path) -> Dict[str, Any]:
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
    """Display service selection menu on console."""
    print("\n" + "="*50)
    print("NETWORK MODEL :)")
    print("="*50)
    print("Please select a service:")
    print("  1. VoIP")
    print("  2. CBR")
    print("  3. Video Streaming")
    print("  4. Exit")
    print("="*50)


def get_user_choice() -> str:
    """Get user's service selection."""
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        if choice == '1':
            return 'voip'
        elif choice == '2':
            return 'cbr'
        elif choice == '3':
            return 'streaming'
        elif choice == '4':
            return 'exit'
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


# DTI LOGGING

def log_dti_result(logger: logging.Logger, service: str, dti_index: int, 
                   cdf_result, beta_result, reward_current, traffic_dti, rb_dti):
    """Log a single DTI result to file in the required format."""
    logger.info(f"Service: {service}")
    logger.info(f"DTI Index: {dti_index}")
    logger.info(f"Traffic: {traffic_dti}")
    logger.info(f"RB: {rb_dti}")
    
    logger.info("CDF Values:")
    for traffic_val, cdf_prob in zip(cdf_result.cdf_x, cdf_result.cdf_y):
        logger.info(f"  Traffic {int(traffic_val)}: {float(cdf_prob):.4f}")
    
    logger.info(f"Beta Value: {beta_result.beta_cumulative:.4f}")
    logger.info(f"Reward (Current DTI): {reward_current:.4f}")
    logger.info("-" * 24)
    logger.info("")



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
        'voip': CONFIG_DIR / 'D2min_VoIP_summary.json',
        'cbr': CONFIG_DIR / 'D30sec_CBR_summary.json',
        'streaming': CONFIG_DIR / 'D90sec_VideoStream_summary.json'
    }
    
    service_names = {
        'voip': 'VoIP',
        'cbr': 'CBR',
        'streaming': 'Video Streaming'
    }

    service_rb_index = {
        'voip': 0,
        'cbr': 1,
        'streaming': 2,
    }
    
    while True:
        display_menu()
        service = get_user_choice()
        
        if service == 'exit':
            break
        
        json_file = service_files[service]
        try:
            metric_data = load_metric_matrices(service, json_file)
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
 
            cdf = dti_result.cdf_result
            beta = dti_result.beta_result

            print("\n" + "="*60)
            print(f"SERVICE: {service_names[service].upper()} | DTI {dti + 1}/{m}")
            print("="*60)
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
            print("="*60 + "\n")
            log_dti_result(
                logger, 
                service, 
                dti, 
                dti_result.cdf_result, 
                dti_result.beta_result,
                dti_result.reward_current,
                traffic_dti,
                rb_dti
            )
            
            
            if dti < m - 1:  
                input("Press ENTER to continue to next DTI...")

        
    
