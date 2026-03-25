from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Optional
import numpy as np

Number = Union[int, float]


@dataclass
class MetricEvaluation:
    """Container for individual metric evaluation at one TTI"""
    metric_name: str
    qos_value: float
    threshold: float
    direction: str
    binary: int  
    status: str 


@dataclass
class TTIEvaluation:
    """Container for combined evaluation of all metrics at one TTI"""
    tti_index: int
    traffic: int
    rbs: int
    metric_evaluations: List[MetricEvaluation]
    combined_binary: int  
    combined_status: str  
    failed_users: int  


@dataclass
class BetaResult:
    """Container for beta calculation results for a single DTI with combined metrics"""
    service_type: str
    dti_index: int
    
    # TTI evaluations (one per TTI)
    tti_evaluations: List[TTIEvaluation]
    
    # DTI-level statistics
    dti_total_failures: int
    dti_total_traffic: int
    
    # Beta values
    beta_current: float
    beta_cumulative: float
    
    # Cumulative counters
    cumulative_failures: int
    cumulative_traffic: int
    
    # Metrics used
    metrics_evaluated: List[str]
    thresholds: Dict[str, float]
    directions: Dict[str, str]


@dataclass
class CDFResult:
    """Container for CDF calculation results for a single DTI"""
    dti_index: int
    traffic_array: List[int]
    all_traffic_values: np.ndarray
    cdf_x: np.ndarray
    cdf_y: np.ndarray
    num_total_values: int


@dataclass
class DTIResult:
    """Container for complete results of one DTI (CDF + Beta)"""
    service: str
    dti_index: int
    cdf_result: CDFResult
    beta_result: BetaResult
    reward_current: float


@dataclass
class NetworkModel:
    """Network Model for CDF and Beta computations."""
    # Basic parameters
    n: int
    m: int
    k: int

    # Traffic elements array
    traffic_elements: List[Any]

    # Q-thresholds per service
    q_thresholds_voip: Dict[str, Number]
    q_thresholds_cbr: Dict[str, Number]
    q_thresholds_streaming: Dict[str, Number]

    # Metric matrices
    metric_matrices: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Internal state
    _current_traffic_data: Optional[List[int]] = field(default=None, init=False)
    _current_rb_data: Optional[List[int]] = field(default=None, init=False)
    _current_service: Optional[str] = field(default=None, init=False)
    
    # CDF tracking
    _cumulative_traffic: Dict[str, List[int]] = field(default_factory=dict, init=False)
    _dti_count: Dict[str, int] = field(default_factory=dict, init=False)
    
    # Beta tracking
    _cumulative_failures: Dict[str, int] = field(default_factory=dict, init=False)
    _cumulative_traffic_beta: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initialize tracking structures."""
        self._cumulative_traffic = {'voip': [], 'cbr': [], 'streaming': []}
        self._dti_count = {'voip': 0, 'cbr': 0, 'streaming': 0}
        self._cumulative_failures = {'voip': 0, 'cbr': 0, 'streaming': 0}
        self._cumulative_traffic_beta = {'voip': 0, 'cbr': 0, 'streaming': 0}

    # SETTERS AND GETTERS
    
    def set_traffic(self, traffic_data: List[int]) -> None:
        if len(traffic_data) != self.n:
            raise ValueError(f"traffic_data must have length {self.n}, got {len(traffic_data)}")
        self._current_traffic_data = traffic_data.copy()
    
    def set_resource_blocks(self, rb_data: List[int]) -> None:
        if len(rb_data) != self.n:
            raise ValueError(f"rb_data must have length {self.n}, got {len(rb_data)}")
        self._current_rb_data = rb_data.copy()
    
    def set_service(self, service: str) -> None:
        if service not in ['voip', 'cbr', 'streaming']:
            raise ValueError(f"Unknown service: {service}")
        self._current_service = service
    
    def get_traffic(self) -> Optional[List[int]]:
        return self._current_traffic_data.copy() if self._current_traffic_data else None
    
    def get_resource_blocks(self) -> Optional[List[int]]:
        return self._current_rb_data.copy() if self._current_rb_data else None
    
    def get_service(self) -> Optional[str]:
        return self._current_service
    
    def set_metric_matrices(self, service: str, metric_data: Dict[str, Any]) -> None:
        if service not in ['voip', 'cbr', 'streaming']:
            raise ValueError(f"Unknown service: {service}")
        
        required_keys = ['metadata', 'mean_matrices', 'std_matrices']
        for key in required_keys:
            if key not in metric_data:
                raise ValueError(f"metric_data must contain '{key}' key")
        
        self.metric_matrices[service] = metric_data

    # RESET
    
    def reset(self, traffic_type: str = None) -> None:
        """Reset cumulative tracking data."""
        if traffic_type is None:
            self._cumulative_traffic = {'voip': [], 'cbr': [], 'streaming': []}
            self._dti_count = {'voip': 0, 'cbr': 0, 'streaming': 0}
            self._cumulative_failures = {'voip': 0, 'cbr': 0, 'streaming': 0}
            self._cumulative_traffic_beta = {'voip': 0, 'cbr': 0, 'streaming': 0}
        elif traffic_type in ['voip', 'cbr', 'streaming']:
            self._cumulative_traffic[traffic_type] = []
            self._dti_count[traffic_type] = 0
            self._cumulative_failures[traffic_type] = 0
            self._cumulative_traffic_beta[traffic_type] = 0
        else:
            raise ValueError(f"Unknown traffic type: {traffic_type}")

    # THRESHOLD HELPERS
    
    def get_threshold_and_direction(self, service: str, metric_name: str) -> tuple[float, str]:
        if service == 'voip':
            thresholds = self.q_thresholds_voip
        elif service == 'cbr':
            thresholds = self.q_thresholds_cbr
        elif service == 'streaming':
            thresholds = self.q_thresholds_streaming
        else:
            raise ValueError(f"Unknown service: {service}")
        
        if metric_name not in thresholds:
            raise ValueError(f"Metric {metric_name} not found in q_thresholds for {service}")
        
        threshold = float(thresholds[metric_name])
        direction = self._infer_direction(metric_name)
        
        return threshold, direction
    
    def get_all_metrics_for_service(self, service: str) -> List[str]:
        if service not in self.metric_matrices:
            raise ValueError(f"Metric matrices not loaded for service: {service}")
        
        return self.metric_matrices[service]['metadata']['metrics']
    
    @staticmethod
    def _infer_direction(metric_name: str) -> str:
        metric_name_lower = metric_name.lower()
        if "throughput" in metric_name_lower:
            return "below"
        return "above"

    # CDF COMPUTATION
    
    def compute_cdf(self, traffic_data: List[int], rb_data: List[int]) -> CDFResult:
        """Compute CDF for ONE DTI."""
        if self._current_service is None:
            raise ValueError("Service not set. Call set_service() first.")
        
        service = self._current_service
        
        self._cumulative_traffic[service].extend(traffic_data)
        self._dti_count[service] += 1
        
        all_traffic_values = np.array(self._cumulative_traffic[service])
        sorted_traffic = np.sort(all_traffic_values)
        
        cdf_x = np.array(self.traffic_elements)
        cdf_y = []
        
        for val in cdf_x:
            count = np.sum(sorted_traffic <= val)
            prob = count / len(sorted_traffic)
            cdf_y.append(prob)
        
        cdf_y = np.array(cdf_y)
        
        return CDFResult(
            dti_index=self._dti_count[service],
            traffic_array=traffic_data.copy(),
            all_traffic_values=all_traffic_values.copy(),
            cdf_x=cdf_x.copy(),
            cdf_y=cdf_y.copy(),
            num_total_values=len(all_traffic_values)
        )

    # BETA COMPUTATION    
    def get_qos_parameters(self, service: str, metric_name: str, 
                          traffic: int, rbs: int) -> tuple[float, float]:
        if service not in self.metric_matrices:
            raise ValueError(f"Metric matrices not loaded for service: {service}")
        
        mean_matrix = self.metric_matrices[service]['mean_matrices'][metric_name]
        std_matrix = self.metric_matrices[service]['std_matrices'][metric_name]
        rbs_values = self.metric_matrices[service]['metadata']['rbs_values']
        
        traffic_key = str(traffic)
        if traffic_key not in mean_matrix:
            raise ValueError(f"Traffic value {traffic} not found in mean matrix for {metric_name}")
        
        if rbs not in rbs_values:
            raise ValueError(f"RBS value {rbs} not found in available RBS values: {rbs_values}")
        
        rbs_idx = rbs_values.index(rbs)
        
        mean_val = mean_matrix[traffic_key][rbs_idx]
        std_val = std_matrix[traffic_key][rbs_idx]
        
        return mean_val, std_val
    
    def evaluate_single_metric(self, service: str, metric_name: str,
                               traffic: int, rbs: int) -> MetricEvaluation:
        """Evaluate a single metric for one TTI. """
        mean_val, std_val = self.get_qos_parameters(service, metric_name, traffic, rbs)
        
        
        qos_val = np.random.normal(mean_val, std_val)
        qos_val = np.clip(qos_val, 0, None)
        
        threshold, direction = self.get_threshold_and_direction(service, metric_name)
        
        if direction == 'above':
            binary = 1 if qos_val >= threshold else 0
        else:
            binary = 1 if qos_val <= threshold else 0
        
        status = "FAIL" if binary == 1 else "PASS"
        
        return MetricEvaluation(
            metric_name=metric_name,
            qos_value=qos_val,
            threshold=threshold,
            direction=direction,
            binary=binary,
            status=status
        )
    
    def evaluate_tti_combined(self, service: str, traffic: int, rbs: int,
                             tti_index: int) -> TTIEvaluation:
        """Evaluate ALL metrics for one TTI with AND logic. """
        all_metrics = self.get_all_metrics_for_service(service)
        
        metric_evaluations = []
        for metric_name in all_metrics:
            eval_result = self.evaluate_single_metric(
                service, metric_name, traffic, rbs
            )
            metric_evaluations.append(eval_result)
        
        combined_binary = 1 if any(m.binary == 1 for m in metric_evaluations) else 0
        combined_status = "FAIL" if combined_binary == 1 else "PASS"
        failed_users = traffic if combined_binary == 1 else 0
        
        return TTIEvaluation(
            tti_index=tti_index,
            traffic=traffic,
            rbs=rbs,
            metric_evaluations=metric_evaluations,
            combined_binary=combined_binary,
            combined_status=combined_status,
            failed_users=failed_users
        )
    
    def compute_beta(self, traffic_data: List[int], rb_data: List[int]) -> BetaResult:
        """Compute Beta for ONE DTI with combined metrics.  """
        if self._current_service is None:
            raise ValueError("Service not set. Call set_service() first.")
        
        service = self._current_service
        num_ttis = len(traffic_data)
        
        tti_evaluations = []
        dti_total_failures = 0
        dti_total_traffic = 0
        
        for tti_idx in range(num_ttis):
            traffic = traffic_data[tti_idx]
            rbs = rb_data[tti_idx]
            
            tti_eval = self.evaluate_tti_combined(
                service, traffic, rbs, tti_idx + 1
            )
            
            tti_evaluations.append(tti_eval)
            
            dti_total_failures += tti_eval.failed_users
            dti_total_traffic += traffic
        
        beta_current = dti_total_failures / dti_total_traffic if dti_total_traffic > 0 else 0
        
        self._cumulative_failures[service] += dti_total_failures
        self._cumulative_traffic_beta[service] += dti_total_traffic
        
        cumulative_failures = self._cumulative_failures[service]
        cumulative_traffic = self._cumulative_traffic_beta[service]
        beta_cumulative = cumulative_failures / cumulative_traffic if cumulative_traffic > 0 else 0
        
        current_dti_index = self._dti_count.get(service, 0) + 1
        
        all_metrics = self.get_all_metrics_for_service(service)
        thresholds = {}
        directions = {}
        for metric in all_metrics:
            threshold, direction = self.get_threshold_and_direction(service, metric)
            thresholds[metric] = threshold
            directions[metric] = direction
        
        return BetaResult(
            service_type=service,
            dti_index=current_dti_index,
            tti_evaluations=tti_evaluations,
            dti_total_failures=dti_total_failures,
            dti_total_traffic=dti_total_traffic,
            beta_current=beta_current,
            beta_cumulative=beta_cumulative,
            cumulative_failures=cumulative_failures,
            cumulative_traffic=cumulative_traffic,
            metrics_evaluated=all_metrics,
            thresholds=thresholds,
            directions=directions
        )

    # COMBINED DTI PROCESSING (CDF + BETA)

    def compute_reward_current(
        self,
        beta_current: Number,
        c_capacity: Number,
        rb_used: Number,
        lambda_reward: Number
    ) -> float:
        """Compute immediate reward for current DTI with input validation."""
        numeric_inputs = {
            "beta_current": beta_current,
            "c_capacity": c_capacity,
            "rb_used": rb_used,
            "lambda_reward": lambda_reward,
        }
        for name, value in numeric_inputs.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

        c_val = float(c_capacity)
        rb_val = float(rb_used)
        beta_val = float(beta_current)
        lambda_val = float(lambda_reward)

        if c_val <= 0:
            raise ValueError(f"c_capacity must be > 0, got {c_val}")
        if rb_val < 0:
            raise ValueError(f"rb_used must be >= 0, got {rb_val}")
        if rb_val > c_val:
            raise ValueError(f"rb_used ({rb_val}) must be <= c_capacity ({c_val})")

        return -beta_val + lambda_val * ((c_val - rb_val) / c_val)
    
    def process_dti(
        self,
        traffic_data: List[int],
        rb_data: List[int],
        c_capacity: Number,
        rb_used: Number,
        lambda_reward: Number,
    ) -> DTIResult:
        """
        Process ONE DTI: compute both CDF and Beta.
        Returns structured result.  
        """
        if self._current_service is None:
            raise ValueError("Service not set. Call set_service() first.")
        
        service = self._current_service
        
        cdf_result = self.compute_cdf(traffic_data, rb_data)
        beta_result = self.compute_beta(traffic_data, rb_data)
        reward_current = self.compute_reward_current(
            beta_result.beta_current,
            c_capacity,
            rb_used,
            lambda_reward,
        )
        
        return DTIResult(
            service=service,
            dti_index=cdf_result.dti_index,
            cdf_result=cdf_result,
            beta_result=beta_result,
            reward_current=reward_current,
        )
    
    def to_dict(self, dti_result: DTIResult) -> Dict[str, Any]:
        """
        Convert DTIResult to dictionary format for logging.
        Returns structured data only - no printing.
        """
        cdf = dti_result.cdf_result
        beta = dti_result.beta_result
        
        return {
            'service': dti_result.service,
            'dti_index': dti_result.dti_index,
            'reward_current': dti_result.reward_current,
            'cdf': {
                'traffic_array': cdf.traffic_array,
                'cdf_values': {
                    int(x): float(y) for x, y in zip(cdf.cdf_x, cdf.cdf_y)
                },
                'num_total_values': cdf.num_total_values
            },
            'beta': {
                'beta_current': beta.beta_current,
                'beta_cumulative': beta.beta_cumulative,
                'dti_total_failures': beta.dti_total_failures,
                'dti_total_traffic': beta.dti_total_traffic,
                'cumulative_failures': beta.cumulative_failures,
                'cumulative_traffic': beta.cumulative_traffic,
                'metrics_evaluated': beta.metrics_evaluated,
                'thresholds': beta.thresholds,
                'directions': beta.directions,
                'tti_evaluations': [
                    {
                        'tti_index': tti.tti_index,
                        'traffic': tti.traffic,
                        'rbs': tti.rbs,
                        'combined_binary': tti.combined_binary,
                        'combined_status': tti.combined_status,
                        'failed_users': tti.failed_users,
                        'metrics': [
                            {
                                'name': m.metric_name,
                                'qos_value': m.qos_value,
                                'threshold': m.threshold,
                                'direction': m.direction,
                                'binary': m.binary,
                                'status': m.status
                            }
                            for m in tti.metric_evaluations
                        ]
                    }
                    for tti in beta.tti_evaluations
                ]
            }
        }