from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(slots=True)
class AllocationDecision:
	requests: List[int]
	allocations: List[int]
	total_requested: int
	capacity: int
	scaling_applied: bool

	@property
	def total_allocated(self) -> int:
		return int(sum(self.allocations))


def _clamp_int(value: int, lower: int, upper: int) -> int:
	if lower > upper:
		raise ValueError(f"lower bound must be <= upper bound, got {lower} > {upper}")
	return max(lower, min(upper, int(value)))


def proportional_allocate_requests(
	requests: Sequence[int],
	capacity: int = 8,
	min_rb: int = 1,
) -> AllocationDecision:
	"""Allocate N RB requests under a shared capacity using proportional fairness.

	Rules:
	- If total requested RBs do not exceed capacity, keep the requests as-is.
	- If total requested RBs exceed capacity, allocate proportionally and convert
	  to integer RB values using a largest-remainder step so the final total never
	  exceeds capacity.
	- Supports any number of requests (2, 3, or more).
	"""
	if not requests:
		raise ValueError("requests must contain at least one value")
	if capacity < 0:
		raise ValueError(f"capacity must be >= 0, got {capacity}")
	if min_rb < 0:
		raise ValueError(f"min_rb must be >= 0, got {min_rb}")
	if min_rb * len(requests) > capacity:
		raise ValueError(
			f"min_rb * num_requests must be <= capacity, got {min_rb} * {len(requests)} > {capacity}"
		)

	clamped = [_clamp_int(value, min_rb, capacity) for value in requests]
	total_requested = int(sum(clamped))

	if total_requested <= capacity:
		return AllocationDecision(
			requests=clamped,
			allocations=list(clamped),
			total_requested=total_requested,
			capacity=capacity,
			scaling_applied=False,
		)

	if total_requested == 0:
		return AllocationDecision(
			requests=clamped,
			allocations=[0] * len(clamped),
			total_requested=0,
			capacity=capacity,
			scaling_applied=True,
		)

	exact = [(value / total_requested) * capacity for value in clamped]
	base = [int(value) for value in exact]
	frac = [exact[i] - base[i] for i in range(len(exact))]

	allocations = list(base)
	remaining = capacity - sum(allocations)

	if remaining > 0:
		# Largest fractional remainder first; ties broken by lowest index for determinism.
		order = sorted(range(len(allocations)), key=lambda i: (-frac[i], i))
		index = 0
		while remaining > 0:
			allocations[order[index % len(order)]] += 1
			remaining -= 1
			index += 1

	allocations = [_clamp_int(value, min_rb, capacity) for value in allocations]
	while sum(allocations) > capacity:
		# Shrink the largest allocation above min_rb until total fits capacity.
		candidates = [i for i in range(len(allocations)) if allocations[i] > min_rb]
		if not candidates:
			break
		shrink_idx = max(candidates, key=lambda i: allocations[i])
		allocations[shrink_idx] -= 1

	return AllocationDecision(
		requests=clamped,
		allocations=allocations,
		total_requested=total_requested,
		capacity=capacity,
		scaling_applied=True,
	)
