from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AllocationDecision:
	request_1: int
	request_2: int
	allocation_1: int
	allocation_2: int
	total_requested: int
	capacity: int
	scaling_applied: bool

	@property
	def total_allocated(self) -> int:
		return int(self.allocation_1 + self.allocation_2)


def _clamp_int(value: int, lower: int, upper: int) -> int:
	if lower > upper:
		raise ValueError(f"lower bound must be <= upper bound, got {lower} > {upper}")
	return max(lower, min(upper, int(value)))


def proportional_allocate_two_requests(
	request_1: int,
	request_2: int,
	capacity: int = 8,
	min_rb: int = 1,
) -> AllocationDecision:
	"""Allocate two RB requests under a shared capacity using proportional fairness.

	Rules:
	- If total requested RBs do not exceed capacity, keep the requests as-is.
	- If total requested RBs exceed capacity, allocate proportionally and convert
	  to integer RB values using a largest-remainder step so the final total never
	  exceeds capacity.
	"""
	if capacity < 0:
		raise ValueError(f"capacity must be >= 0, got {capacity}")
	if min_rb < 0:
		raise ValueError(f"min_rb must be >= 0, got {min_rb}")
	if min_rb > capacity:
		raise ValueError(f"min_rb must be <= capacity, got {min_rb} > {capacity}")

	r1 = _clamp_int(request_1, min_rb, capacity)
	r2 = _clamp_int(request_2, min_rb, capacity)
	total_requested = int(r1 + r2)

	if total_requested <= capacity:
		return AllocationDecision(
			request_1=r1,
			request_2=r2,
			allocation_1=r1,
			allocation_2=r2,
			total_requested=total_requested,
			capacity=capacity,
			scaling_applied=False,
		)

	if total_requested == 0:
		return AllocationDecision(
			request_1=0,
			request_2=0,
			allocation_1=0,
			allocation_2=0,
			total_requested=0,
			capacity=capacity,
			scaling_applied=True,
		)

	exact_1 = (r1 / total_requested) * capacity
	exact_2 = (r2 / total_requested) * capacity

	base_1 = int(exact_1)
	base_2 = int(exact_2)
	frac_1 = exact_1 - base_1
	frac_2 = exact_2 - base_2

	alloc_1 = base_1
	alloc_2 = base_2
	remaining = capacity - (alloc_1 + alloc_2)

	if remaining > 0:
		frac_pairs = [(1, frac_1), (2, frac_2)]
		frac_pairs.sort(key=lambda item: (-item[1], item[0]))
		index = 0
		while remaining > 0:
			service_idx = frac_pairs[index % len(frac_pairs)][0]
			if service_idx == 1:
				alloc_1 += 1
			else:
				alloc_2 += 1
			remaining -= 1
			index += 1

	alloc_1 = _clamp_int(alloc_1, min_rb, capacity)
	alloc_2 = _clamp_int(alloc_2, min_rb, capacity)
	while alloc_1 + alloc_2 > capacity:
		if alloc_1 >= alloc_2 and alloc_1 > min_rb:
			alloc_1 -= 1
		elif alloc_2 > min_rb:
			alloc_2 -= 1
		else:
			break

	return AllocationDecision(
		request_1=r1,
		request_2=r2,
		allocation_1=alloc_1,
		allocation_2=alloc_2,
		total_requested=total_requested,
		capacity=capacity,
		scaling_applied=True,
	)
