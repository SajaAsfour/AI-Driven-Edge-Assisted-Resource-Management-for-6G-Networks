from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def build_ue_profiles(
    ue_values: Sequence[int],
    values_per_profile: int = 2,
) -> Dict[str, List[int]]:
    """
    Build contiguous UE traffic profiles from original UE values.

    Example:
        ue_values=[5,10,15,20] ->
        {
            "profile_1": [5,10],
            "profile_2": [15,20],
        }
    """
    values = [int(v) for v in ue_values]
    if not values:
        raise ValueError("ue_values cannot be empty")
    if values_per_profile <= 0:
        raise ValueError("values_per_profile must be > 0")
    if len(values) % values_per_profile != 0:
        raise ValueError(
            "ue_values length must be divisible by values_per_profile "
            f"(got {len(values)} and {values_per_profile})"
        )

    if any(v < 0 for v in values):
        raise ValueError("ue_values must be >= 0")

    profiles: Dict[str, List[int]] = {}
    profile_idx = 1
    for i in range(0, len(values), values_per_profile):
        chunk = values[i:i + values_per_profile]
        if len(chunk) != values_per_profile:
            raise ValueError("internal profile chunk length mismatch")
        profiles[f"profile_{profile_idx}"] = chunk
        profile_idx += 1

    return profiles


def build_dti_from_profile(
    profile_values: Sequence[int],
    n: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
        Build one DTI vector by random sampling from selected profile values.

    Rule:
            - For each TTI position, randomly pick one value from `profile_values`.
            - Every generated value must belong to the selected profile.

    Examples:
            - profile [5, 10], n=8 -> random values like [5, 10, 10, 5, 5, 10, 5, 10]
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    vals = [int(v) for v in profile_values]
    if not vals:
        raise ValueError("profile_values must be non-empty")
    if any(v < 0 for v in vals):
        raise ValueError("profile_values must be >= 0")

    chooser = rng.choice if rng is not None else random.choice
    out = [int(chooser(vals)) for _ in range(n)]

    if len(out) != n:
        raise ValueError("built DTI length mismatch")
    return out

#matrix for the fixed profile mode
def build_traffic_matrix_from_profile(
    profile_values: Sequence[int],
    m: int,
    n: int,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    """Build full traffic matrix [m x n] from one selected profile.

    Each DTI row is generated independently by random sampling from
    `profile_values` via `build_dti_from_profile(...)`.
    """
    if m <= 0:
        raise ValueError("m must be > 0")

    matrix = [
        build_dti_from_profile(profile_values=profile_values, n=n, rng=rng)
        for _ in range(m)
    ]

    if len(matrix) != m:
        raise ValueError("traffic matrix row count mismatch")
    for row in matrix:
        if len(row) != n:
            raise ValueError("traffic matrix column count mismatch")
    return matrix


def validate_dti_values_in_profile(dti: Sequence[int], profile_values: Sequence[int], n: int) -> None:
    """Validate DTI shape and value membership against selected profile values."""
    if len(dti) != n:
        raise ValueError(f"DTI length must be n={n}, got {len(dti)}")

    allowed = {int(v) for v in profile_values}
    if not allowed:
        raise ValueError("profile_values must be non-empty")
    if any(v < 0 for v in allowed):
        raise ValueError("profile_values must be >= 0")

    for idx, v in enumerate(dti):
        if int(v) not in allowed:
            raise ValueError(
                f"DTI value out of selected profile at index {idx}: {v} not in {sorted(allowed)}"
            )


def canonical_ue_values_from_range(start: int = 5, end: int = 80, step: int = 5) -> List[int]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if end < start:
        raise ValueError("end must be >= start")
    return list(range(int(start), int(end) + 1, int(step)))


def get_default_profiles() -> Dict[str, List[int]]:
    """Default UE profiles from [5..80] with step 5, 2 values per profile."""
    return build_ue_profiles(canonical_ue_values_from_range(), values_per_profile=2)


def get_profile_or_raise(profiles: Mapping[str, Sequence[int]], profile_name: str) -> Tuple[str, List[int]]:
    if profile_name not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")
    return profile_name, [int(v) for v in profiles[profile_name]]
