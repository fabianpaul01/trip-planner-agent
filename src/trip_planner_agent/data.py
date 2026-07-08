"""Default vacation input data and loader.

In a real-time system this dict would come from an
onboarding chatbot; here we ship a sensible default and validate it into a
:class:`~trip_planner_agent.models.VacationInfo`.
"""

from __future__ import annotations

from typing import Any, Optional

from .models import VacationInfo

# Mock data exists for Voyagia between 2026-08-15 and 2026-08-31.
DEFAULT_VACATION_INFO_DICT: dict[str, Any] = {
    "travelers": [
        {
            "name": "",
            "age": 60,
            "interests": ["tennis", "cooking", "comedy", "technology"],
        },
        {
            "name": "",
            "age": 60,
            "interests": ["reading", "music", "theatre", "art"],
        },
    ],
    "destination": "Voyagia",
    "date_of_arrival": "2026-08-15",
    "date_of_departure": "2026-08-31",
    "budget": 1000,  # Budget is in fictional currency units.
}


def load_vacation_info(data: Optional[dict[str, Any]] = None) -> VacationInfo:
    """Validate raw vacation data into a :class:`VacationInfo`.

    Args:
        data: A dict of vacation details. Falls back to
            :data:`DEFAULT_VACATION_INFO_DICT` when ``None``.

    Returns:
        A validated :class:`VacationInfo` instance.
    """
    return VacationInfo.model_validate(data or DEFAULT_VACATION_INFO_DICT)
