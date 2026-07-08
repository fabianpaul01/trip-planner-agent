""" Trip Planner — an LLM travel-planning agent.

Provides: role-based prompting,
chain-of-thought planning, ReAct tool use, and evaluation feedback loops.
"""

from .config import DEFAULT_MODEL, OpenAIModel, get_client
from .data import DEFAULT_VACATION_INFO_DICT, load_vacation_info
from .models import (
    Activity,
    ActivityRecommendation,
    AgentError,
    EvaluationResults,
    Interest,
    ItineraryDay,
    Traveler,
    TravelPlan,
    VacationInfo,
    Weather,
)
from .planner import PlanResult, generate_initial_plan, plan_trip, revise_plan

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "get_client",
    "OpenAIModel",
    "DEFAULT_MODEL",
    # data
    "load_vacation_info",
    "DEFAULT_VACATION_INFO_DICT",
    # models
    "Interest",
    "Traveler",
    "VacationInfo",
    "Weather",
    "Activity",
    "ActivityRecommendation",
    "ItineraryDay",
    "TravelPlan",
    "AgentError",
    "EvaluationResults",
    # workflow
    "plan_trip",
    "generate_initial_plan",
    "revise_plan",
    "PlanResult",
]
