"""End-to-end planning workflow.
The central idea of this file is:

It acts as the orchestration layer (workflow coordinator) for the entire travel planning system.

It does not contain the logic for generating itineraries, evaluating plans, fetching weather, 
or revising plans. Instead, it coordinates all of these independent components into one end-to-end workflow.

This is the "glue" that wires
the schedule → prompts → agents → evals together. It contains no business rules
of its own; each responsibility lives in its own module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from .agents import ItineraryAgent, ItineraryRevisionAgent
from .config import DEFAULT_MODEL, OpenAIModel
from .evaluations import build_all_eval_functions, get_eval_results
from .models import EvaluationResults, TravelPlan, VacationInfo
from .prompts import (
    build_itinerary_agent_system_prompt,
    build_revision_agent_system_prompt,
)
from .schedule import get_activities_for_dates, get_weather_for_dates
from .tools import build_all_tools, get_tool_descriptions_string


@dataclass
class PlanResult:
    """The output of :func:`plan_trip`."""

    initial_plan: TravelPlan
    initial_eval: EvaluationResults
    revised_plan: Optional[TravelPlan] = None
    revised_eval: Optional[EvaluationResults] = None

    @property
    def final_plan(self) -> TravelPlan:
        """The revised plan if available, otherwise the initial one."""
        return self.revised_plan or self.initial_plan


def generate_initial_plan(
    vacation_info: VacationInfo,
    client: OpenAI,
    model: OpenAIModel = DEFAULT_MODEL,
) -> TravelPlan:
    """Fetch schedules, build the CoT prompt, and generate the first itinerary."""
    weather = get_weather_for_dates(vacation_info)
    activities = get_activities_for_dates(vacation_info)

    system_prompt = build_itinerary_agent_system_prompt(weather, activities)
    agent = ItineraryAgent(system_prompt=system_prompt, client=client, model=model)
    return agent.get_itinerary(vacation_info, model=model)


def revise_plan(
    vacation_info: VacationInfo,
    initial_plan: TravelPlan,
    client: OpenAI,
    traveler_feedback: str,
    model: OpenAIModel = DEFAULT_MODEL,
    max_steps: int = 15,
) -> TravelPlan:
    """Run the ReAct revision agent to incorporate feedback and pass all evals."""
    eval_functions = build_all_eval_functions(client, traveler_feedback=traveler_feedback)
    tools = build_all_tools(vacation_info, eval_functions)
    system_prompt = build_revision_agent_system_prompt(
        tool_descriptions=get_tool_descriptions_string(tools),
        traveler_feedback=traveler_feedback,
    )
    agent = ItineraryRevisionAgent(
        tools=tools, system_prompt=system_prompt, client=client, model=model
    )
    return agent.run_react_cycle(
        original_travel_plan=initial_plan,
        max_steps=max_steps,
        model=model,
        client=client,
    )


def plan_trip(
    vacation_info: VacationInfo,
    client: OpenAI,
    model: OpenAIModel = DEFAULT_MODEL,
    traveler_feedback: Optional[str] = None,
    max_steps: int = 15,
) -> PlanResult:
    """Full pipeline: generate an itinerary, evaluate it, then optionally revise.

    Args:
        vacation_info: The validated trip details.
        client: An OpenAI client (see :func:`trip_planner_agent.config.get_client`).
        model: The model to use for planning.
        traveler_feedback: If given, a revision pass is run to incorporate it.
        max_steps: Max ReAct steps for the revision agent.

    Returns:
        A :class:`PlanResult` holding the initial (and, if revised, final) plans
        and their evaluation results.
    """
    initial_plan = generate_initial_plan(vacation_info, client, model)
    initial_eval = get_eval_results(
        vacation_info, initial_plan, build_all_eval_functions(client)
    )

    result = PlanResult(initial_plan=initial_plan, initial_eval=initial_eval)

    if traveler_feedback:
        revised_plan = revise_plan(
            vacation_info,
            initial_plan,
            client,
            traveler_feedback=traveler_feedback,
            model=model,
            max_steps=max_steps,
        )

        eval_functions = build_all_eval_functions(
            client, traveler_feedback=traveler_feedback
        )

        result.revised_plan = revised_plan
        
        result.revised_eval = get_eval_results(
            vacation_info, revised_plan, eval_functions
        )

    return result
