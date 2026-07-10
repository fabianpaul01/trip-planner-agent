"""Evaluation functions ("evals") for scoring a travel plan.

At a high level, this file is an evaluation framework for a travel planner.
 Its responsibility is not to generate a travel plan, but to verify whether a 
 generated travel plan satisfies a collection of independent rules.

The core idea is to decouple each validation rule from the evaluation engine. Instead of writing one giant function that checks everything (dates, budget, interests, weather, etc.), each rule is implemented as an independent evaluator. The evaluation engine simply executes all of them and aggregates the results.

Pure evals are module-level functions with the signature
``(vacation_info, final_output) -> None`` that raise :class:`AgentError` on
failure. Evals that need an LLM (weather compatibility, feedback incorporation)
are built by factory functions so the client/model can be injected explicitly
instead of relying on notebook globals.

TravelPlan
     │
     ▼
+------------------------+
| Evaluation Engine       |
| (get_eval_results)      |
+------------------------+
     │
     ├── Date evaluator
     ├── Budget evaluator
     ├── Cost evaluator
     ├── Event evaluator
     ├── Interest evaluator
     ├── Weather evaluator
     └── Feedback evaluator
     │
     ▼
EvaluationResults
"""

from __future__ import annotations

from typing import Callable, List, Optional

from openai import OpenAI

from .config import OpenAIModel
from .models import (
    Activity,
    AgentError,
    EvaluationResults,
    TravelPlan,
    VacationInfo,
)
from .project_lib import (
    ChatAgent,
    call_activity_by_id_api_mocked,
    do_chat_completion,
    print_in_box,
)
from .prompts import ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT

# An eval takes the inputs and raises AgentError on failure.
# The common interface is a callable with signature (vacation_info, final_output) -> None.
# Every evaluator follows the same contract:
EvalFn = Callable[[VacationInfo, TravelPlan], None]


def get_eval_results(
    vacation_info: VacationInfo,
    final_output: TravelPlan,
    eval_functions: List[EvalFn],
) -> EvaluationResults:
    """Run each eval, collecting failures into an :class:`EvaluationResults`."""
    if not isinstance(vacation_info, VacationInfo):
        raise ValueError("vacation_info must be an instance of VacationInfo")
    if not isinstance(final_output, TravelPlan):
        raise ValueError("final_output must be an instance of TravelPlan")
    if not isinstance(eval_functions, list) or not all(
        callable(fn) for fn in eval_functions
    ):
        raise ValueError("eval_functions must be a list of callable functions")

    failures: List[str] = []
    for eval_fn in eval_functions:
        try:
            eval_fn(vacation_info, final_output)
        except AgentError as e:
            error_msg = str(e)
            print_in_box(error_msg, title="Evaluation Error")
            failures.append(error_msg)

    return EvaluationResults(
        success=len(failures) == 0,
        failures=failures,
        eval_functions=[fn.__name__ for fn in eval_functions],
    )


# --------------------------------------------------------------------------- #
# Pure (non-LLM) evaluation functions
# --------------------------------------------------------------------------- #
def eval_start_end_dates_match(vacation_info: VacationInfo, final_output: TravelPlan) -> None:
    """Arrival/departure dates must match the plan's start/end dates."""
    if (
        vacation_info.date_of_arrival != final_output.start_date
        or vacation_info.date_of_departure != final_output.end_date
    ):
        raise AgentError(
            f"Dates do not match: {vacation_info.date_of_arrival} != {final_output.start_date} "
            f"or {vacation_info.date_of_departure} != {final_output.end_date}"
        )
    if final_output.start_date > final_output.end_date:
        raise AgentError(
            f"Start date is after end date: {final_output.start_date} > {final_output.end_date}"
        )


def eval_total_cost_is_accurate(vacation_info: VacationInfo, final_output: TravelPlan) -> None:
    """Stated total cost must equal the sum of activity prices."""
    actual_total_cost = sum(
        rec.activity.price
        for day in final_output.itinerary_days
        for rec in day.activity_recommendations
    )
    stated_total_cost = int(final_output.total_cost)
    if actual_total_cost != stated_total_cost:
        raise AgentError(
            "Stated total cost does not match calculated total cost: "
            f"{actual_total_cost} != {stated_total_cost}"
        )


def eval_total_cost_is_within_budget(vacation_info: VacationInfo, final_output: TravelPlan) -> None:
    """Total cost must not exceed the budget."""
    stated_total_cost = int(final_output.total_cost)
    if stated_total_cost > vacation_info.budget:
        raise AgentError(
            f"Total cost exceeds budget: {stated_total_cost} > {vacation_info.budget}"
        )


def eval_itinerary_events_match_actual_events(
    vacation_info: VacationInfo, final_output: TravelPlan
) -> None:
    """Every activity in the plan must match a real (non-hallucinated) event."""
    event_ids_not_matching: List[str] = []
    event_ids_missing: List[str] = []

    for day in final_output.itinerary_days:
        for rec in day.activity_recommendations:
            event_id = rec.activity.activity_id
            reference_event = call_activity_by_id_api_mocked(event_id)

            if reference_event is None:
                event_ids_missing.append(event_id)
            elif Activity(**reference_event) != rec.activity:
                print(
                    "---\n"
                    f"Event ID {event_id} does not match the reference event:\n"
                    f"Reference Event: {reference_event}\n"
                    f"Activity Event: {rec.activity.model_dump()}"
                )
                event_ids_not_matching.append(event_id)

    if event_ids_missing or event_ids_not_matching:
        raise AgentError(
            f"Event IDs missing: {event_ids_missing}\n"
            f"Event IDs not matching: {event_ids_not_matching}"
        )


def eval_itinerary_satisfies_interests(
    vacation_info: VacationInfo, final_output: TravelPlan
) -> None:
    """Each traveler must get at least one activity matching their interests."""
    traveler_to_interests = {t.name: set(t.interests) for t in vacation_info.travelers}
    hit_counts = {t.name: 0 for t in vacation_info.travelers}

    for name, interests in traveler_to_interests.items():
        for day in final_output.itinerary_days:
            for rec in day.activity_recommendations:
                matching = interests & set(rec.activity.related_interests)
                if matching:
                    hit_counts[name] += 1
                    print(
                        f"✅ Traveler {name} matches interest {matching} at {rec.activity.name}"
                    )

    travelers_with_no_hits = [name for name, count in hit_counts.items() if count == 0]
    if travelers_with_no_hits:
        raise AgentError(
            f"Travelers {travelers_with_no_hits} have no matches with the itinerary."
        )


# --------------------------------------------------------------------------- #
# LLM-backed evaluation functions (built via factories so the client is injected)
# --------------------------------------------------------------------------- #
def make_eval_activities_and_weather_are_compatible(
    client: OpenAI,
    model: OpenAIModel = OpenAIModel.GPT_41_NANO,
) -> EvalFn:
    """Build an eval that asks an LLM if each activity suits that day's weather."""

    def eval_activities_and_weather_are_compatible(
        vacation_info: VacationInfo, final_output: TravelPlan
    ) -> None:
        incompatible: List[str] = []

        for day in final_output.itinerary_days:
            weather_condition = day.weather.condition
            for rec in day.activity_recommendations:
                resp = do_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Activity: {rec.activity.name}\n"
                                f"Description: {rec.activity.description}\n"
                                f"Weather Condition: {weather_condition}"
                            ),
                        },
                    ],
                    client=client,
                    model=model,
                )

                if "IS_INCOMPATIBLE" in (resp or ""):
                    incompatible.append(rec.activity.name)
                    print(
                        f"❌ {rec.activity.name} (on {day.date}) is incompatible with '{weather_condition}'."
                    )
                elif "IS_COMPATIBLE" in (resp or ""):
                    print(
                        f"✅ {rec.activity.name} (on {day.date}) is compatible with '{weather_condition}'."
                    )
                else:
                    raise RuntimeError(
                        f"Unexpected response from the model: {resp}. "
                        "Expected 'IS_COMPATIBLE' or 'IS_INCOMPATIBLE'."
                    )

        if incompatible:
            raise AgentError(
                f"Activities that may be ruined by inclement weather: {incompatible}"
            )

    return eval_activities_and_weather_are_compatible


def make_eval_traveler_feedback_is_incorporated(
    client: OpenAI,
    traveler_feedback: str,
    model: OpenAIModel = OpenAIModel.GPT_41,
) -> EvalFn:
    """Build an eval that checks whether traveler feedback was incorporated."""

    def eval_traveler_feedback_is_incorporated(
        vacation_info: VacationInfo, final_output: TravelPlan
    ) -> None:
        agent = ChatAgent(
            system_prompt="""You are an expert in evaluating whether a travel plan incorporates traveler feedback.

    ## Output Format

    Respond using two sections (ANALYSIS and FINAL OUTPUT) in the following format:

        ANALYSIS:
        * [step-by-step analysis]

        FINAL OUTPUT:
        [FULLY_INCORPORATED, PARTIALLY_INCORPORATED, NOT_INCORPORATED, or UNKNOWN]
        REASON: [reasoning for the final output]
    """,
            client=client,
            model=model,
        )

        resp = agent.chat(
            f"""Traveler Feedback: {traveler_feedback}
    Revised Travel Plan: {final_output.model_dump_json()}
    """,
        )
        if "FINAL OUTPUT:" not in resp:
            raise RuntimeError(
                f"Unexpected response from the model: {resp}. Expected 'FINAL OUTPUT:'."
            )
        if "FULLY_INCORPORATED" not in resp:
            verdict = resp.split("FINAL OUTPUT:")[-1].strip()
            raise AgentError(
                "Traveler feedback was not successfully incorporated into the revised "
                f"travel plan. Response: {verdict}"
            )

    return eval_traveler_feedback_is_incorporated


def build_base_eval_functions() -> List[EvalFn]:
    """The pure (non-LLM) evals that need no client."""
    return [
        eval_start_end_dates_match,
        eval_total_cost_is_accurate,
        eval_itinerary_events_match_actual_events,
        eval_itinerary_satisfies_interests,
        eval_total_cost_is_within_budget,
    ]


def build_all_eval_functions(
    client: OpenAI,
    traveler_feedback: Optional[str] = None,
) -> List[EvalFn]:
    """The full eval suite. Adds the feedback eval only when feedback is given."""
    evals = build_base_eval_functions()
    evals.append(make_eval_activities_and_weather_are_compatible(client))
    if traveler_feedback:
        evals.append(
            make_eval_traveler_feedback_is_incorporated(client, traveler_feedback)
        )
    return evals
