"""Tools available to the ReAct revision agent.

The main purpose of this file is to define the tools that the ReAct agent can use to revise a travel plan.
The tools are designed to be stateless and independent, so they can be used in any order
and combined in different ways. The tools are also designed to be composable, so they can be combined to create more complex tools.

(``vacation_info``, ``ALL_EVAL_FUNCTIONS``) are built by factory functions so
those dependencies are injected explicitly.
"""

from __future__ import annotations

from typing import Callable, List

from .evaluations import EvalFn, get_eval_results
from .models import Activity, TravelPlan, VacationInfo
from .project_lib import call_activities_api_mocked


def get_tool_descriptions_string(fns: List[Callable]) -> str:
    """Format a list of tool functions into a bulleted description string."""
    resp = ""
    for fn in fns:
        function_name = fn.__name__
        function_doc = fn.__doc__ or "No description provided."
        resp += f"* `{function_name}`: {function_doc}\n"
    return resp


def calculator_tool(input_expression: str) -> float:
    """Evaluates a mathematical expression and returns the result as a float.

    Args:
        input_expression (str): A valid mathematical expression to evaluate.

    Returns:
        float: The result of the evaluated expression.

    Example:
        >>> calculator_tool("1 + 1")
        2.0
    """
    import numexpr as ne

    return float(ne.evaluate(input_expression))


def get_activities_by_date_tool(date: str, city: str) -> List[dict]:
    """Returns the list of available activities for a given date and city.

    Use this to discover activities you can add to or swap into the itinerary.

    Args:
        date (str): The date to look up, in ``YYYY-MM-DD`` format.
        city (str): The city to look up (only ``Voyagia`` is supported).

    Returns:
        List[dict]: Each dict is a validated activity (id, name, times, price, interests, ...).
    """
    resp = call_activities_api_mocked(date=date, city=city)
    return [Activity.model_validate(activity).model_dump() for activity in resp]


def final_answer_tool(final_output: TravelPlan) -> TravelPlan:
    """Returns the final travel plan.

    Args:
        final_output (TravelPlan): The final travel plan to return.

    Returns:
        TravelPlan: The final travel plan.
    """
    return final_output


def make_run_evals_tool(
    vacation_info: VacationInfo,
    eval_functions: List[EvalFn],
) -> Callable[[TravelPlan | dict], dict]:
    """Build a `run_evals_tool` bound to a vacation and eval suite."""

    def run_evals_tool(travel_plan: TravelPlan | dict) -> dict:
        """Runs all evaluation functions on the provided travel plan.

        Args:
            travel_plan (TravelPlan): The travel plan to evaluate.

        Returns:
            dict: ``{"success": bool, "failures": [str, ...]}``.
        """
        if isinstance(travel_plan, dict):
            travel_plan = TravelPlan.model_validate(travel_plan)

        resp = get_eval_results(
            vacation_info=vacation_info,
            final_output=travel_plan,
            eval_functions=eval_functions,
        )
        return {"success": resp.success, "failures": resp.failures}

    return run_evals_tool


def build_all_tools(
    vacation_info: VacationInfo,
    eval_functions: List[EvalFn],
) -> List[Callable]:
    """Assemble the full tool list for the revision agent."""
    return [
        calculator_tool,
        get_activities_by_date_tool,
        make_run_evals_tool(vacation_info, eval_functions),
        final_answer_tool,
    ]
