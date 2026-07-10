# Copyright (c) 2026 Fabian Paul. All rights reserved.

"""The itinerary agents.

This module provides AI agents that create structured day-by-day travel plans
from vacation details and iteratively improve them using a ReAct workflow. The
revision agent executes validation and utility tools, incorporates their
observations, and produces a validated final itinerary.
"""

from __future__ import annotations

import json
from typing import Callable, List, Optional

from .config import OpenAIModel
from .models import TravelPlan, VacationInfo
from .project_lib import ChatAgent, print_in_box


class ItineraryAgent(ChatAgent):
    """Generates an initial day-by-day itinerary from vacation info."""

    def get_itinerary(
        self,
        vacation_info: VacationInfo,
        model: Optional[OpenAIModel] = None,
    ) -> TravelPlan:
        """Ask the model for an itinerary and parse it into a :class:`TravelPlan`."""
        response = (
            self.chat(
                user_message=vacation_info.model_dump_json(indent=2),
                add_to_messages=False,
                model=model or self.model,
            )
            or ""
        ).strip()

        print_in_box(response, "Raw Response")

        json_text = response
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()

        try:
            return TravelPlan.model_validate_json(json_text)
        except Exception:
            print("Error validating the following text as TravelPlan JSON:")
            print(json_text)
            raise


class ItineraryRevisionAgent(ChatAgent):
    """ReAct agent that revises an itinerary using tools until evals pass."""

    def __init__(
        self,
        tools: List[Callable],
        system_prompt: str,
        client=None,
        model: Optional[OpenAIModel] = None,
        name: Optional[str] = None,
    ) -> None:
        self.tools = tools
        super().__init__(
            name=name, system_prompt=system_prompt, client=client, model=model
        )

    def get_observation_string(self, tool_call_obj: dict) -> str:
        """Execute the requested tool call and format the OBSERVATION string."""
        if "tool_name" not in tool_call_obj:
            return "OBSERVATION: No tool name specified."
        if "arguments" not in tool_call_obj:
            return "OBSERVATION: No arguments specified."
        if not isinstance(tool_call_obj["arguments"], dict):
            return (
                "OBSERVATION: Arguments should be a dictionary, got "
                f"{type(tool_call_obj['arguments'])} instead."
            )
        if not isinstance(tool_call_obj["tool_name"], str):
            return (
                "OBSERVATION: Tool name should be a string, got "
                f"{type(tool_call_obj['tool_name'])} instead."
            )

        tool_name = tool_call_obj["tool_name"]
        arguments = tool_call_obj["arguments"]

        tool_fn = next((t for t in self.tools if t.__name__ == tool_name), None)
        if tool_fn is None:
            return f"OBSERVATION: Unknown tool name '{tool_name}' in action string."

        try:
            tool_response = tool_fn(**arguments)
            return (
                f"OBSERVATION: Tool {tool_name} called successfully with "
                f"response: {tool_response}"
            )
        except Exception as e:
            return f"OBSERVATION: Error occurred while calling tool {tool_name}: {e}"

    def run_react_cycle(
        self,
        original_travel_plan: TravelPlan,
        max_steps: int = 10,
        model: Optional[OpenAIModel] = None,
        client=None,
    ) -> TravelPlan:
        """Run THOUGHT/ACTION/OBSERVATION cycles until a final answer is produced."""
        from json_repair import repair_json

        self.add_message(
            role="user",
            content=(
                "Here is the itinerary for revision:\n"
                f"{original_travel_plan.model_dump_json()}"
            ),
        )
        resp = None

        for _ in range(max_steps):
            resp = self.get_response(model=model, client=client) or ""

            if "ACTION:" not in resp:
                self.add_message(role="user", content="No action found in response.")
                continue

            action_string = resp.split("ACTION:")[1].strip()

            try:
                action_string = repair_json(action_string)
                tool_call_obj = json.loads(action_string)
            except json.JSONDecodeError:
                print(f"Invalid JSON in action string: {action_string}")
                self.add_message(
                    role="user",
                    content=f"Invalid JSON in action string: {action_string}",
                )
                continue

            tool_name = tool_call_obj.get("tool_name", None)

            if tool_name == "final_answer_tool":
                try:
                    return TravelPlan.model_validate(
                        tool_call_obj["arguments"].get(
                            "final_output", tool_call_obj["arguments"]
                        )
                    )
                except Exception as e:
                    self.add_message(
                        role="user", content=f"Error validating final answer: {e}"
                    )
                    continue
            else:
                observation_string = self.get_observation_string(
                    tool_call_obj=tool_call_obj
                )
                self.add_message(role="user", content=observation_string)

        raise RuntimeError(
            f"ReAct cycle did not complete within {max_steps} steps. Last response: {resp}"
        )
