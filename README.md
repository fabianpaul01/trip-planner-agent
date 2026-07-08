# Trip Planner Agent

An AI-powered travel planning application that generates personalized, 
day-by-day itineraries using an LLM. The system creates tailored travel 
experiences based on user preferences and demonstrates:

- **Role-based prompting** — a specialized itinerary-planning agent
- **Chain-of-thought reasoning** — step-by-step day-by-day planning
- **ReAct prompting** — a revision agent that uses tools (THOUGHT → ACTION → OBSERVATION)
- **Evaluation feedback loops** — programmatic + LLM-as-judge evals that gate the plan

## Project layout

```
src/trip_planner_agent/
├── config.py        # env + OpenAI client + model enum
├── models.py        # Pydantic data models (VacationInfo, TravelPlan, ...)
├── data.py          # default vacation input + loader
├── schedule.py      # weather + activity retrieval for the trip dates
├── prompts.py       # prompt templates / builders
├── evaluations.py   # eval functions + get_eval_results
├── tools.py         # calculator / activities / run_evals / final_answer tools
├── agents.py        # ItineraryAgent, ItineraryRevisionAgent
├── planner.py       # plan_trip() orchestration
├── main.py          # CLI entry point
└── project_lib.py   # vendored provided SDK (ChatAgent, mocked APIs, data)
tests/               # offline unit tests (no API key required)
docs/                # reference PDFs
```

Note: `project_lib.py` is the course-provided "SDK" (the mocked weather/activity
APIs, the `ChatAgent` base class, and the Voyagia data). It is vendored
unchanged so the mocked APIs remain the single source of truth.

## Setup

```bash
uv sync                 # install dependencies into .venv
cp .env.example .env    # then add your OPENAI_API_KEY
```

## Run

```bash
uv run trip-planner-agent          # generate + revise an itinerary and print it
# or
uv run python -m trip_planner_agent.main
```

## Use as a library

```python
from trip_planner_agent import get_client, load_vacation_info, plan_trip

client = get_client()
vacation_info = load_vacation_info()          # or pass your own dict
result = plan_trip(vacation_info, client, traveler_feedback="At least two activities per day.")

print(result.final_plan.model_dump_json(indent=2))
print(result.revised_eval.success)
```

## Test

```bash
uv run pytest            # offline tests — no API key needed
```

The tests exercise the pure logic (models, schedule retrieval, tools, and the
non-LLM evals) against the mocked APIs, so they run without network access.
Mock data covers **Voyagia, 2025-06-10 → 2025-06-15**.
