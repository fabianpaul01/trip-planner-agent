# Copyright (c) 2026 Fabian Paul. All rights reserved.

"""Pydantic data models and shared exceptions.

Defines the core data structures used to represent travelers, vacation
preferences, activities, itineraries, evaluation results, and shared
exceptions across the application.
"""

from __future__ import annotations

import datetime
from typing import List

from pydantic import BaseModel

from .project_lib import Interest

__all__ = [
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
]


class Traveler(BaseModel):
    """A traveler with a name, age, and list of interests."""

    name: str
    age: int
    interests: List[Interest]


class VacationInfo(BaseModel):
    """Vacation details: travelers, destination, dates, and budget."""

    travelers: List[Traveler]
    destination: str
    date_of_arrival: datetime.date
    date_of_departure: datetime.date
    budget: int


class Weather(BaseModel):
    """Weather forecast for a single day."""

    temperature: float
    temperature_unit: str
    condition: str


class Activity(BaseModel):
    """A bookable activity/event."""

    activity_id: str
    name: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    location: str
    description: str
    price: int
    related_interests: List[Interest]


class ActivityRecommendation(BaseModel):
    """An activity plus the reasons it was recommended."""

    activity: Activity
    reasons_for_recommendation: List[str]


class ItineraryDay(BaseModel):
    """A single day of the itinerary: its date, weather, and recommendations."""

    date: datetime.date
    weather: Weather
    activity_recommendations: List[ActivityRecommendation]


class TravelPlan(BaseModel):
    """The full day-by-day travel plan produced by the agent."""

    city: str
    start_date: datetime.date
    end_date: datetime.date
    total_cost: int
    itinerary_days: List[ItineraryDay]


class AgentError(Exception):
    """Raised by an evaluation function when a plan fails a check."""


class EvaluationResults(BaseModel):
    """Aggregated result of running a set of evaluation functions."""

    success: bool
    failures: List[str]
    eval_functions: List[str]
