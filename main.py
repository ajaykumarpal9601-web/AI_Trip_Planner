from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
import os
import json
import requests  # For calling OpenAI API (or use openai library if installed)

app = FastAPI(title="AI Trip Planner API", version="1.0.0")

# If calling from the browser, allow all origins (for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ItineraryRequest(BaseModel):
    destination: str
    startDate: str  # YYYY-MM-DD
    endDate: str    # YYYY-MM-DD
    partySize: int = Field(default=1)
    budgetUsd: Optional[float] = None
    interests: Optional[List[str]] = Field(default_factory=list)
    notes: Optional[str] = ""

class DayPlan(BaseModel):
    title: str
    details: str
    note: Optional[str] = ""

class ItineraryResponse(BaseModel):
    days: List[DayPlan]

def generate_ai_itinerary(destination, start_date, end_date, party_size, budget_usd, interests, notes):
    # Calculate number of days
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    num_days = max(1, (end - start).days + 1)
    
    # Prepare prompt for AI
    interests_str = ", ".join(interests) if interests else "general interests"
    budget_str = f" with a budget of ${budget_usd}" if budget_usd else ""
    notes_str = f" Additional notes: {notes}" if notes else ""
    
    prompt = f"""
Generate a detailed {num_days}-day itinerary for a trip to {destination} from {start_date} to {end_date} for {party_size} travelers.
Interests: {interests_str}{budget_str}.{notes_str}
Return the response as a JSON object with a key "days" containing an array of objects, each with "title", "details", and "note" fields.
Example format:
{{
  "days": [
    {{
      "title": "Day 1 — {destination}",
      "details": "Morning: Visit landmark. Afternoon: Explore museum. Evening: Dinner at local restaurant.",
      "note": "Focus on culture"
    }},
    ...
  ]
}}
Ensure the itinerary is realistic, includes specific activities, and considers the interests and budget.
"""
    
    # Call OpenAI API (replace with your API key and endpoint)
    api_key = os.getenv("OPENAI_API_KEY")  # Set this environment variable
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",  # or gpt-4 if available
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to generate itinerary from AI")
    
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    
    # Parse the JSON response
    try:
        itinerary = json.loads(content)
        return itinerary
    except json.JSONDecodeError:
        # Fallback: extract days from text if JSON fails
        lines = content.split("\n")
        days = []
        for i, line in enumerate(lines):
            if line.strip():
                days.append(DayPlan(
                    title=f"Day {i+1} — {destination}",
                    details=line.strip(),
                    note=""
                ))
        return {"days": days}

@app.post("/api/itinerary", response_model=ItineraryResponse)
async def generate_itinerary(req: ItineraryRequest):
    try:
        itinerary = generate_ai_itinerary(
            req.destination, req.startDate, req.endDate, req.partySize, req.budgetUsd, req.interests, req.notes
        )
        return itinerary
    except Exception as e:
        # Fallback to simple logic if AI fails
        start = datetime.strptime(req.startDate, "%Y-%m-%d")
        end = datetime.strptime(req.endDate, "%Y-%m-%d")
        num_days = max(1, (end - start).days + 1)
        destination = req.destination or "your amazing destination"
        activities = [
            "Explore the top landmarks, museums, and cultural sites.",
            "City walking tour, local market visit, and downtown cafes.",
            "Excursion to popular parks or nature reserves.",
            "Try regional cuisine and street food. Food tour night.",
            "Relax in a scenic area or day-trip to nearby towns.",
            "Visit historic districts, art galleries, or creative events."
        ]
        days = []
        for i in range(num_days):
            act = activities[i % len(activities)]
            interest_list = req.interests or []
            interest = interest_list[i % len(interest_list)] if interest_list else ""
            days.append(DayPlan(
                title=f"Day {i + 1} — {destination}",
                details=f"{act}" + (f" Try something focused on: {interest}" if interest else ""),
                note=req.notes if i == 0 and req.notes else ""
            ))
        return {"days": days}

# Run this file with: uvicorn filename:app --reload
