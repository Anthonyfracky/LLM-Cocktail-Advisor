from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, Optional
import uuid
from pydantic import BaseModel
from main import CocktailSystem
import asyncio
import os

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store active sessions
sessions: Dict[str, CocktailSystem] = {}
preference_stores: Dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/session")
async def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = CocktailSystem()
    preference_stores[session_id] = []
    return {"session_id": session_id}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.session_id not in sessions:
        return {"error": "Invalid session"}

    system = sessions[request.session_id]
    response = await system.process_query(request.message)

    # Get updated preferences
    preferences = []
    if system.preference_store.preferences:
        preferences = [p["text"] for p in system.preference_store.preferences]
        preference_stores[request.session_id] = preferences

    return {
        "response": response,
        "preferences": preferences
    }


@app.post("/api/recommendations")
async def get_recommendations(request: ChatRequest):
    if request.session_id not in sessions:
        return {"error": "Invalid session"}

    system = sessions[request.session_id]
    stored_preferences = preference_stores.get(request.session_id, [])

    if not stored_preferences:
        return {"recommendations": "No preferences stored yet. Tell me what kind of drinks you like!"}

    preferences_text = " ".join(stored_preferences)
    response = await system.process_query(
        f"Based on these preferences: {preferences_text}, what cocktails would you recommend?"
    )

    return {"recommendations": response}


@app.on_event("startup")
async def startup_event():
    # Ensure required directories exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)


# Session cleanup (optional, implement based on your needs)
@app.on_event("shutdown")
async def shutdown_event():
    sessions.clear()
    preference_stores.clear()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)