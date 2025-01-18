import os
import uuid
from typing import Dict
from pydantic import BaseModel
from main import CocktailSystem
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

    preference_stores[session_id] = {
        "liked_ingredients": [],
        "liked_cocktails": [],
        "liked_characteristics": []
    }
    return {"session_id": session_id}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.session_id not in sessions:
        return {"error": "Invalid session"}

    system = sessions[request.session_id]
    response = await system.process_query(request.message)

    new_preferences = response.get("preferences", {})

    if any(new_preferences.get(category, []) for category in
           ["liked_ingredients", "liked_cocktails", "liked_characteristics"]):
        stored_prefs = preference_stores[request.session_id]

        for category in ["liked_ingredients", "liked_cocktails", "liked_characteristics"]:
            new_items = new_preferences.get(category, [])
            if new_items:
                stored_prefs[category] = list(set(stored_prefs[category] + new_items))

        preference_stores[request.session_id] = stored_prefs
    else:
        stored_prefs = preference_stores[request.session_id]

    return {
        "response": response["response"],
        "preferences": stored_prefs
    }


@app.post("/api/recommendations")
async def get_recommendations(request: ChatRequest):
    if request.session_id not in sessions:
        return {"error": "Invalid session"}

    system = sessions[request.session_id]
    stored_preferences = preference_stores.get(request.session_id, {})

    if not any(stored_preferences.values()):
        return {"recommendations": "No preferences stored yet. Tell me what kind of drinks you like!"}

    # Create a readable preferences string
    preferences_text = []
    if stored_preferences.get("liked_ingredients"):
        preferences_text.append(f"ingredients: {', '.join(stored_preferences['liked_ingredients'])}")
    if stored_preferences.get("liked_cocktails"):
        preferences_text.append(f"cocktails: {', '.join(stored_preferences['liked_cocktails'])}")
    if stored_preferences.get("liked_characteristics"):
        preferences_text.append(f"characteristics: {', '.join(stored_preferences['liked_characteristics'])}")

    preferences_string = "Based on your preferences for " + "; ".join(preferences_text)
    response = await system.process_query(f"{preferences_string}, what cocktails would you recommend?")

    return {"recommendations": response["response"]}


@app.on_event("startup")
async def startup_event():
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    sessions.clear()
    preference_stores.clear()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
