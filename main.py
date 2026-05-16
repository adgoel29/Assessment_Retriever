from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import SHLAgent
from faiss_store import FAISSStore
import os



class Message(BaseModel):
    role: str      # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool



store = FAISSStore()
agent: SHLAgent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    catalog_path = os.environ.get("CATALOG_PATH", "catalog.json")
    store.load(catalog_path)
    agent = SHLAgent(store)
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = agent.chat(messages)

    return ChatResponse(
        reply=result["reply"],
        recommendations=result.get("recommendations", []),
        end_of_conversation=result.get("end_of_conversation", False),
    )