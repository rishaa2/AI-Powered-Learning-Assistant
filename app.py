from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal
from datetime import datetime, timedelta
import asyncio

# Import our chatbot manager
from chatbot_manager import ChatbotManager

# Initialize FastAPI app
app = FastAPI(
    title="Document Summarizer LLM Chatbot API",
    description="API for chatting with documents using LLM",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class ChatResponse(BaseModel):
    response: str

class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]

# Singleton pattern for the ChatbotManager
def get_chatbot_manager():
    if not hasattr(get_chatbot_manager, "instance"):
        get_chatbot_manager.instance = ChatbotManager()
    return get_chatbot_manager.instance

# Initialize the background tasks
@app.on_event("startup")
async def startup_event():
    app.state.chatbot_manager = get_chatbot_manager()
    
    # Schedule session cleanup task
    app.state.cleanup_task = asyncio.create_task(cleanup_sessions_periodically())

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "cleanup_task"):
        app.state.cleanup_task.cancel()
        try:
            await app.state.cleanup_task
        except asyncio.CancelledError:
            pass

async def cleanup_sessions_periodically():
    while True:
        try:
            chatbot_manager = get_chatbot_manager()
            cleaned = chatbot_manager.cleanup_old_sessions()
            print(f"Cleaned up {cleaned} old sessions")
            # Run cleanup every hour
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in cleanup task: {e}")
            await asyncio.sleep(3600)  # Still sleep to avoid tight loop on errors

@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(
    query: str = Form(...),
    session_id: Optional[str] = None,
    language: str = Query("english", description="Response language: english, bengali, or hindi"),
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """
    Process a user query and return an LLM response.
    Uses form data with an input box for the query.
    Automatically generates a session ID if not provided.
    Supports response translation to Bengali or Hindi if specified.
    """
    # Validate language parameter
    if language.lower() not in ["english", "bengali", "hindi"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid language parameter. Supported languages: english, bengali, hindi"
        )
    
    # Process the query with language preference
    response = await chatbot_manager.process_query(session_id, query, language.lower())
    
    return response

@app.get("/history/{session_id}", response_model=SessionHistoryResponse)
async def get_history(
    session_id: str,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """Get conversation history for a specific session"""
    history = chatbot_manager.get_session_history(session_id)
    
    return {
        "session_id": session_id,
        "history": history
    }

@app.get("/")
async def root():
    return {"message": "Document Summarizer Chatbot API is running. Go to /docs for documentation."}

if __name__ == "__main__":
    import uvicorn
    print("Starting Document Summarizer Chatbot API")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)