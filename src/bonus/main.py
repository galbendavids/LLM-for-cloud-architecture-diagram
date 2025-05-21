import os
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
from datetime import datetime
import json
from diagrams import Cluster, Edge, Diagram, Server

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Gemini API Integration - Bonus Features")

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# In-memory storage for conversation history
conversation_history: Dict[str, List[dict]] = {}

class ChatRequest(BaseModel):
    messages: List[dict]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str

async def save_conversation(conversation_id: str, message: dict, response: str):
    """Save conversation to history"""
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []
    
    conversation_history[conversation_id].extend([
        message,
        {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
    ])

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{datetime.now().timestamp()}"

        # System prompt to guide the LLM as an architecture assistant
        system_prompt = (
            "You are a helpful assistant that helps users design and finalize their cloud architecture. "
            "Ask clarifying questions about requirements, missing details, and best practices. "
            "If the user seems ready, gently encourage them to use the POST /generate-diagram endpoint to visualize their architecture. "
            "If the user says 'ready', 'done', 'generate diagram', 'finished', or similar, respond with: 'Your architecture is ready! Please use the POST /generate-diagram endpoint to generate your diagram.' "
            "Otherwise, continue to ask helpful questions or offer suggestions to help the user finalize their architecture."
        )

        # Prepare chat history: start with system prompt, then user messages
        chat = model.start_chat(history=[{"role": "system", "content": system_prompt}])

        # Process each message
        for message in request.messages:
            if message["role"] == "user":
                user_content = message["content"].strip().lower()
                response = chat.send_message(
                    message["content"],
                    generation_config=genai.types.GenerationConfig(
                        temperature=request.temperature,
                        max_output_tokens=request.max_tokens
                    )
                )

                # Check if user is ready to generate the diagram
                ready_phrases = ["ready", "done", "generate diagram", "finished", "let's generate"]
                if any(phrase in user_content for phrase in ready_phrases):
                    response_text = (
                        "Your architecture is ready! Please use the POST /generate-diagram endpoint to generate your diagram."
                    )
                else:
                    # Encourage the user to use the diagram endpoint if their description is detailed enough
                    if len(user_content.split()) > 30 or any(word in user_content for word in ["architecture", "diagram", "infrastructure"]):
                        response_text = response.text + "\n\nIf you feel your architecture is ready, you can use the POST /generate-diagram endpoint to visualize it!"
                    else:
                        response_text = response.text

                # Save conversation in background
                background_tasks.add_task(
                    save_conversation,
                    conversation_id,
                    message,
                    response_text
                )

                return ChatResponse(
                    response=response_text,
                    conversation_id=conversation_id,
                    timestamp=datetime.now().isoformat()
                )

        raise HTTPException(status_code=400, detail="No user message found in the conversation")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history for a specific conversation ID"""
    if conversation_id not in conversation_history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation_history[conversation_id]

@app.get("/conversations")
async def list_conversations():
    """List all conversation IDs"""
    return list(conversation_history.keys())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 