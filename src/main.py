"""
Main Application Module

This module serves as the entry point for the FastAPI application that provides:
1. Chat functionality with an LLM
2. Infrastructure diagram generation
3. AI assistant capabilities

The application exposes several REST endpoints:
- /chat: For direct interaction with the LLM
- /generate-diagram: For creating infrastructure diagrams
- /assistant: For interacting with the AI assistant
- /assistant/history: For retrieving conversation history

The application can run in two modes:
- Production mode: Uses real LLM services
- Mock mode: Uses mock services for development/testing (controlled by USE_MOCK env var)
"""

import os  # For environment variables and file operations
from typing import List, Optional, Dict  # Type hints for better code clarity
from fastapi import FastAPI, HTTPException, Depends  # Web framework and error handling
from fastapi.responses import FileResponse  # For serving generated diagram files
from pydantic import BaseModel  # For request/response validation
import tempfile  # For temporary file operations
import logging  # For application logging
from services import LLMService, DiagramService, AssistantService  # Core service implementations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Initialize FastAPI app
app = FastAPI(title="Gemini API Integration")

# Determine if we should use mock services
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

# Initialize services
llm_service = LLMService(use_mock=USE_MOCK)
diagram_service = DiagramService(llm_service)
assistant_service = AssistantService(llm_service)

@app.on_event("startup")
async def startup_event():
    """Print the ASCII art banner after application startup."""
    print("\n  API Integration: http://localhost:8000/docs#/\n")

class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.
    
    Attributes:
        messages: List of conversation messages
        temperature: Controls response randomness (default: 0.7)
        max_tokens: Maximum response length (default: 1000)
    """
    messages: List[dict]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ChatResponse(BaseModel):
    """
    Response model for chat endpoint.
    
    Attributes:
        response: The generated response text
    """
    response: str

class DiagramRequest(BaseModel):
    """
    Request model for diagram generation endpoint.
    
    Attributes:
        description: Infrastructure description to generate diagram from
    """
    description: str

class AssistantRequest(BaseModel):
    """
    Request model for assistant endpoint.
    
    Attributes:
        message: User's message to the assistant
        context: Optional context for the conversation
    """
    message: str
    context: Optional[Dict] = None

class AssistantResponse(BaseModel):
    """
    Response model for assistant endpoint.
    
    Attributes:
        response: Assistant's response text
        context: Updated conversation context
        history: Conversation history
    """
    response: str
    context: Dict
    history: List[Dict]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for direct interaction with the LLM.
    
    Args:
        request: ChatRequest containing messages and optional parameters
        
    Returns:
        ChatResponse containing the generated response
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        response = llm_service.generate_response(
            request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-diagram")
async def generate_diagram(request: DiagramRequest):
    """
    Endpoint for generating infrastructure diagrams.
    
    Args:
        request: DiagramRequest containing the infrastructure description
        
    Returns:
        FileResponse containing the generated diagram image
        
    Raises:
        HTTPException: If diagram generation fails
    """
    try:
        # Generate the diagram
        diagram_path = diagram_service.generate_diagram(request.description)
        
        if not os.path.exists(diagram_path):
            raise HTTPException(status_code=500, detail="Failed to generate diagram")

        # Return the diagram file
        return FileResponse(
            diagram_path,
            media_type="image/png",
            filename=os.path.basename(diagram_path)
        )

    except Exception as e:
        logger.error(f"Error in generate-diagram endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant", response_model=AssistantResponse)
async def assistant(request: AssistantRequest):
    """
    Endpoint for interacting with the AI assistant.
    
    Args:
        request: AssistantRequest containing the message and optional context
        
    Returns:
        AssistantResponse containing the response, context, and history
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        response = assistant_service.process_message(
            request.message,
            context=request.context
        )
        return AssistantResponse(**response)
    except Exception as e:
        logger.error(f"Error in assistant endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/history")
async def get_assistant_history():
    """
    Endpoint for retrieving assistant conversation history.
    
    Returns:
        List of conversation messages
        
    Raises:
        HTTPException: If there's an error retrieving the history
    """
    try:
        return assistant_service.get_history()
    except Exception as e:
        logger.error(f"Error getting assistant history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add cleanup of old temporary files
def cleanup_old_files():
    tmp_dir = "tmp"
    if os.path.exists(tmp_dir):
        for file in os.listdir(tmp_dir):
            if file.endswith(".png"):
                file_path = os.path.join(tmp_dir, file)
                # Remove files older than 24 hours
                if os.path.getmtime(file_path) < time.time() - 86400:
                    os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
