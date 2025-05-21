"""
Mock LLM (Large Language Model) Service Module

This module provides a mock implementation of the LLM service for development and testing purposes.
It simulates the behavior of a real LLM service without making actual API calls, making it useful for:
- Development without API keys
- Testing without incurring API costs
- Offline development
- Consistent test results

The mock service provides predefined responses for:
- Diagram generation
- Chat responses
- Diagram naming
"""

from typing import List, Dict, Optional  # Type hints for better code clarity and IDE support
import logging  # For logging operations and debugging
import re  # For regular expression operations in parsing responses
import json  # For JSON parsing and formatting
from unittest.mock import Mock  # For creating mock objects in testing

logger = logging.getLogger(__name__)

class MockLLM:
    """
    Mock LLM service that simulates the behavior of a real LLM service.
    
    This class provides mock implementations of LLM operations including:
    - Parsing diagram descriptions
    - Generating chat responses
    - Managing conversation history
    - Generating content based on prompts
    """
    
    def __init__(self):
        """
        Initialize the mock LLM with predefined responses.
        
        The mock responses include:
        - A sample diagram with basic infrastructure components
        - A default chat response
        - A default diagram name
        """
        self.conversation_history = []
        self.mock_responses = {
            "diagram": {
                "nodes": [
                    {"type": "internet", "label": "Internet"},
                    {"type": "vpc", "label": "VPC"},
                    {"type": "ec2", "label": "Web Server"},
                    {"type": "rds", "label": "Database"}
                ]
            },
            "chat": "This is a mock response from the LLM service.",
            "diagram_name": "Mock Infrastructure Architecture"
        }
    
    def parse_diagram_description(self, description: str) -> List[Dict]:
        """
        Mock implementation of diagram description parsing.
        
        Args:
            description: The infrastructure description to parse
            
        Returns:
            List of dictionaries representing the infrastructure components
            
        This method:
        1. Logs the description being parsed
        2. Extracts JSON from the response
        3. Validates the JSON structure
        4. Returns the parsed nodes
        """
        logger.info(f"Mock parsing diagram description: {description}")
        response_text = description.strip()
        logger.info(f"LLM raw response: {response_text}")

        # Remove Markdown code block if present
        if response_text.startswith("```"):
            response_text = "\n".join(response_text.splitlines()[1:])
            if response_text.strip().endswith("```"):
                response_text = "\n".join(response_text.splitlines()[:-1])

        # Extract the first JSON array
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            nodes = json.loads(json_str)
        else:
            logger.error(f"LLM did not return a JSON array. Response was: {response_text}")
            raise ValueError("LLM did not return a JSON array.")

        return nodes

    def generate_chat_response(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Mock implementation of chat response generation.
        
        Args:
            messages: List of conversation messages
            temperature: Controls randomness (not used in mock)
            max_tokens: Maximum response length (not used in mock)
            
        Returns:
            A predefined mock response
        """
        logger.info(f"Mock generating chat response for messages: {messages}")
        return self.mock_responses["chat"]
    
    def add_to_history(self, message: Dict):
        """
        Add a message to the conversation history.
        
        Args:
            message: The message to add to history
        """
        self.conversation_history.append(message)
    
    def get_history(self) -> List[Dict]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history

    def generate_content(self, prompt: str) -> Mock:
        """
        Mock implementation of content generation.
        
        Args:
            prompt: The prompt to generate content for
            
        Returns:
            A mock response object with appropriate text based on the prompt type
            
        This method handles different types of prompts:
        - Diagram name generation
        - Node extraction
        - Connection extraction
        - General chat responses
        """
        logger.info(f"Mock generating content for prompt: {prompt}")
        
        mock_response = Mock()
        
        if "generate a concise and descriptive name for the diagram" in prompt:
            mock_response.text = self.mock_responses["diagram_name"]
        elif "extract the main components and their types" in prompt:
            mock_response.text = json.dumps(self.mock_responses["diagram"]["nodes"])
        elif "extract the connections between nodes" in prompt:
            mock_response.text = "[]"  # Empty connections list as mock
        else:
            mock_response.text = self.mock_responses["chat"]
            
        return mock_response 