"""
Services Module

This module contains the core service classes that handle:
1. LLM (Large Language Model) interactions
2. Diagram generation and processing
3. AI Assistant functionality

The module provides three main services:
- LLMService: Handles communication with the LLM (either real or mock)
- DiagramService: Manages infrastructure diagram generation
- AssistantService: Provides AI assistant capabilities

Each service is designed to be modular and can be used independently or together.
"""

from typing import List, Dict, Optional, Union  # Type hints for better code clarity
import logging  # For application logging 
import os  # For environment variables and file operations
from dotenv import load_dotenv  # For loading environment variables
import google.generativeai as genai  # Google's Generative AI library
from mock_llm import MockLLM  # Mock LLM implementation for testing
from diagram_tools import DiagramGenerator  # Diagram generation utilities
import json  # For JSON parsing and formatting
import shutil  # For file operations
from diagrams import Diagram, Cluster  # For creating infrastructure diagrams
import re  # For regular expression operations
import uuid  # For generating unique identifiers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for handling LLM (Large Language Model) interactions.
    
    This service provides a unified interface for interacting with either:
    - Real LLM (Google's Gemini model)
    - Mock LLM (for development and testing)
    
    The service handles:
    - API key management
    - Response generation
    - Error handling
    """
    
    def __init__(self, use_mock: bool = False):
        """
        Initialize the LLM service.
        
        Args:
            use_mock: Whether to use the mock LLM implementation
        """
        self.use_mock = use_mock
        if use_mock:
            self.llm = MockLLM()
        else:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel('gemini-1.5-flash')

    def generate_response(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            temperature: Controls response randomness
            max_tokens: Maximum response length
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If there's an error generating the response
        """
        try:
            if self.use_mock:
                return self.llm.generate_chat_response(messages, temperature, max_tokens)
            
            chat = self.llm.start_chat(history=[])
            for message in messages:
                if message["role"] == "user":
                    response = chat.send_message(
                        message["content"],
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens
                        )
                    )
                    return response.text
            
            raise ValueError("No user message found in the conversation")
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise

class DiagramService:
    """
    Service for handling diagram generation.
    
    This service manages the process of:
    1. Extracting components from infrastructure descriptions
    2. Generating meaningful diagram names
    3. Creating and saving infrastructure diagrams
    4. Validating and filtering components
    
    The service uses the LLM to understand infrastructure descriptions
    and the DiagramGenerator to create visual representations.
    """
    
    def __init__(self, llm_service: LLMService):
        """
        Initialize the diagram service.
        
        Args:
            llm_service: LLM service instance for processing descriptions
        """
        self.llm_service = llm_service
        self.diagram_generator = DiagramGenerator()
    
    def _clean_json_response(self, text: str) -> str:
        """
        Clean the LLM response to extract valid JSON.
        
        Args:
            text: Raw response text from LLM
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        return text

    def _extract_diagram_name(self, description: str) -> str:
        """
        Extract a meaningful name for the diagram from the description.
        
        Args:
            description: Natural language description of the infrastructure
            
        Returns:
            A concise and descriptive name for the diagram
        """
        try:
            prompt = f"""
Given the following infrastructure description, generate a concise and descriptive name for the diagram.
The name should be professional and reflect the main purpose or key components of the infrastructure.
Keep it under 50 characters. Do not include the word "diagram" in the name.

Description:
{description}

Respond with ONLY the name, without any additional text, quotes, or formatting.
"""
            response = self.llm_service.llm.generate_content(prompt)
            name = response.text.strip()
            
            # Remove any quotes if present
            name = name.strip('"\'')
            
            # If empty or too long, use default
            if not name or len(name) > 50:
                return "Infrastructure Architecture"
                
            return name
            
        except Exception as e:
            logger.error(f"Error extracting diagram name: {str(e)}")
            return "Infrastructure Architecture"
    
    def _filter_components(self, description: str, nodes: List[Dict]) -> List[Dict]:
        """
        Review and filter components based on the original description.
        
        Args:
            description: Original user description
            nodes: List of nodes extracted by the first LLM pass
            
        Returns:
            Filtered list of nodes that were explicitly mentioned
            
        This method ensures that only components explicitly mentioned
        in the original description are included in the diagram.
        """
        try:
            prompt = f"""
You are an expert cloud architect. Review the following infrastructure description and the extracted components.
Your task is to identify and remove any components that were NOT explicitly mentioned in the original description.

Original Description:
{description}

Extracted Components:
{json.dumps(nodes, indent=2)}

IMPORTANT:
1. Only keep components that were EXPLICITLY mentioned in the original description
2. Remove any components that were inferred or added automatically
3. Do not add the Internet component unless it was explicitly mentioned
4. Keep the exact same format for the components you keep

Respond ONLY with a JSON array of the filtered components, e.g.:
[
  {{"type": "ec2", "label": "Web Server", "cluster": "Web Tier"}},
  {{"type": "rds", "label": "Database", "cluster": "DB Tier"}}
]
"""
            response = self.llm_service.llm.generate_content(prompt)
            response_text = self._clean_json_response(response.text)
            
            # Log the raw response for debugging
            logger.info(f"Raw LLM review response: {response_text}")
            
            if not response_text:
                raise ValueError("Empty response from LLM review")
            
            try:
                filtered_nodes = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM review response as JSON: {response_text}")
                raise ValueError(f"Invalid JSON response from LLM review: {str(e)}")
            
            if not isinstance(filtered_nodes, list):
                raise ValueError("LLM review response is not a list")
            
            # Validate node structure
            for node in filtered_nodes:
                if not isinstance(node, dict):
                    raise ValueError(f"Invalid node format in review: {node}")
                if "type" not in node or "label" not in node:
                    raise ValueError(f"Node missing required fields in review: {node}")
            
            return filtered_nodes
            
        except Exception as e:
            logger.error(f"Error filtering components: {str(e)}")
            # If filtering fails, return original nodes
            return nodes

    def _extract_components_with_order(self, description: str) -> List[Dict]:
        """
        Extract components and their order from the description.
        
        Args:
            description: Natural language description of the infrastructure
            
        Returns:
            List of nodes with their order in the description
            
        This method uses the LLM to:
        1. Determine the cloud provider (AWS or GCP)
        2. Extract components with their types
        3. Maintain the order of components as mentioned
        4. Use provider-specific icons for common services
        """
        try:
            prompt = f"""
You are an expert cloud architect. Given an infrastructure description, extract the main components and their types as a JSON list.
For each component, also include its position (order) in the description, where 1 is the first component mentioned.

IMPORTANT ASSUMPTIONS:
1. First, determine if the user is describing an AWS or GCP architecture. Only use elements from the detected provider. Do NOT mix AWS and GCP components in the same diagram. If you detect a mix, raise an error or ask the user to clarify.

2. For common service names, ALWAYS use the provider-specific icon:
   - For AWS diagrams:
     * "API Gateway" → use "apigateway"
     * "Load Balancer" → use "elb" or "alb"
     * "CDN" → use "cloudfront"
     * "Message Queue" → use "sqs"
     * "Storage" → use "s3"
     * "Database" → use "rds"
     * "Monitoring" → use "cloudwatch"
     * "Logging" → use "cloudtrail"
     * "DNS" → use "route53"
     * "Container Service" → use "ecs" or "fargate"
     * "Serverless" → use "lambda"
     * "Compute" → use "ec2"
     * "Kubernetes" → use "eks"

   - For GCP diagrams:
     * "API Gateway" → use "endpoints"
     * "Load Balancer" → use "load balancing"
     * "CDN" → use "cdn"
     * "Message Queue" → use "pubsub"
     * "Storage" → use "gcs"
     * "Database" → use "sql" or "spanner"
     * "Monitoring" → use "monitoring"
     * "Logging" → use "logging"
     * "DNS" → use "dns"
     * "Container Service" → use "gke"
     * "Serverless" → use "functions"
     * "Compute" → use "compute engine"
     * "Kubernetes" → use "kubernetes engine"

3. Any service or application component should be represented as EC2 (for AWS) or Compute Engine (for GCP) by default, unless the description explicitly mentions serverless or container services.

4. Only use Lambda (AWS) or Cloud Functions (GCP) if the description explicitly mentions:
   - "function"
   - "lambda"
   - "serverless"
   - "FaaS"
   - or similar serverless/function concepts

5. If a component is described as a "service" or "application" without specifying serverless/function, use EC2 (AWS) or Compute Engine (GCP).

6. For each component, match the user's term to the most specific cloud service icon available from the detected provider.

7. Do NOT include any elements from the other cloud provider.

Description:
{description}

Respond ONLY with a JSON array, e.g.:
[
  {{"type": "alb", "label": "App Load Balancer", "order": 1}},
  {{"type": "ec2", "label": "Web Server 1", "cluster": "Web Tier", "order": 2}},
  {{"type": "ec2", "label": "Web Server 2", "cluster": "Web Tier", "order": 3}},
  {{"type": "rds", "label": "Database", "cluster": "DB Tier", "order": 4}}
]

IMPORTANT:
- Ensure all components use the correct provider-specific icons
- Maintain the order of components as mentioned in the description
- Include any clustering information if mentioned
- Use the most specific service type available
"""
            response = self.llm_service.llm.generate_content(prompt)
            response_text = self._clean_json_response(response.text)
            
            try:
                nodes = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
            
            if not isinstance(nodes, list):
                raise ValueError("LLM response is not a list")
            
            # Validate node structure
            for node in nodes:
                if not isinstance(node, dict):
                    raise ValueError(f"Invalid node format: {node}")
                if "type" not in node or "label" not in node:
                    raise ValueError(f"Node missing required fields: {node}")
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error extracting components: {str(e)}")
            raise

    def generate_diagram(self, description: str) -> str:
        """
        Generate a diagram based on the description.
        Args:
            description: Natural language description of the infrastructure
        Returns:
            Path to the generated diagram
        """
        try:
            # Ensure tmp directory exists and clean it up before generating a new diagram
            tmp_dir = "tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            for f in os.listdir(tmp_dir):
                if f.endswith(".png"):
                    try:
                        os.remove(os.path.join(tmp_dir, f))
                        logger.info(f"Removed old file from tmp directory: {f}")
                    except Exception as e:
                        logger.warning(f"Failed to remove file {f} from tmp: {e}")

            # Log loading message
            logger.info("Loading... Generating your architecture diagram. Please wait.")

            # Extract diagram name
            diagram_name = self._extract_diagram_name(description)
            logger.info(f"Generated diagram name: {diagram_name}")

            # Generate unique filename
            filename = f"diagram_{uuid.uuid4().hex}.png"
            output_path = os.path.join(tmp_dir, filename)

            # First pass: Extract components with order
            logger.info("First request to Gemini (gemini-1.5-flash) - allocating all the elements from the description.")
            nodes = self._extract_components_with_order(description)
            
            # Second pass: Filter out components not in original description
            logger.info("Second request to Gemini (gemini-1.5-flash) - checking again if there are no hallucinations based on the given prompt.")
            filtered_nodes = self._filter_components(description, nodes)
            logger.info(f"Filtered nodes: {filtered_nodes}")
            
            # Filter out nodes that don't have a valid type
            filtered_nodes = [node for node in nodes if node["type"].lower() in self.diagram_generator._get_node_mapping(nodes)]
            
            # Sort nodes by their order in the description
            filtered_nodes.sort(key=lambda x: x.get('order', float('inf')))
            
            # Extract connections with architectural knowledge
            connections = self._extract_connections_with_patterns(description, filtered_nodes)
            
            # Generate the diagram with the extracted name
            logger.info(f"Generating diagram with nodes: {filtered_nodes}")
            logger.info(f"Connections: {connections}")
            
            result = self.diagram_generator.generate_diagram(filtered_nodes, connections, output_path, diagram_name)
            logger.info(f"Diagram saved to: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating diagram: {str(e)}")
            raise

    def _extract_connections_with_patterns(self, description: str, nodes: List[Dict]) -> List[List[str]]:
        """
        Extract connections between nodes from the description, using architectural patterns.
        Args:
            description: Natural language description
            nodes: List of nodes with their labels
        Returns:
            List of [source, target] pairs
        """
        try:
            prompt = f"""
You are an expert cloud architect. Given an infrastructure description and list of nodes, determine the connections between them.

IMPORTANT:
- First, determine if the user is describing an AWS or GCP architecture. Only use elements and connection patterns from the detected provider. Do NOT mix AWS and GCP components in the same diagram. If you detect a mix, raise an error or ask the user to clarify.
- If the user explicitly mentions a connection, trigger, or relationship between components, you MUST include it in the output, even if it breaks the default or left-to-right order.
- Only infer connections using architectural patterns if the user does NOT specify them.
- Do NOT omit any explicit connections or triggers described by the user.
- If you are unsure, it is better to include a connection than to omit it.

COMMON SENSE RULES:
- IAM roles (AWS) or IAP/Security (GCP) should be connected to any compute resource that accesses cloud resources.
- Databases and storage (RDS, DynamoDB, Athena, S3 for AWS; SQL, Bigtable, BigQuery, Storage for GCP) are likely to be connected to servers or functions unless the user specifies otherwise.
- For monitoring, use CloudWatch (AWS) or Google Observability/Stackdriver (GCP). For logging, use CloudTrail (AWS) or Logging (GCP). For CDN, use CloudFront (AWS) or Cloud CDN (GCP). For API Gateway, use API Gateway (AWS) or Endpoints (GCP). For message queue, use SQS (AWS) or Google Cloud Pub/Sub (GCP). For storage, use S3 (AWS) or Google Cloud Storage Bucket (GCS) for GCP. For IoT, use IoT Core (AWS) or fallback to a generic server for GCP. For any AWS-only element with no GCP equivalent, use a generic GCP or on-prem icon.
- If a server or function is described as accessing a database or storage, always include that connection.
- Do NOT include any elements from the other cloud provider.

Core Connection Patterns:
- EC2 typically connects to RDS, S3, VPC, CloudWatch, SQS, and requires IAM roles
- Lambda is usually triggered by API Gateway, S3, SQS and connects to RDS, DynamoDB, S3, and requires IAM roles
- ECS and Fargate are used to run containers and typically connect to RDS, S3, VPC, CloudWatch, and require IAM roles
- CloudTrail records API calls and events for all AWS resources and is often connected to CloudWatch and S3 for logging
- IoT Core ingests data from devices and typically sends data to Kinesis Data Streams, Lambda, or S3
- Kinesis Data Streams is used for real-time data ingestion and processing, often connected to Lambda, S3, Redshift, or Athena
- Athena is used to query data stored in S3
- CloudFront is a CDN that typically sits in front of S3, API Gateway, or ELB
- Compute Engine connects to Cloud SQL, Cloud Storage, Internet, VPC, and requires IAM roles
- RDS is typically accessed by EC2, Lambda, ECS, Fargate, API Gateway (via Lambda)
- DynamoDB is commonly accessed by Lambda, EC2, ECS, Fargate, API Gateway
- VPC houses EC2, RDS, ELB, ECS, Fargate, and optionally Lambda
- ELB routes traffic to EC2 instances, ECS/Fargate services, or sometimes Lambda
- API Gateway typically fronts Lambda, EC2, ECS/Fargate, or S3 static websites
- SQS is used to decouple services between EC2, Lambda, ECS, Fargate, API Gateway
- CloudWatch monitors all other components
- IAM roles are required for EC2, Lambda, ECS, Fargate, and other services to access AWS resources
- Route53 provides DNS services and can route to ELB, CloudFront, or S3
- SNS can trigger Lambda functions or send notifications to SQS queues
- CloudTrail is used for auditing and logging API activity across AWS services

Common Flow Patterns:
1. IoT Core → Kinesis Data Streams → Lambda/S3/Redshift/Athena
2. User/Internet → Route53 → CloudFront → API Gateway → Lambda/ECS/Fargate → RDS/DynamoDB/S3
3. User/Internet → Route53 → ELB → EC2/ECS/Fargate → RDS
4. Batch data → S3 → Lambda/Redshift/ECS/Fargate/Athena
5. Lambda/EC2/ECS/Fargate → SQS → Lambda/ECS/Fargate workers
6. All components → CloudWatch for monitoring
7. Services → IAM roles for permissions
8. SNS → Lambda/SQS/ECS/Fargate for event processing
9. CloudFront → S3/API Gateway for content delivery
10. CloudTrail → CloudWatch/S3 for logging and auditing
11. Athena → S3 for querying data

Description:
{description}

Available Nodes:
{json.dumps(nodes, indent=2)}

If the description doesn't explicitly specify connections, use the architectural patterns above to infer likely connections.
IMPORTANT: Respond with ONLY a JSON array of [source, target] pairs, without any markdown or explanation.

Example format:
[
  ["Internet", "API Gateway"],
  ["API Gateway", "Lambda"],
  ["Lambda", "RDS"],
  ["Lambda", "IAM Role"]
]
"""
            response = self.llm_service.llm.generate_content(prompt)
            response_text = self._clean_json_response(response.text)
            
            # Log the raw response for debugging
            logger.info(f"Raw LLM connections response: {response_text}")
            
            if not response_text:
                logger.warning("Empty response from LLM for connections")
                return []
            
            try:
                connections = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM connections response as JSON: {response_text}")
                return []
            
            if not isinstance(connections, list):
                logger.warning("LLM connections response is not a list")
                return []
            
            # Validate connections
            valid_connections = []
            node_labels = {node["label"] for node in nodes}
            
            for connection in connections:
                if not isinstance(connection, list) or len(connection) != 2:
                    logger.warning(f"Invalid connection format: {connection}")
                    continue
                    
                src, dst = connection
                if src in node_labels and dst in node_labels:
                    valid_connections.append([src, dst])
                else:
                    logger.warning(f"Invalid connection skipped: {src} -> {dst}")
            
            return valid_connections
            
        except Exception as e:
            logger.error(f"Error extracting connections: {str(e)}")
            return []

class AssistantService:
    """
    Service for handling AI assistant interactions.
    
    This service provides:
    1. Message processing
    2. Context management
    3. Conversation history tracking
    
    The service uses the LLM to generate responses while maintaining
    conversation context and history.
    """
    
    def __init__(self, llm_service: LLMService):
        """
        Initialize the assistant service.
        
        Args:
            llm_service: LLM service instance for generating responses
        """
        self.llm_service = llm_service
        self.history = []
    
    def process_message(self, message: str, context: Optional[Dict] = None) -> Dict:
        """
        Process a message and generate a response.
        
        Args:
            message: User's message
            context: Optional conversation context
            
        Returns:
            Dictionary containing:
            - response: Generated response text
            - context: Updated conversation context
            - history: Conversation history
        """
        try:
            # Add message to history
            self.history.append({"role": "user", "content": message})
            
            # Generate response
            response = self.llm_service.generate_response(self.history)
            
            # Add response to history
            self.history.append({"role": "assistant", "content": response})
            
            return {
                "response": response,
                "context": context or {},
                "history": self.history
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise
    
    def get_history(self) -> List[Dict]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.history 