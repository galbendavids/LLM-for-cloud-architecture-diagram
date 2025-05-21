# Gemini API Integration - Bonus Features

This is an enhanced version of the Gemini API integration that includes additional features for cloud architecture design and diagramming:

- Assistant-style chat for refining your architecture
- Conversation history management
- Background task processing
- Timestamp tracking
- Conversation ID management

## Setup

1. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1.5 Install Rust (If not already installed) - this is mandatory for requirements.txt install. It is also needed to add rust to your current shell:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

2. Create a virtual environment using UV (suggested: `-p python3.11`):
```bash
uv venv -p python3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Change directory to the project directory:
```bash
cd path/to/src
```

4. Install dependencies:
```bash
uv pip install -r requirements.txt
```

5. Run the bonus API:
```bash
python bonus/main.py
```

The service will be available at `http://localhost:8001`

---

## API Reference

### 1. **POST `/chat`**
- **Purpose:** Assistant-style chat for refining your cloud architecture.
- **Request:**
```json
{
  "messages": [
    { "role": "user", "content": "I want a serverless image processing app..." }
  ],
  "conversation_id": "optional-conv-id"
}
```
- **Response:**
```json
{
  "response": "Great! Do you want to use S3 for storage? ...",
  "conversation_id": "conv_1234567890",
  "timestamp": "2024-03-14T12:00:00.000Z"
}
```
- **Behavior:**
  - The assistant will ask clarifying questions, suggest best practices, and help you finalize your architecture.
  - When you say "ready", "done", or similar, the assistant will instruct you to use `/generate-diagram`.

### 2. **POST `/generate-diagram`** *(planned/placeholder)*
- **Purpose:** Generate a cloud architecture diagram from your finalized description.
- **Request:**
```json
{
  "description": "A user accesses an application via Route53 and CloudFront..."
}
```
- **Response:**
- Returns a PNG image of the generated diagram.
- **Note:** This endpoint is referenced by the assistant, and is intended for use after your architecture is finalized in chat.

### 3. **GET `/conversations/{conversation_id}`**
- Retrieve the full history of a specific conversation.
- Returns 404 if conversation not found.

### 4. **GET `/conversations`**
- List all available conversation IDs.

---

## Example Architecture Prompts

- "Build a serverless application using AWS Lambda functions for image processing, an S3 bucket to store uploaded images, and an API Gateway to trigger the Lambda functions. Use DynamoDB to store image metadata. Add IAM roles for security and CloudWatch for logging and metrics."
- "Design a containerized web application with a front-end React app and a back-end Node.js API. Use ECS Fargate for deployment, an Application Load Balancer for traffic distribution, and an RDS PostgreSQL database for data persistence. Include Route 53 for DNS and CloudTrail for auditing."
- "Create an IoT architecture that uses AWS IoT Core to receive sensor data, stores the data in DynamoDB, and triggers a Lambda function when a threshold is exceeded. Visualize alerts with Amazon SNS notifications. Enable CloudWatch for real-time monitoring."
- "Design a data analytics pipeline on AWS. Use Kinesis Data Streams to collect live data, process it using AWS Glue, store results in Amazon S3, and query the data using Amazon Athena. Enable CloudWatch for job performance monitoring and alerts."
- "Implement a scalable chatbot architecture using Amazon Lex for natural language understanding, AWS Lambda for backend logic, and DynamoDB for storing conversation history. Add CloudFront for content delivery and API Gateway for connecting external clients. Monitor with CloudWatch dashboards."

---

## How to Use the Assistant

1. **Start a conversation** with `/chat` and describe your architecture idea.
2. **Refine your design** as the assistant asks questions and suggests improvements.
3. **When ready**, say "ready", "done", or "generate diagram". The assistant will instruct you to use `/generate-diagram`.
4. **POST your finalized description** to `/generate-diagram` to receive your architecture diagram as a PNG.

---

## Troubleshooting & Tips
- If the assistant doesn't ask about a key requirement, mention it explicitly.
- If you want to start over, use a new conversation ID.
- If you encounter errors, check your API key and dependencies.
- The assistant is designed to help you clarify and finalize your architecture before diagram generation.

---

## Limitations & Considerations
- Conversation history is stored in memory (not persistent).
- The `/generate-diagram` endpoint is referenced by the assistant and planned for future implementation.
- Only AWS and GCP icons are supported for diagrams.
- The assistant may occasionally miss or infer connections; explicit user instructions are prioritized.

---

## Contributing & Testing
- Contributions are welcome! Please open issues or PRs for improvements.
- To run tests, use:
```bash
python -m unittest discover tests
```
- For local development, a mock LLM is available.
- Error handling and logging are implemented throughout.
