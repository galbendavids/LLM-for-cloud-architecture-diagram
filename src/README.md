# Cloud Architecture Diagram Generator

This project generates cloud architecture diagrams from natural language descriptions using the `diagrams` library and an LLM (Google Gemini or mock LLM for local development).

## Setup

### 1. Clone the repository
```
git clone <your-repo-url>
cd API_cloud_architecture/src
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and fill in your API key:
```
cp .env.example .env
```
Edit `.env` and set your `GOOGLE_API_KEY`.

### 4. Run Locally
```
python main.py
```

### 5. Run with Docker
```
docker build -t cloud-arch-diagram .
docker run --env-file .env -v $(pwd)/tmp:/app/tmp cloud-arch-diagram
```

## Example Input/Output

**Input:**
```
A user accesses an application via Route53 and CloudFront. The request goes to an API Gateway, which triggers a Lambda function. The Lambda function writes to DynamoDB and stores files in S3. All compute resources use IAM roles. CloudWatch monitors all services.
```

**Output:**
- A PNG diagram in the `tmp/` directory, e.g. `diagram_<uuid>.png`.
- The diagram will show the described flow, with CloudWatch off to the side labeled "monitors all services".

## Considerations & Limitations
- Only AWS and GCP icons supported in my current implementation (see `diagram_tools.py` for full list).
- LLM may occasionally miss or infer connections; explicit user instructions are prioritized.
- Temporary files are stored in `tmp/` and cleaned up after use.
- For local development, a mock LLM is available (see `mock_llm.py`).
- Error handling and logging are implemented throughout.

## Bonus Features
- Assistant-style interface with context and memory. (i didnt had time to make it very good but it is nice to have)
- Modular, clean code structure.
- Mock LLM for local development.
- Logging and error handling.
- Unit test example in `tests/`.
- have also added a little more internsive tests (in the same test directory..)

## Temporary Files
- Diagrams are generated in `tmp/`.
- Temporary files are cleaned up after use.

## Testing
- See `tests/` for example unit tests.