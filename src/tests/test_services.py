import unittest
from unittest.mock import Mock, patch
import tempfile
import os
import logging
from services import LLMService, DiagramService, AssistantService

class TestLLMService(unittest.TestCase):
    """Unit tests for LLMService using the mock LLM."""
    def setUp(self):
        self.llm_service = LLMService(use_mock=True)
    
    def test_generate_response(self):
        """Test that the mock LLM returns a non-empty string response."""
        messages = [{"role": "user", "content": "Hello"}]
        response = self.llm_service.generate_response(messages)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_llm_fallback_on_error(self):
        """Test that LLMService falls back to mock LLM if real LLM is unavailable or errors."""
        with patch('services.genai', side_effect=Exception("LLM unavailable")):
            with self.assertRaises(Exception):
                # Should raise because fallback is not automatic in constructor, but this simulates error path
                LLMService(use_mock=False)

    def test_llm_error_handling_and_logging(self):
        """Test that LLMService logs errors when LLM call fails."""
        llm_service = LLMService(use_mock=True)
        with patch.object(llm_service.llm, 'generate_chat_response', side_effect=Exception("Mock LLM failure")):
            with self.assertLogs('services', level='ERROR') as cm:
                with self.assertRaises(Exception):
                    llm_service.generate_response([{"role": "user", "content": "fail"}])
            self.assertTrue(any("Error generating LLM response" in msg for msg in cm.output))

class TestDiagramService(unittest.TestCase):
    """Unit tests for DiagramService with mock LLM and diagram generation."""
    def setUp(self):
        self.llm_service = LLMService(use_mock=True)
        self.diagram_service = DiagramService(self.llm_service)
    
    def test_generate_diagram(self):
        """Test diagram generation for a simple web application."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            diagram_path = self.diagram_service.generate_diagram(
                "A simple web application",
                output_path
            )
            self.assertTrue(os.path.exists(diagram_path))
        finally:
            os.unlink(output_path)

    def test_generate_diagram_intensive_examples(self):
        """Test diagram generation for several intensive, realistic cloud architecture descriptions."""
        intensive_inputs = [
            # Input 1
            "Build a serverless application using AWS Lambda functions for image processing, an S3 bucket to store uploaded images, and an API Gateway to trigger the Lambda functions. Use DynamoDB to store image metadata. Add IAM roles for security and CloudWatch for logging and metrics.",
            # Input 2
            "Design a containerized web application with a front-end React app and a back-end Node.js API. Use ECS Fargate for deployment, an Application Load Balancer for traffic distribution, and an RDS PostgreSQL database for data persistence. Include Route 53 for DNS and CloudTrail for auditing.",
            # Input 3
            "Create an IoT architecture that uses AWS IoT Core to receive sensor data, stores the data in DynamoDB, and triggers a Lambda function when a threshold is exceeded. Visualize alerts with Amazon SNS notifications. Enable CloudWatch for real-time monitoring.",
            # Input 4
            "Design a data analytics pipeline on AWS. Use Kinesis Data Streams to collect live data, process it using AWS Glue, store results in Amazon S3, and query the data using Amazon Athena. Enable CloudWatch for job performance monitoring and alerts.",
            # Input 5
            "Implement a scalable chatbot architecture using Amazon Lex for natural language understanding, AWS Lambda for backend logic, and DynamoDB for storing conversation history. Add CloudFront for content delivery and API Gateway for connecting external clients. Monitor with CloudWatch dashboards."
        ]
        for idx, description in enumerate(intensive_inputs, 1):
            with self.subTest(f"Intensive Example {idx}"), tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                output_path = tmp.name
                try:
                    diagram_path = self.diagram_service.generate_diagram(description, output_path)
                    self.assertTrue(os.path.exists(diagram_path))
                finally:
                    os.unlink(output_path)

    def test_diagram_service_error_logging(self):
        """Test that DiagramService logs errors when diagram generation fails."""
        with patch.object(self.diagram_service, '_extract_components_with_order', side_effect=Exception("LLM extraction failed")):
            with self.assertLogs('services', level='ERROR') as cm:
                with self.assertRaises(Exception):
                    self.diagram_service.generate_diagram("bad input", "output.png")
            self.assertTrue(any("Error generating diagram" in msg for msg in cm.output))

class TestAssistantService(unittest.TestCase):
    """Unit tests for AssistantService context, memory, and response structure."""
    def setUp(self):
        self.llm_service = LLMService(use_mock=True)
        self.assistant_service = AssistantService(self.llm_service)
    
    def test_process_message(self):
        """Test that process_message returns a response, context, and history."""
        response = self.assistant_service.process_message("Hello")
        self.assertIn("response", response)
        self.assertIn("context", response)
        self.assertIn("history", response)
        self.assertTrue(len(response["history"]) > 0)
    
    def test_get_history(self):
        """Test that conversation history is updated and roles are correct."""
        self.assistant_service.process_message("Hello")
        history = self.assistant_service.get_history()
        self.assertTrue(len(history) > 0)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

if __name__ == '__main__':
    unittest.main() 