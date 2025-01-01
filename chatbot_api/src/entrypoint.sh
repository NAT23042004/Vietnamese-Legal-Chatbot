#!/bin/bash

# Run any setup steps or pre-processing tasks here
echo "Starting Vietnamese Legal RAG FastAPI service..."

# Start the main application
uvicorn main:app --host localhost --port 8000