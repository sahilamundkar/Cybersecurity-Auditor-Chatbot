#!/bin/bash
set -e

# Wait for Ollama to start
until curl -s -f -o /dev/null "http://ollama:11434/api/tags"; do
    echo "Waiting for Ollama to start..."
    sleep 5
done

# Pull the required models
ollama pull llama2
ollama pull llama3-7b-8192

echo "Ollama initialization complete."