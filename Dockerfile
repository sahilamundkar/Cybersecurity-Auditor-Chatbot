# Use Ubuntu as the base image
FROM ubuntu:22.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl https://ollama.ai/install.sh | sh

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Expose the ports for Streamlit and Ollama
EXPOSE 8501 11434

# Create a script to start both Ollama and your Streamlit app
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 10\n\
ollama pull llama2\n\
ollama pull llama3\n\
streamlit run app.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Command to run the start script
CMD ["/app/start.sh"]