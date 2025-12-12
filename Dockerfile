FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_handstandanalyzer.txt
RUN pip install --no-cache-dir -r requirements_handstandanalyzer.txt

# Copy app files
COPY . .

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "handstand_analyzer.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]