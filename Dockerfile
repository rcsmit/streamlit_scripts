FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements_handstandanalyzer.txt .
RUN pip install --no-cache-dir -r requirements_handstandanalyzer.txt

# Copy app
COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "handstand_analyzer.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]