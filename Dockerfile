FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y build-essential gcc g++ && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY pyproject.toml ./
COPY uv.lock ./

# Install build dependencies and download model
RUN uv sync --no-dev
RUN uv run python -c "from transformers import pipeline; pipeline('text-generation', model='thiborose/SmolLM2-FT-SQL')"

# Copy app code
COPY app.py ./

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]