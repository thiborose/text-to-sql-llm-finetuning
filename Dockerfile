FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY pyproject.toml ./
COPY uv.lock ./

# Install build tools for numpy and other scientific packages
RUN apt-get update && apt-get install -y build-essential gcc g++ && rm -rf /var/lib/apt/lists/*

RUN uv sync --no-dev

# Download the model at build time to avoid slow startup
RUN uv run python -c "from transformers import pipeline; pipeline('text-generation', model='thiborose/SmolLM2-FT-SQL')"

# Copy app code
COPY app.py ./
COPY .env ./

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]