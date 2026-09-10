FROM python:3.11-slim

WORKDIR /app

# Dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY earnings_rag/ ./earnings_rag/
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "earnings_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]