FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-frontend.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements-frontend.txt

COPY RAG_client.py ./

EXPOSE 8501

CMD ["streamlit", "run", "RAG_client.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
