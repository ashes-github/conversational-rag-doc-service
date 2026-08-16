"""Compatibility entrypoint for the conversational RAG backend."""

from backend.main import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
