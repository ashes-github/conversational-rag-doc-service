# Application screenshots

Store submission screenshots in this directory and use these filenames so the main README can reference them consistently later:

1. [`01-upload-and-index.png`](01-upload-and-index.png) — captured; shows a PDF successfully indexed and its resulting chunk count.
2. [`02-grounded-answer.png`](02-grounded-answer.png) — captured; shows representative document questions and grounded conversational responses.
3. [`03-no-context-guardrail.png`](03-no-context-guardrail.png) — captured; shows questions unsupported by the document and the deterministic fallback response.
4. [`04-api-docs.png`](04-api-docs.png) — captured; shows FastAPI `/docs` and the LangServe chain endpoints.

For an optional demo video, show this sequence in under two minutes: start the application, upload a safe sample PDF, index it, ask a grounded question, ask a follow-up, demonstrate the no-context guardrail, and briefly show the backend request logs. Do not record `.env`, terminal history containing secrets, or provider dashboards.
