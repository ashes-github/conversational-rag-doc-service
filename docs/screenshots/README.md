# Application screenshots

Store submission screenshots in this directory and use these filenames so the main README can reference them consistently later:

1. `01-upload-and-index.png` — application after one or more PDFs have been indexed, with no private filenames or content visible.
2. [`02-grounded-answer.png`](02-grounded-answer.png) — captured; shows representative document questions and grounded conversational responses.
3. `03-no-context-guardrail.png` — a question that is not answered by the documents and the deterministic fallback response.
4. `04-api-docs.png` — FastAPI `/docs` showing the main endpoints.

Capture at a readable desktop width, crop unrelated browser chrome if appropriate, and review every image for API keys, local paths, personal data, and confidential document text before committing it.

For an optional demo video, show this sequence in under two minutes: start the application, upload a safe sample PDF, index it, ask a grounded question, ask a follow-up, demonstrate the no-context guardrail, and briefly show the backend request logs. Do not record `.env`, terminal history containing secrets, or provider dashboards.
