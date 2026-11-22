# Xander AI Agent

A FastAPI + LangGraph backend for Xander Restaurant's AI support assistant.

## What it does

- Routes each question to the right worker: **menu**, **complaint**, or **offer**.
- Answers menu and price questions from real tool data (prices in PKR).
- Handles offers and discounts.
- Collects complaint details, validates the order ID, and saves the complaint to SQLite.
- Keeps per-session history and returns clean replies without raw tool calls or handoff text.
- Retries transient provider errors and falls back to a plain answer if tool calling fails.

## Stack

Python 3.11+, FastAPI, LangChain, LangGraph, Groq, SQLite, and `uv`.

## Setup

```bash
git clone https://github.com/muteeb25/xanders-aiagent.git
cd xanders-aiagent
cp .env.example .env
```

Set your key in `.env`:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Run:

```bash
uv sync
uv run fastapi dev
```

The API runs at `http://127.0.0.1:8000`; interactive docs are at `/docs`.

## Endpoints

- `POST /ai/chat` — body `{"query": "...", "session_id": "..."}` → `{"session_id": "...", "answer": "..."}`
- `GET /ai/history?session_id=...` — visible user/assistant turns
- `GET /` — service status message

Example:

```bash
curl -X POST http://127.0.0.1:8000/ai/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"any discounts today?","session_id":"demo-session-1"}'
```

## Demo data

- Menu: Zinger Burger Rs. 850, Chicken Biryani Rs. 450, Club Sandwich Rs. 700
- Offer: Buy 2 Zinger Burgers get 1 Club Sandwich
- Valid order IDs: `OR-1045`, `OR-1046`, `OR-1047`

## Security

Only `.env.example` is committed. Never commit real API keys or local databases.
