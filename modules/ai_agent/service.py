"""Load history -> run graph -> save new messages."""
import logging
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.database import sqlite
from modules.ai_agent.graph import create_chat_model
from modules.ai_agent.supervisor import build_supervisor

logger = logging.getLogger(__name__)

graph = build_supervisor("groq")   # or "ollama", or graph.build_graph("ollama") for one agent

WORKERS = ("menu_agent", "complaint_agent", "offer_agent")
GREETINGS = {
    "hi", "hello", "hey", "yo",
    "good morning", "good afternoon", "good evening",
}
MAX_ATTEMPTS = 3

# Groq's gpt-oss models occasionally emit broken tool output: an empty
# generation, malformed JSON args, or a tool name with Harmony channel tokens
# appended (e.g. "transfer_to_complaint_agent<|channel|>commentary").
MODEL_OUTPUT_ERRORS = (
    "parsing failed", "could not be parsed", "failed_generation",
    "tool call validation", "not in request.tools",
    "invalid tool call", "invalid tool arguments", "<|channel|>",
)
RETRYABLE_TEXT = MODEL_OUTPUT_ERRORS + (
    "rate limit", "timed out", "timeout", "temporarily unavailable",
)
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Used only if the tool-calling model keeps failing. No tools are bound here,
# so Groq's tool-call parser cannot produce the empty-generation 400 error.
FALLBACK_SYSTEM = """You are Xander Restaurant's friendly support assistant. Answer briefly and accurately using only these facts:
- Menu (PKR): Zinger Burger Rs. 850; Chicken Biryani Rs. 450; Club Sandwich Rs. 700.
- Offer: Buy 2 Zinger Burgers get 1 Club Sandwich.
- Valid order IDs are exactly OR-1045, OR-1046 and OR-1047.
- A complaint needs the customer name, phone, order ID and the issue.
If an order ID is invalid or details are missing, ask the customer to correct them.
Never claim that a complaint was saved."""


def get_graph():
    """Return the compiled graph. Indirection keeps the graph patchable in tests."""
    return graph


def is_answer(message):
    # A real AI text reply: not a tool call, not the "Transferring back" hand-off
    return (message.type == "ai" and message.content and not message.tool_calls
            and not message.content.startswith("Transferring back"))


def visible_turns(messages):
    """Only user questions and final answers, without tool/hand-off noise."""
    shown = []
    has_answer = False
    for message in messages:
        if message.type == "human":
            item = {"role": "user", "content": message.content.strip()}
            has_answer = False
        elif is_answer(message) and not has_answer:
            item = {"role": "ai", "content": message.content.strip()}
            has_answer = True
        else:
            continue
        if item["content"] and item != (shown[-1] if shown else None):
            shown.append(item)
    return shown


def clean_history(messages):
    """Rebuild cross-turn history as plain turns.

    Feeding the raw tool-call trace back into the supervisor makes Groq's
    structured-output parser much more likely to fail with an empty generation.
    """
    return [
        HumanMessage(content=turn["content"]) if turn["role"] == "user"
        else AIMessage(content=turn["content"])
        for turn in visible_turns(messages)
    ]


def is_parse_failure(error):
    """True for any malformed model output (empty generation, bad tool name)."""
    body = getattr(error, "body", None) or {}
    if isinstance(body, dict) and "parse" in str(body.get("code", "")).lower():
        return True
    return any(fragment in str(error).lower() for fragment in MODEL_OUTPUT_ERRORS)


def is_retryable(error):
    """Only retry transient provider failures, never bad input or missing keys."""
    status = getattr(error, "status_code", None)
    if status in RETRYABLE_STATUS:
        return True
    return any(fragment in str(error).lower() for fragment in RETRYABLE_TEXT)


def invoke_graph(messages):
    """Invoke the graph, retrying brief Groq tool-call parsing hiccups."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return get_graph().invoke({"messages": messages})
        except Exception as error:
            if not is_retryable(error) or attempt == MAX_ATTEMPTS:
                raise
            logger.warning(
                "Provider call failed (attempt %s/%s); retrying: %s",
                attempt, MAX_ATTEMPTS, error,
            )
            time.sleep(0.5 * attempt)


def fallback_answer(history, query):
    """Answer without tools when the provider cannot parse a tool call."""
    model = create_chat_model("groq")
    reply = model.invoke([
        SystemMessage(content=FALLBACK_SYSTEM),
        *history,
        HumanMessage(content=query),
    ])
    return reply.content.strip()


def tidy_answer(text):
    """Keep raw tool dumps and repeated lines out of the user-facing reply."""
    kept = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
            continue
        if stripped and stripped in {k.strip() for k in kept}:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def chat(query, session_id):
    # Long replays make Groq's tool-call parser less stable; keep recent context.
    history = clean_history(sqlite.load_messages(session_id))[-10:]
    user_message = HumanMessage(content=query)

    # Greetings do not need the supervisor or any tool call. Answering them here
    # avoids the Groq parsing failure that can occur on hand-off tool calls.
    if query.strip().lower().rstrip("!.") in GREETINGS:
        answer = "Hello! How can I help you today?"
        sqlite.append_messages(session_id, [
            user_message, AIMessage(content=answer, name="supervisor"),
        ])
        return answer

    try:
        result = invoke_graph(history + [user_message])
    except Exception as error:
        logger.warning(
            "Graph failed (%s); parse=%s retryable=%s",
            type(error).__name__, is_parse_failure(error), is_retryable(error),
        )
        if not is_retryable(error):
            raise
        answer = tidy_answer(fallback_answer(history, query))
        sqlite.append_messages(session_id, [
            user_message, AIMessage(content=answer, name="supervisor"),
        ])
        return answer

    new_messages = result["messages"][len(history):]
    answer = None
    # The supervisor often adds a closing line after the worker's answer.
    # Prefer the worker's text answer, not the supervisor's handoff wording.
    for message in reversed(new_messages):
        if is_answer(message) and message.name in WORKERS:
            answer = message.content
            break

    if answer is None:
        worker_used = any(getattr(m, "name", None) in WORKERS for m in new_messages)
        if worker_used:
            # A worker was delegated to but returned no answer. The supervisor
            # tends to invent one in that case, so answer from known facts instead.
            logger.warning("Worker returned no answer; using plain answer")
            answer = tidy_answer(fallback_answer(history, query))
            sqlite.append_messages(session_id, [
                user_message, AIMessage(content=answer, name="supervisor"),
            ])
            return answer
        answer = result["messages"][-1].content

    answer = tidy_answer(answer)
    # Persist one clean turn per exchange. Raw tool/hand-off messages are never
    # needed again and are what caused the parser failures on later turns.
    sqlite.append_messages(session_id, [
        user_message, AIMessage(content=answer, name="supervisor"),
    ])
    return answer


def get_history(session_id):
    return visible_turns(sqlite.load_messages(session_id))
