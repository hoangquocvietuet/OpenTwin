"""Shared chat logic used by both the API endpoint and the Gradio UI.

Single source of truth for: validation, retrieval, prompt building,
LLM call, error handling, and DB persistence.
"""

import json
from dataclasses import dataclass

import openai

from app.database import ChatMessage
from app.retrieval import retrieve_chunks, format_few_shot_examples

MAX_MESSAGE_LENGTH = 10_000


@dataclass
class ChatResult:
    content: str
    retrieval_metadata: dict | None = None
    error: bool = False


def chat(
    content: str,
    collection,
    session_factory,
    twin_slug: str,
    system_prompt: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
) -> ChatResult:
    """Process a chat message through the full RAG pipeline.

    1. Validate input
    2. Retrieve similar chunks from ChromaDB
    3. Build prompt with system prompt + few-shot + history
    4. Call LLM
    5. Save to DB
    6. Return result
    """
    # 1. Validate
    content = content.strip()
    if not content:
        return ChatResult(content="Please enter a message.", error=True)

    # Truncate silently if too long
    if len(content) > MAX_MESSAGE_LENGTH:
        content = content[:MAX_MESSAGE_LENGTH]

    # 2. Check if twin has data
    if collection.count() == 0:
        return ChatResult(
            content="I don't have enough context to answer that authentically. Import data first.",
            error=True,
        )

    # 3. Retrieve
    retrieved = retrieve_chunks(collection, content, n_results=5)

    if not retrieved:
        return ChatResult(
            content="I don't have enough context to answer that authentically.",
            retrieval_metadata={"chunks": 0, "avg_similarity": 0},
            error=True,
        )

    # 4. Build messages
    few_shot = format_few_shot_examples(retrieved, max_examples=3)
    avg_distance = sum(r["distance"] for r in retrieved) / len(retrieved)

    # Get recent chat history
    with session_factory() as session:
        recent_msgs = (
            session.query(ChatMessage)
            .filter_by(twin_slug=twin_slug)
            .order_by(ChatMessage.id.desc())
            .limit(10)
            .all()
        )
        recent_msgs.reverse()

    messages = [{"role": "system", "content": system_prompt}]

    if few_shot:
        messages.append({
            "role": "system",
            "content": f"Here are examples of how you actually talk:\n\n{few_shot}",
        })

    for msg in recent_msgs:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": content})

    # 5. LLM call
    try:
        client = openai.OpenAI(base_url=llm_base_url, api_key=llm_api_key)
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            stream=False,
            timeout=30,
        )
        assistant_content = response.choices[0].message.content or ""
    except openai.APIConnectionError:
        return ChatResult(
            content="Could not reach the LLM. Check that Ollama is running.",
            error=True,
        )
    except openai.APITimeoutError:
        return ChatResult(
            content="LLM took too long to respond. Try again.",
            error=True,
        )
    except openai.RateLimitError:
        return ChatResult(
            content="Rate limited. Wait a moment and try again.",
            error=True,
        )
    except json.JSONDecodeError:
        return ChatResult(
            content="Unexpected response format from LLM.",
            error=True,
        )

    if not assistant_content.strip():
        return ChatResult(
            content="No response generated. Try rephrasing.",
            error=True,
        )

    retrieval_meta = {
        "chunks": len(retrieved),
        "avg_similarity": round(1 - avg_distance, 3),
    }

    # 6. Save to DB
    try:
        with session_factory() as session:
            session.add(ChatMessage(
                twin_slug=twin_slug,
                role="user",
                content=content,
            ))
            session.add(ChatMessage(
                twin_slug=twin_slug,
                role="assistant",
                content=assistant_content,
                retrieval_metadata=retrieval_meta,
                tokens_used=response.usage.total_tokens if response.usage else None,
            ))
            session.commit()
    except Exception:
        pass  # Response still shown, history not saved

    return ChatResult(
        content=assistant_content,
        retrieval_metadata=retrieval_meta,
    )
