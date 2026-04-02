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
    retrieved_chunks: list[dict] | None = None
    error: bool = False


def _extract_utterances_by_speaker(document: str, speaker: str) -> list[str]:
    """Extract utterances from a chunk document for a single speaker.

    Our stored chunk document format is one line per turn: "Author: text".
    We only keep lines attributed to `speaker` to avoid role-mixing in few-shot.
    """
    out: list[str] = []
    prefix = f"{speaker}:"
    for raw_line in (document or "").splitlines():
        line = raw_line.strip()
        if not line.startswith(prefix):
            continue
        text = line[len(prefix):].strip()
        if text:
            out.append(text)
    return out


def chat(
    content: str,
    collection,
    session_factory,
    twin_slug: str,
    twin_name: str,
    system_prompt: str,
    rewrite_prompt: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    mode: str = "answer",
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

    # Normalize mode (API / UI)
    mode = (mode or "answer").strip().lower()
    if mode in ("chat", "answer"):
        mode = "answer"

    # Lightweight UX: allow "rewrite:" prefix without changing UI.
    lowered = content.lower()
    if lowered.startswith("rewrite:"):
        mode = "rewrite"
        content = content[len("rewrite:"):].strip()
    elif lowered.startswith("rewrite "):
        mode = "rewrite"
        content = content[len("rewrite"):].strip()

    # Truncate silently if too long
    if len(content) > MAX_MESSAGE_LENGTH:
        content = content[:MAX_MESSAGE_LENGTH]

    # 2. Check if twin has data
    if collection.count() == 0:
        return ChatResult(
            content="This twin doesn't have any imported data yet. Import chat data first, then ask again.",
            error=True,
        )

    # 3. Retrieve
    retrieved = retrieve_chunks(collection, content, n_results=5)

    # 4. Build messages
    few_shot = format_few_shot_examples(retrieved, max_examples=3)
    avg_distance = (
        (sum(r["distance"] for r in retrieved) / len(retrieved)) if retrieved else 1.0
    )

    if mode == "rewrite":
        # Copy / rewrite mode: rephrase user's text in the twin's voice.
        # Extract twin's utterances from retrieved chunks as style examples.
        style_lines: list[str] = []
        for chunk in retrieved[:5]:
            for u in _extract_utterances_by_speaker(chunk.get("document", ""), twin_name):
                style_lines.append(u)
        seen: set[str] = set()
        style_lines = [s for s in style_lines if not (s in seen or seen.add(s))]
        style_block = "\n- " + "\n- ".join(style_lines[:12]) if style_lines else ""

        messages = [
            {"role": "system", "content": rewrite_prompt},
            {
                "role": "system",
                "content": f"Examples of how you type (use for style only, NOT for content):{style_block}".strip(),
            },
            {"role": "user", "content": content},
        ]
    else:
        # Answer mode — normal chat
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

        if not retrieved:
            messages.append({
                "role": "system",
                "content": (
                    "No relevant retrieved context was found for the user's message. "
                    "Do your best with general knowledge and the system prompt. "
                    "If the question depends on personal history or tone, ask 1-2 specific follow-up questions."
                ),
            })

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
    except openai.NotFoundError:
        return ChatResult(
            content=f"Model '{llm_model}' not found. Check the model name in Settings.",
            error=True,
        )
    except openai.APIConnectionError:
        return ChatResult(
            content="Could not reach the LLM. Check the base URL in Settings.",
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
                content=content if mode != "rewrite" else f"[copy] {content}",
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
        retrieved_chunks=retrieved,
    )
