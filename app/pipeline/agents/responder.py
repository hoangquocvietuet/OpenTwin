"""Responder Agent — generates the twin's response.

Uses the system/rewrite prompt from prompt.py, plus tone and content chunks
as contextual references. Includes critic feedback on retry attempts.
"""

from app.pipeline.state import PipelineState


def responder_agent(
    state: PipelineState,
    llm_client=None,
    llm_model: str | None = None,
    system_prompt: str = "",
    rewrite_prompt: str = "",
    session_factory=None,
    twin_slug: str = "",
) -> PipelineState:
    """Generate a response as the twin.

    For answer mode: uses system_prompt + tone chunks (style) + content chunks (context) + history
    For rewrite mode: uses rewrite_prompt + tone chunks (style only)
    """
    if not llm_client:
        state.draft_response = "LLM not configured."
        return state

    messages = []

    # System prompt based on mode
    if state.mode == "rewrite":
        messages.append({"role": "system", "content": rewrite_prompt})
    else:
        messages.append({"role": "system", "content": system_prompt})

    # Check retrieval quality — warn LLM if chunks are low relevance
    all_chunks = (state.tone_chunks or []) + (state.content_chunks or [])
    if all_chunks:
        avg_distance = sum(c.get("distance", 1.0) for c in all_chunks) / len(all_chunks)
        if avg_distance > 1.0:
            messages.append({
                "role": "system",
                "content": (
                    "The retrieved context below has very low relevance to the user's message. "
                    "Rely on your identity and the system prompt to answer. "
                    "Only use retrieved context if it is clearly relevant."
                ),
            })

    # Tone chunks — style reference
    if state.tone_chunks:
        tone_docs = "\n\n---\n\n".join(c.get("document", "") for c in state.tone_chunks if c.get("document"))
        if tone_docs:
            messages.append({
                "role": "system",
                "content": f"This is how you sound in this register (use for style only, NOT content):\n\n{tone_docs}",
            })

    # Content chunks — personal context (answer mode mainly, but available for both)
    if state.content_chunks:
        content_docs = "\n\n---\n\n".join(c.get("document", "") for c in state.content_chunks if c.get("document"))
        if content_docs:
            messages.append({
                "role": "system",
                "content": f"Related things you've said before (personal context):\n\n{content_docs}",
            })

    # Conversation history (answer mode only)
    if state.mode == "answer" and session_factory and twin_slug:
        try:
            from app.database import ChatMessage
            with session_factory() as session:
                recent_msgs = (
                    session.query(ChatMessage)
                    .filter_by(twin_slug=twin_slug)
                    .order_by(ChatMessage.id.desc())
                    .limit(10)
                    .all()
                )
                recent_msgs.reverse()
                for msg in recent_msgs:
                    messages.append({"role": msg.role, "content": msg.content})
        except Exception:
            pass

    # Critic feedback on retry
    if state.critic_feedback and state.retry_count > 0:
        messages.append({
            "role": "system",
            "content": f"Your previous attempt was rejected. Feedback: {state.critic_feedback}\nPlease try again following the feedback.",
        })

    # User message
    messages.append({"role": "user", "content": state.resolved_content or state.raw_input})

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=messages,
            stream=False,
            timeout=30,
        )
        state.draft_response = response.choices[0].message.content or ""
    except Exception:
        state.draft_response = "Sorry, I couldn't generate a response right now. Please try again."

    return state
