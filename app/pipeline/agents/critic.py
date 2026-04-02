"""Critic Agent — reviews the Responder's draft before sending.

Checks style match, tone consistency, mode compliance (rewrite vs answer),
and appropriate response length. Can reject with feedback for retry.
"""

import json

from app.pipeline.state import PipelineState

_SYSTEM_PROMPT = """You are a quality reviewer for a digital twin chatbot. Review the draft response.

Given:
- The original user input
- The mode (answer or rewrite)
- The detected tone
- The draft response
- Style reference chunks (how the twin actually types)

Check:
1. STYLE: Does the draft sound like the twin? (casual, lowercase, slang vs formal)
2. TONE: Does the tone match? Casual input → casual output, formal → formal.
3. MODE COMPLIANCE:
   - If mode=rewrite: Did it REPHRASE (not answer)? Statements must stay statements, questions stay questions.
   - If mode=answer: Is it on-topic? No hallucination?
4. LENGTH: Appropriate for the input energy? Short question → short answer.

Return JSON:
- "approved": true if the response is acceptable, false if it needs retry
- "feedback": if not approved, explain what's wrong and how to fix it. Be specific.

Return ONLY valid JSON."""


def critic_agent(
    state: PipelineState,
    llm_client=None,
    llm_model: str | None = None,
) -> PipelineState:
    """Review the draft response for quality.

    If rejected, sets critic_feedback and increments retry_count.
    """
    if not llm_client:
        state.approved = True
        return state

    # Build review context
    style_ref = ""
    if state.tone_chunks:
        style_ref = "\n---\n".join(c.get("document", "") for c in state.tone_chunks[:2])

    review_input = (
        f"Mode: {state.mode}\n"
        f"Detected tone: {state.tone}\n"
        f"Original input: {state.raw_input[:1000]}\n"
        f"Draft response: {state.draft_response[:1000]}\n"
    )
    if style_ref:
        review_input += f"\nStyle reference (how twin types):\n{style_ref[:500]}"

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": review_input},
            ],
            timeout=15,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)
        approved = bool(parsed.get("approved", True))
        feedback = parsed.get("feedback", "")

        if approved:
            state.approved = True
        else:
            state.approved = False
            state.critic_feedback = feedback
            state.retry_count += 1
    except (json.JSONDecodeError, Exception):
        # Fail-safe: reject on parse failure so bad output doesn't slip through
        state.approved = False
        state.critic_feedback = "Critic review failed (unparseable response). Retrying."
        state.retry_count += 1

    return state
