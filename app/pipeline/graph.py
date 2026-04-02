"""Pipeline execution.

Runs the 5 agents sequentially with conditional routing:
Intent → (Context if needed) → Retriever → Responder → Critic → (retry or done)
"""

from app.pipeline.state import PipelineState
from app.pipeline.agents.intent import intent_agent
from app.pipeline.agents.context import context_agent
from app.pipeline.agents.retriever import retriever_agent
from app.pipeline.agents.responder import responder_agent
from app.pipeline.agents.critic import critic_agent

MAX_RETRIES = 2


def run_pipeline(
    raw_input: str,
    mode: str,
    collection,
    llm_client,
    llm_model: str,
    classifier_client,
    classifier_model: str,
    system_prompt: str,
    rewrite_prompt: str,
    session_factory=None,
    twin_slug: str = "",
) -> PipelineState:
    """Run the full pipeline with all dependencies injected.

    This is the main entry point. It creates the state, binds agent functions
    with their dependencies, and executes the graph.
    """
    state = PipelineState(raw_input=raw_input, mode=mode)

    # 1. Intent
    state = intent_agent(state, llm_client=classifier_client, llm_model=classifier_model)

    # 2. Context (conditional)
    if state.needs_context:
        state = context_agent(state, llm_client=classifier_client, llm_model=classifier_model)

    # Ensure resolved_content is set
    if not state.resolved_content:
        state.resolved_content = state.raw_input

    # 3. Retriever
    state = retriever_agent(state, collection=collection)

    # 4. Responder + Critic loop (max retries)
    for _ in range(MAX_RETRIES + 1):
        state = responder_agent(
            state,
            llm_client=llm_client,
            llm_model=llm_model,
            system_prompt=system_prompt,
            rewrite_prompt=rewrite_prompt,
            session_factory=session_factory,
            twin_slug=twin_slug,
        )

        state = critic_agent(
            state,
            llm_client=classifier_client,
            llm_model=classifier_model,
        )

        if state.approved:
            break

    return state
