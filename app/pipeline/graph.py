"""LangGraph pipeline definition and execution.

Wires the 5 agents into a StateGraph with conditional routing:
Intent → (Context if needed) → Retriever → Responder → Critic → (retry or done)
"""

from langgraph.graph import StateGraph, END

from app.pipeline.state import PipelineState
from app.pipeline.agents.intent import intent_agent
from app.pipeline.agents.context import context_agent
from app.pipeline.agents.retriever import retriever_agent
from app.pipeline.agents.responder import responder_agent
from app.pipeline.agents.critic import critic_agent

MAX_RETRIES = 2


def _route_after_intent(state: PipelineState) -> str:
    """Route after intent: to context agent if needed, else to retriever."""
    if state.needs_context:
        return "node_context"
    return "node_retriever"


def _route_after_critic(state: PipelineState) -> str:
    """Route after critic: retry responder or finish."""
    if not state.approved and state.retry_count < MAX_RETRIES:
        return "node_responder"
    return END


def build_pipeline() -> StateGraph:
    """Build and compile the LangGraph pipeline.

    Returns a compiled graph. Call .invoke(state) to run it.
    Note: Agent dependencies (llm_client, collection, etc.) are bound
    at runtime via run_pipeline(), not at build time.

    Node names are prefixed with "node_" to avoid collision with
    PipelineState field names (e.g. state.intent, state.context, etc.).
    """
    graph = StateGraph(PipelineState)

    # Nodes are added as string names; actual functions are bound in run_pipeline
    graph.add_node("node_intent", lambda state: state)  # placeholder
    graph.add_node("node_context", lambda state: state)
    graph.add_node("node_retriever", lambda state: state)
    graph.add_node("node_responder", lambda state: state)
    graph.add_node("node_critic", lambda state: state)

    graph.set_entry_point("node_intent")
    graph.add_conditional_edges("node_intent", _route_after_intent, {
        "node_context": "node_context",
        "node_retriever": "node_retriever",
    })
    graph.add_edge("node_context", "node_retriever")
    graph.add_edge("node_retriever", "node_responder")
    graph.add_edge("node_responder", "node_critic")
    graph.add_conditional_edges("node_critic", _route_after_critic, {
        "node_responder": "node_responder",
        END: END,
    })

    return graph.compile()


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

    # Run agents sequentially (simpler than LangGraph node binding for now)
    # This gives us the same pipeline flow with cleaner dependency injection.

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
