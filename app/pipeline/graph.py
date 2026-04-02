"""LangGraph pipeline definition and execution.

Wires the 5 agents into a StateGraph with conditional routing:
Intent → (Context if needed) → Retriever → Responder → Critic → (retry or done)
"""

from functools import partial

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


def _ensure_resolved_content(state: PipelineState) -> PipelineState:
    """Fill resolved_content from raw_input if context agent didn't set it."""
    if not state.resolved_content:
        state.resolved_content = state.raw_input
    return state


def build_pipeline(
    *,
    llm_client=None,
    llm_model: str = "",
    classifier_client=None,
    classifier_model: str = "",
    collection=None,
    system_prompt: str = "",
    rewrite_prompt: str = "",
    session_factory=None,
    twin_slug: str = "",
) -> StateGraph:
    """Build and compile the LangGraph pipeline with bound dependencies.

    Returns a compiled graph. Call .invoke(state) to run it.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("node_intent", partial(
        intent_agent, llm_client=classifier_client, llm_model=classifier_model,
    ))
    graph.add_node("node_context", partial(
        context_agent, llm_client=classifier_client, llm_model=classifier_model,
    ))
    graph.add_node("node_ensure_content", _ensure_resolved_content)
    graph.add_node("node_retriever", partial(
        retriever_agent, collection=collection,
    ))
    graph.add_node("node_responder", partial(
        responder_agent,
        llm_client=llm_client,
        llm_model=llm_model,
        system_prompt=system_prompt,
        rewrite_prompt=rewrite_prompt,
        session_factory=session_factory,
        twin_slug=twin_slug,
    ))
    graph.add_node("node_critic", partial(
        critic_agent, llm_client=classifier_client, llm_model=classifier_model,
    ))

    graph.set_entry_point("node_intent")
    graph.add_conditional_edges("node_intent", _route_after_intent, {
        "node_context": "node_context",
        "node_retriever": "node_ensure_content",
    })
    graph.add_edge("node_context", "node_ensure_content")
    graph.add_edge("node_ensure_content", "node_retriever")
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

    Builds the graph with bound dependencies, then invokes it.
    """
    compiled = build_pipeline(
        llm_client=llm_client,
        llm_model=llm_model,
        classifier_client=classifier_client,
        classifier_model=classifier_model,
        collection=collection,
        system_prompt=system_prompt,
        rewrite_prompt=rewrite_prompt,
        session_factory=session_factory,
        twin_slug=twin_slug,
    )

    state = PipelineState(raw_input=raw_input, mode=mode)
    result = compiled.invoke(state)

    # LangGraph returns a dict-like object; convert back to PipelineState
    if isinstance(result, PipelineState):
        return result
    return PipelineState(**{k: v for k, v in result.items() if k in PipelineState.__dataclass_fields__})
