"""Pipeline state shared across all agents."""

from dataclasses import dataclass, field


@dataclass
class PipelineState:
    """Shared state that flows through the LangGraph pipeline.

    All agents read from and write to this state.
    """
    # Input
    raw_input: str
    mode: str  # "answer" | "rewrite"

    # Intent Agent output
    intent: str | None = None
    tone: str | None = None
    needs_context: bool = False
    context_source: str | None = None  # "url", "clipboard", None
    context_url: str | None = None

    # Context Agent output
    resolved_content: str | None = None

    # Retriever Agent output
    tone_chunks: list[dict] = field(default_factory=list)
    content_chunks: list[dict] = field(default_factory=list)

    # Responder Agent output
    draft_response: str | None = None

    # Critic Agent output
    approved: bool = False
    critic_feedback: str | None = None
    retry_count: int = 0
