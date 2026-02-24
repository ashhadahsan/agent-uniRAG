"""State schema for the Agent-UniRAG LangGraph workflow."""

from typing import Any, Literal

from typing_extensions import TypedDict


class AgentUniRAGState(TypedDict, total=False):
    """State passed between nodes in the Agent-UniRAG graph.

    Attributes:
        question: The user's input question.
        conversation_context: Optional previous Q&A for multi-turn (e.g. "Q: ...\nA: ...").
        trajectory: Planning conversation: question + Thought/Search Input/Observation turns.
        gathered_evidence: Evidence strings appended by the Evidence Reflector.
        retrieved_sources: Raw documents from the last search (input to reflector).
        current_search_query: Last search query from the planning node.
        final_answer: The produced long-form answer.
        step_count: Number of search steps so far (computing budget).
        next_node: Routing flag set by planning: "search", "web_search", or "final_answer".
        max_searches: Maximum number of search steps (None = no limit).
        top_k: Number of documents to retrieve per search.
        retrieval_mode: "pdf" = document search only, "web" = web search only.
    """

    question: str
    conversation_context: str
    retrieval_mode: Literal["pdf", "web"]
    trajectory: str
    gathered_evidence: list[str]
    retrieved_sources: list[dict[str, Any]]
    current_search_query: str
    final_answer: str | None
    step_count: int
    next_node: Literal["search", "web_search", "final_answer"]
    max_searches: int | None
    top_k: int
