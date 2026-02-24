"""Build and compile the Agent-UniRAG LangGraph workflow."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agent_unirag.nodes import (
    evidence_reflector_node,
    final_answer_node,
    planning_node,
    search_node,
    web_search_node,
)
from agent_unirag.retriever import RetrieverProtocol
from agent_unirag.state import AgentUniRAGState


def route_after_planning(state: AgentUniRAGState) -> str:
    """Conditional edge: planning -> search, web_search, or final_answer."""
    return state.get("next_node") or "final_answer"


def create_agent_unirag_graph(
    llm: BaseChatModel,
    retriever: RetrieverProtocol,
    max_searches: int | None = None,
    top_k: int = 8,
) -> Any:
    """Build and compile the Agent-UniRAG StateGraph.

    Args:
        llm: Chat model for planning, evidence reflector, and final answer.
        retriever: Retriever for search node (query, top_k) -> list of sources.
        max_searches: Max search steps (computing budget). None = no limit.
        top_k: Number of documents to retrieve per search.

    Returns:
        Compiled LangGraph executable.
    """
    workflow = StateGraph(AgentUniRAGState)

    async def planning(s: AgentUniRAGState) -> AgentUniRAGState:
        return await planning_node(s, llm=llm, max_searches=max_searches)

    async def search(s: AgentUniRAGState) -> AgentUniRAGState:
        return await search_node(s, retriever=retriever, top_k=top_k)

    async def web_search(s: AgentUniRAGState) -> AgentUniRAGState:
        return await web_search_node(s)

    async def evidence_reflector(s: AgentUniRAGState) -> AgentUniRAGState:
        return await evidence_reflector_node(s, llm=llm)

    async def final_answer(s: AgentUniRAGState) -> AgentUniRAGState:
        return await final_answer_node(s, llm=llm)

    workflow.add_node("planning", planning)
    workflow.add_node("search", search)
    workflow.add_node("web_search", web_search)
    workflow.add_node("evidence_reflector", evidence_reflector)
    workflow.add_node("final_answer", final_answer)

    workflow.add_edge(START, "planning")
    workflow.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "search": "search",
            "web_search": "web_search",
            "final_answer": "final_answer",
        },
    )
    workflow.add_edge("search", "evidence_reflector")
    workflow.add_edge("web_search", "evidence_reflector")
    workflow.add_edge("evidence_reflector", "planning")
    workflow.add_edge("final_answer", END)

    return workflow.compile()
