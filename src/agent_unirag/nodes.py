"""Agent-UniRAG graph nodes: planning, search, web_search, evidence_reflector, final_answer."""

import asyncio
import json
import os
import re
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agent_unirag.prompts import (
    build_evidence_reflector_prompt,
    build_final_answer_prompt,
    build_planning_prompt,
)
from agent_unirag.retriever import RetrieverProtocol
from agent_unirag.state import AgentUniRAGState

# Patterns for parsing planning output (Figure 9 format)
FINAL_ANSWER_THOUGHT = re.compile(
    r"###\s*Thought:\s*I have the final answer",
    re.IGNORECASE | re.DOTALL,
)
FINAL_ANSWER_BLOCK = re.compile(
    r"###\s*(?:Action\s*-\s*)?Final\s*Answer:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
SEARCH_INPUT_BLOCK = re.compile(
    r"###\s*(?:Action\s*-\s*)?Search\s*Input:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
WEB_SEARCH_INPUT_BLOCK = re.compile(
    r"###\s*(?:Action\s*-\s*)?Web\s*Search\s*Input:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_query_from_block(raw: str) -> str:
    raw = re.sub(
        r"\s*###\s*Observation:.*$", "", raw, flags=re.IGNORECASE | re.DOTALL
    ).strip()
    if "```" in raw:
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if code_match:
            raw = code_match.group(1).strip()
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            return obj.get("search_query", raw) if isinstance(obj, dict) else raw
        except json.JSONDecodeError:
            return raw
    return raw


def _parse_planning_response(
    text: str, retrieval_mode: str = "pdf"
) -> tuple[str | None, str | None, bool, bool]:
    """Parse LLM planning output. Returns (search_query, final_answer, is_final, use_web_search).

    When retrieval_mode is "pdf", only Search Input is parsed. When "web", only Web Search Input.
    """
    is_final = bool(FINAL_ANSWER_THOUGHT.search(text))
    if is_final:
        match = FINAL_ANSWER_BLOCK.search(text)
        answer = match.group(1).strip() if match else None
        return (None, answer, True, False)
    if retrieval_mode == "web":
        web_match = WEB_SEARCH_INPUT_BLOCK.search(text)
        if web_match:
            query = _extract_query_from_block(web_match.group(1).strip())
            return (query, None, False, True)
    else:
        match = SEARCH_INPUT_BLOCK.search(text)
        if match:
            query = _extract_query_from_block(match.group(1).strip())
            return (query, None, False, False)
    return (None, None, True, False)


def _parse_evidence_reflector_response(text: str) -> list[str]:
    """Parse Evidence Reflector JSON output into list of evidence strings."""
    evidence_list: list[str] = []
    try:
        # Try to find JSON array in the response
        start = text.find("[")
        if start == -1:
            return ["No supporting evidence found."]
        end = text.rfind("]") + 1
        if end <= start:
            return ["No supporting evidence found."]
        arr = json.loads(text[start:end])
        for item in arr:
            if isinstance(item, dict) and "evidence" in item:
                evidence_list.append(item["evidence"].strip())
            elif isinstance(item, str):
                evidence_list.append(item.strip())
    except (json.JSONDecodeError, TypeError):
        evidence_list = (
            [text.strip()] if text.strip() else ["No supporting evidence found."]
        )
    return evidence_list if evidence_list else ["No supporting evidence found."]


async def planning_node(
    state: AgentUniRAGState,
    *,
    llm: BaseChatModel,
    max_searches: int | None = None,
) -> AgentUniRAGState:
    """Planning module: decide next action (search, web_search, or final answer)."""
    question = state["question"]
    conversation_context = state.get("conversation_context") or ""
    trajectory = state.get("trajectory") or ""
    step_count = state.get("step_count") or 0
    retrieval_mode = state.get("retrieval_mode") or "pdf"
    at_budget = max_searches is not None and step_count >= max_searches

    prompt = build_planning_prompt(
        question,
        trajectory,
        at_budget_limit=at_budget,
        conversation_context=conversation_context or None,
        retrieval_mode=retrieval_mode,
    )
    messages = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = content[0].get("text", "") if content else ""
    content = content.strip()

    search_query, _, is_final, use_web_search = _parse_planning_response(
        content, retrieval_mode
    )

    if at_budget and not is_final:
        is_final = True

    new_trajectory = trajectory + "\n\n" + content
    if not is_final and search_query:
        new_trajectory += "\n\n### Observation:\n[WAITING]"

    if is_final:
        next_node: Literal["search", "web_search", "final_answer"] = "final_answer"
    elif use_web_search:
        next_node = "web_search"
    else:
        next_node = "search"

    updates: AgentUniRAGState = {
        "trajectory": new_trajectory,
        "next_node": next_node,
    }
    if not is_final and search_query:
        updates["current_search_query"] = search_query

    return updates


async def search_node(
    state: AgentUniRAGState,
    *,
    retriever: RetrieverProtocol,
    top_k: int = 8,
) -> AgentUniRAGState:
    """Search tool: retrieve top_k sources from the document knowledge base."""
    query = state.get("current_search_query") or ""
    sources = await asyncio.to_thread(retriever.retrieve, query, top_k)
    return {"retrieved_sources": sources}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _openai_web_search_sync(
    query: str, model: str, api_key: str | None
) -> list[dict[str, Any]]:
    """Call OpenAI Responses API with web_search tool; return list of {content, source_id} for evidence_reflector."""
    if not api_key or not query.strip():
        return []
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=f"Search the web for relevant, factual information: {query.strip()}. Return concise facts and cite sources.",
            tools=[{"type": "web_search"}],
        )
        sources: list[dict[str, Any]] = []
        text = ""
        annotations: list[Any] = []
        for item in _get(response, "output") or []:
            if _get(item, "type") != "message" or _get(item, "role") != "assistant":
                continue
            for block in _get(item, "content") or []:
                if _get(block, "type") == "output_text":
                    text = _get(block, "text") or ""
                    annotations = _get(block, "annotations") or []
                    break
            if text:
                break
        if not text:
            return []
        for i, ann in enumerate(annotations):
            if _get(ann, "type") == "url_citation":
                start = _get(ann, "start_index", 0) or 0
                end = _get(ann, "end_index", len(text)) or len(text)
                url = _get(ann, "url") or ""
                title = _get(ann, "title") or ""
                snippet = (
                    text[start:end].strip()
                    if 0 <= start < end <= len(text)
                    else text[:500]
                )
                if snippet:
                    sources.append(
                        {
                            "content": snippet,
                            "source_id": url or str(i),
                            "url": url,
                            "title": title,
                        }
                    )
        if not sources:
            sources.append(
                {"content": text[:2000].strip(), "source_id": "openai_web_search"}
            )
        return sources
    except Exception:
        return []


async def web_search_node(
    state: AgentUniRAGState,
    *,
    model: str | None = None,
) -> AgentUniRAGState:
    """Web search module using OpenAI built-in web search (Responses API). Returns retrieved_sources for evidence_reflector."""
    query = state.get("current_search_query") or ""
    api_key = os.environ.get("OPENAI_API_KEY")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    sources = await asyncio.to_thread(_openai_web_search_sync, query, model, api_key)
    return {"retrieved_sources": sources}


async def evidence_reflector_node(
    state: AgentUniRAGState,
    *,
    llm: BaseChatModel,
) -> AgentUniRAGState:
    """Evidence Reflector: extract condensed evidence from retrieved sources."""
    sources = state.get("retrieved_sources") or []
    query = state.get("current_search_query") or ""
    gathered = list(state.get("gathered_evidence") or [])

    if not sources:
        evidence_parts = ["No supporting evidence found."]
        gathered.append(evidence_parts[0])
    else:
        prompt = build_evidence_reflector_prompt(sources, query)
        messages = [HumanMessage(content=prompt)]
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = content[0].get("text", "") if content else ""
        content = content.strip()
        evidence_parts = _parse_evidence_reflector_response(content)
        for part in evidence_parts:
            if part and part != "No supporting evidence found.":
                gathered.append(part)
        if not evidence_parts or all(
            p == "No supporting evidence found." for p in evidence_parts
        ):
            gathered.append("No supporting evidence found.")
            evidence_parts = ["No supporting evidence found."]

    step_count = (state.get("step_count") or 0) + 1
    trajectory = state.get("trajectory") or ""
    last_evidence = (
        " ".join(evidence_parts) if evidence_parts else "No supporting evidence found."
    )
    if "[WAITING]" in trajectory:
        trajectory = trajectory.replace("[WAITING]", last_evidence, 1)
    else:
        trajectory = trajectory + "\n\n### Observation:\n" + last_evidence

    return {
        "gathered_evidence": gathered,
        "step_count": step_count,
        "trajectory": trajectory,
    }


async def final_answer_node(
    state: AgentUniRAGState,
    *,
    llm: BaseChatModel,
) -> AgentUniRAGState:
    """Produce final answer from question and gathered evidence."""
    question = state.get("question") or ""
    gathered = state.get("gathered_evidence") or []
    existing = state.get("final_answer")

    if existing:
        return {}

    evidence_str = "\n\n".join(gathered) if gathered else "No evidence was gathered."
    prompt = build_final_answer_prompt(evidence_str, question)
    messages = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = content[0].get("text", "") if content else ""
    answer = content.strip() if content else ""

    return {"final_answer": answer}
