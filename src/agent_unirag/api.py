"""FastAPI application for Agent-UniRAG."""

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from agent_unirag import create_agent_unirag_graph
from agent_unirag.retriever import ChromaPDFRetriever


def check_required_env() -> None:
    """Exit the process if required env vars are missing. Call at startup so the server does not start without them."""
    missing = []
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        missing.append("OPENAI_API_KEY")
    if missing:
        print(
            "Error: Required environment variable(s) not set: "
            + ", ".join(missing)
            + ".",
            "Set them (e.g. export OPENAI_API_KEY=...) and restart the server.",
            sep="\n",
            file=sys.stderr,
        )
        sys.exit(1)


check_required_env()

_graph: Any = None
_retriever: ChromaPDFRetriever | None = None


def get_graph() -> Any:
    if _graph is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized; check OPENAI_API_KEY and server startup.",
        )
    return _graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _retriever
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=api_key,
        )
        embeddings = OpenAIEmbeddings(
            model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=api_key,
        )
        _retriever = ChromaPDFRetriever(embedding=embeddings)
        _graph = create_agent_unirag_graph(
            llm=llm,
            retriever=_retriever,
            max_searches=int(os.environ.get("AGENT_MAX_SEARCHES", "5")),
            top_k=int(os.environ.get("AGENT_TOP_K", "8")),
        )
    else:
        _retriever = None
    yield
    _graph = None
    _retriever = None


app = FastAPI(
    title="Agent-UniRAG API",
    description="Unified RAG agent for single-hop and multi-hop QA (LangGraph)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConversationTurn(BaseModel):
    """One Q&A turn for context."""

    question: str = Field(..., min_length=1)
    answer: str = Field(default="")


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=1, description="Question to answer.")
    mode: Literal["pdf", "web"] = Field(
        default="pdf",
        description="Retrieval mode: 'pdf' = search uploaded document only, 'web' = web search only.",
    )
    conversation: list[ConversationTurn] = Field(
        default_factory=list,
        description="Previous Q&A turns for follow-up context (last 5 used).",
    )


class AskResponse(BaseModel):
    """Response for POST /ask."""

    question: str
    final_answer: str | None
    step_count: int
    gathered_evidence: list[str] = Field(default_factory=list)
    trajectory: str = ""


@app.get("/health")
def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness: ok only if agent is initialized (OPENAI_API_KEY set)."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return {"status": "ready"}


@app.get("/documents")
def get_documents() -> dict[str, int]:
    """Return the number of chunks currently in the retriever (from uploaded PDF)."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return {"chunk_count": _retriever.chunk_count()}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str | int]:
    """Upload a single PDF; it replaces the current document corpus. Questions will run over this PDF."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if file.content_type and "pdf" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        raw = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}") from e
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        _retriever.replace_documents(raw)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid or unreadable PDF: {e}"
        ) from e
    return {
        "status": "ok",
        "filename": file.filename or "document.pdf",
        "chunk_count": _retriever.chunk_count(),
    }


MAX_CONVERSATION_TURNS = 5


def _format_conversation_context(turns: list[ConversationTurn]) -> str:
    """Format previous Q&A for the planning prompt. Uses at most last MAX_CONVERSATION_TURNS."""
    if not turns:
        return ""
    recent = turns[-MAX_CONVERSATION_TURNS:]
    parts = []
    for t in recent:
        parts.append(f"Q: {t.question}\nA: {t.answer or '(no answer)'}")
    return "\n\n".join(parts)


def _initial_state(
    question: str,
    conversation: list[ConversationTurn],
    mode: Literal["pdf", "web"] = "pdf",
) -> dict[str, Any]:
    return {
        "question": question,
        "retrieval_mode": mode,
        "conversation_context": _format_conversation_context(conversation),
        "trajectory": "",
        "gathered_evidence": [],
        "retrieved_sources": [],
        "step_count": 0,
        "final_answer": None,
    }


def _state_for_sse(state: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable subset of state for SSE."""
    sources = state.get("retrieved_sources") or []
    return {
        "question": state.get("question"),
        "trajectory": state.get("trajectory"),
        "gathered_evidence": state.get("gathered_evidence") or [],
        "retrieved_sources": sources,
        "step_count": state.get("step_count", 0),
        "final_answer": state.get("final_answer"),
        "current_search_query": state.get("current_search_query"),
    }


_NODE_LABELS: dict[str, str] = {
    "planning": "Planning",
    "search": "Search",
    "web_search": "Web search",
    "evidence_reflector": "Evidence reflector",
    "final_answer": "Final answer",
    "done": "Done",
}

_NODE_WIDTH = 200
_PROCESS_NODE_HEIGHT = 56
_DECISION_NODE_SIZE = 72
_NODE_VERTICAL_GAP = 24


def _build_flow_graph(invocation_order: list[str]) -> dict[str, Any]:
    """Build React Flow-compatible nodes and edges from execution order (xyflow format)."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    y = 0

    for i, node_name in enumerate(invocation_order):
        step_id = f"step-{i}"
        label = _NODE_LABELS.get(node_name, node_name)

        nodes.append(
            {
                "id": step_id,
                "type": "stepNode",
                "position": {"x": 0, "y": y},
                "data": {"label": label, "stepIndex": i},
            }
        )
        y += _PROCESS_NODE_HEIGHT + _NODE_VERTICAL_GAP

        next_name = invocation_order[i + 1] if i + 1 < len(invocation_order) else None
        is_planning = node_name == "planning"
        has_branch = next_name in ("search", "web_search", "final_answer")

        if is_planning and has_branch and next_name:
            decision_id = f"decision-{i}"
            decision_label = _NODE_LABELS.get(next_name, next_name)
            nodes.append(
                {
                    "id": decision_id,
                    "type": "decisionNode",
                    "position": {"x": 0, "y": y},
                    "data": {"label": decision_label},
                }
            )
            y += _DECISION_NODE_SIZE + _NODE_VERTICAL_GAP
            edges.append(
                {
                    "id": f"e-{step_id}-{decision_id}",
                    "source": step_id,
                    "target": decision_id,
                }
            )
            edges.append(
                {
                    "id": f"e-{decision_id}-step-{i + 1}",
                    "source": decision_id,
                    "target": f"step-{i + 1}",
                }
            )
        elif next_name:
            edges.append(
                {
                    "id": f"e-{step_id}-step-{i + 1}",
                    "source": step_id,
                    "target": f"step-{i + 1}",
                }
            )

    return {"nodes": nodes, "edges": edges}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Run the Agent-UniRAG graph for one question (non-blocking). Optional conversation for follow-ups."""
    graph = get_graph()
    initial_state = _initial_state(req.question, req.conversation, req.mode)
    result = await graph.ainvoke(initial_state)
    return AskResponse(
        question=req.question,
        final_answer=result.get("final_answer"),
        step_count=result.get("step_count", 0),
        gathered_evidence=result.get("gathered_evidence") or [],
        trajectory=result.get("trajectory") or "",
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Run the agent and stream progress via Server-Sent Events (SSE). Optional conversation for follow-ups."""
    graph = get_graph()
    initial_state = _initial_state(req.question, req.conversation, req.mode)

    def sse_message(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def event_generator():
        accumulated: dict[str, Any] = dict(initial_state)
        invocation_order: list[str] = []
        try:
            async for chunk in graph.astream(
                initial_state,
                stream_mode="updates",
            ):
                if not isinstance(chunk, dict):
                    continue
                for node_name, state_update in chunk.items():
                    invocation_order.append(node_name)
                    accumulated.update(state_update)
                    payload = _state_for_sse(accumulated)
                    yield sse_message(node_name, {"node": node_name, "state": payload})
        except Exception as e:
            yield sse_message("error", {"error": str(e)})
            return
        done_payload = _state_for_sse(accumulated)
        done_payload["flow_graph"] = _build_flow_graph(invocation_order)
        yield sse_message("done", done_payload)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
