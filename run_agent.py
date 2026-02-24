#!/usr/bin/env python3
"""Run the Agent-UniRAG graph with a sample question."""

import asyncio
import os

from langchain_openai import ChatOpenAI

from agent_unirag import create_agent_unirag_graph
from agent_unirag.retriever import create_mock_retriever_for_demo


async def run() -> None:
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    retriever = create_mock_retriever_for_demo()
    graph = create_agent_unirag_graph(
        llm=llm,
        retriever=retriever,
        max_searches=5,
        top_k=8,
    )

    question = "What highway was renamed in honor of Tim Russert?"
    initial_state = {
        "question": question,
        "trajectory": "",
        "gathered_evidence": [],
        "retrieved_sources": [],
        "step_count": 0,
        "final_answer": None,
    }

    result = await graph.ainvoke(initial_state)
    print("Question:", question)
    print("Final answer:", result.get("final_answer", "(none)"))
    print("Steps:", result.get("step_count", 0))
    print("Evidence gathered:", len(result.get("gathered_evidence") or []))


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
