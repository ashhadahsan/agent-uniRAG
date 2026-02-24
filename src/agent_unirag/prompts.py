"""Prompt templates for Agent-UniRAG (from paper Figures 8, 9, 14)."""

import datetime


PLANNING_SYSTEM_PDF = """You will be utilizing the following tool to assist with answering questions:

search_engine - Document search (uploaded PDF / knowledge base):
{
  "tool_name": "search_engine",
  "tool_description": "Search the uploaded document to find text evidence for the provided search query.",
  "tool_note": "One clear query at a time. You can only use this tool.",
  "tool_input": [{"param_name": "search_query", "param_type": "string", "param_description": "A detailed search query for the knowledge base"}]
}

Your task is to solve the question using the search_engine tool. Follow these steps:

1. Reason step by step; use only one tool call per step then get the response and continue.
2. DO NOT assume facts; use the tool to get evidence from the document.
3. Provide a CLEAR and CONCISE final answer.

Format for each search:

### Thought: A short rationale for using the search_engine. (one sentence)
### Search Input: {"search_query": "your query here"}
### Observation: [WAITING] (do not generate; you will receive evidence)

When you have the final answer:

### Thought: I have the final answer
### Final Answer: your condensed answer to the main question

Let's begin with the question:"""

PLANNING_SYSTEM_WEB = """You will be utilizing the following tool to assist with answering questions:

web_search - Web search:
{
  "tool_name": "web_search",
  "tool_description": "Search the web for up-to-date or external information.",
  "tool_note": "One clear query at a time. You can only use this tool.",
  "tool_input": [{"param_name": "search_query", "param_type": "string", "param_description": "A detailed search query for the web"}]
}

Today's date: {{current_date}} (use this for date-aware queries).

When forming search queries: Do NOT add a year (e.g. 2023 or 2024) to the query unless the user's question explicitly asks about that year. For general factual questions (e.g. "longest river in the world"), use a neutral query like "longest river in the world" or "longest river world current"; the search will return up-to-date results without needing a year in the query.

Your task is to solve the question using the web_search tool. Follow these steps:

1. Reason step by step; use only one tool call per step then get the response and continue.
2. DO NOT assume facts; use the tool to get evidence from the web.
3. Provide a CLEAR and CONCISE final answer.

Format for each search:

### Thought: A short rationale for using the web_search. (one sentence)
### Web Search Input: {"search_query": "your query here"}
### Observation: [WAITING] (do not generate; you will receive evidence)

When you have the final answer:

### Thought: I have the final answer
### Final Answer: your condensed answer to the main question

Let's begin with the question:"""


def build_planning_prompt(
    question: str,
    trajectory: str,
    at_budget_limit: bool = False,
    conversation_context: str | None = None,
    retrieval_mode: str = "pdf",
    current_date: str | None = None,
) -> str:
    """Build the planning prompt. retrieval_mode: "pdf" = document only, "web" = web search only."""
    if current_date is None:
        current_date = datetime.date.today().isoformat()
    system = PLANNING_SYSTEM_WEB if retrieval_mode == "web" else PLANNING_SYSTEM_PDF
    system = system.replace("{{current_date}}", current_date)
    budget_note = ""
    if at_budget_limit:
        budget_note = "\n\nYou have reached the maximum number of searches. You must provide your final answer now based on the evidence gathered so far.\n\n"
    if conversation_context and conversation_context.strip():
        question_block = (
            "Previous questions and answers:\n"
            + conversation_context.strip()
            + "\n\nCurrent question: "
            + question
        )
    else:
        question_block = question
    return system + budget_note + "\n\n" + question_block + "\n\n" + trajectory


EVIDENCE_REFLECTOR_TASK = """### Task: Synthesize a condensed text evidence from given sources to support a search query.

### Sources:

{sources}

### Search Query: {search_query}

### Selection Guidelines:

1. Clarity: Evidence must be clear, concise.
2. Conciseness: Evidence must be presented in a succinct manner, condensed and AVOIDING unnecessary details.
3. Relevance: Evidence must directly correspond and relevant to the search query.
4. Source Integrity: Only use information from the provided sources, AVOIDING generated or unnecessary information.
5. If multiple part of a source is relevant to the search query, combine them into one element in the response list.

### Response MUST be in a JSON list as below:

[
  {{
    "evidence": "condensed text supporting the search query from a source",
    "source_id": "an identifier of the source text"
  }}
]

If no evidence is found, respond with the following json:

[
  {{
    "evidence": "No supporting evidence found.",
    "source_id": null
  }}
]"""


def build_evidence_reflector_prompt(sources: list[dict], search_query: str) -> str:
    """Build the Evidence Reflector prompt (Figure 8)."""
    import json

    sources_str = json.dumps(sources, indent=2)
    return EVIDENCE_REFLECTOR_TASK.format(
        sources=sources_str, search_query=search_query
    )


FINAL_ANSWER_TASK = """### Task: Given the question and a list of evidence, your task is to provide the final answer to the question based on the information within the evidence.

### Evidence: {evidence}

### Question: {question}

Provide the final answer below. Be clear, concise, and base your answer only on the evidence provided."""


def build_final_answer_prompt(evidence: str, question: str) -> str:
    """Build the final answer prompt (Figure 14)."""
    return FINAL_ANSWER_TASK.format(evidence=evidence, question=question)
