"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Background,
  Controls,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EXAMPLE_QUESTIONS: { category: string; questions: string[] }[] = [
  {
    category: "Single-hop",
    questions: [
      "What is the capital of France?",
      "Who wrote Romeo and Juliet?",
      "When was the first iPhone released?",
    ],
  },
  {
    category: "Multi-hop",
    questions: [
      "What highway was renamed in honor of Tim Russert?",
      "Which country has the largest population and what is its capital?",
      "Who directed the highest-grossing film of 2023 and what was the film?",
    ],
  },
  {
    category: "Geography",
    questions: [
      "What is the longest river in the world?",
      "How many countries border Germany?",
    ],
  },
  {
    category: "History & people",
    questions: [
      "Who was the first person to walk on the moon?",
      "In which year did World War II end?",
    ],
  },
  {
    category: "Science & tech",
    questions: [
      "What does DNA stand for?",
      "Who invented the World Wide Web?",
    ],
  },
];

type AskResponse = {
  question: string;
  final_answer: string | null;
  step_count: number;
  gathered_evidence: string[];
  trajectory: string;
};

type RetrievedSource = { content?: string; url?: string; title?: string; source_id?: string };

type StreamState = {
  node?: string;
  state?: {
    question?: string;
    trajectory?: string;
    gathered_evidence?: string[];
    retrieved_sources?: RetrievedSource[];
    step_count?: number;
    final_answer?: string | null;
    current_search_query?: string;
  };
};

type StepNodeData = { label: string; summary?: string; stepIndex: number };
type DecisionNodeData = { label: string };

const NODE_WIDTH = 200;
const NODE_VERTICAL_GAP = 24;
const PROCESS_NODE_HEIGHT = 56;
const DECISION_NODE_SIZE = 72;

function formatNodeName(node: string): string {
  const names: Record<string, string> = {
    planning: "Planning",
    search: "Search",
    web_search: "Web search",
    evidence_reflector: "Evidence reflector",
    final_answer: "Final answer",
    done: "Done",
  };
  return names[node] ?? node;
}

function stepSummary(s: StreamState, nodeType?: string): string | undefined {
  const isAnswerStep = nodeType === "final_answer" || nodeType === "done";
  if (isAnswerStep) {
    const ans = s.state?.final_answer?.trim();
    if (ans) return ans.length > 50 ? ans.slice(0, 50) + "..." : ans;
  }
  const q = s.state?.current_search_query ? formatSearchQuery(s.state.current_search_query) : "";
  if (q.trim()) return q.length > 50 ? q.slice(0, 50) + "..." : q;
  const n = s.state?.retrieved_sources?.length ?? 0;
  if (n > 0) return `Retrieved (${n})`;
  const e = s.state?.gathered_evidence?.length ?? 0;
  if (e > 0) return `Evidence (${e})`;
  const ans = s.state?.final_answer?.trim();
  if (ans) return ans.length > 50 ? ans.slice(0, 50) + "..." : ans;
  return undefined;
}

type FlowNode = Node<StepNodeData> | Node<DecisionNodeData>;

function stepsToFlow(steps: StreamState[]): { nodes: FlowNode[]; edges: Edge[] } {
  const nodes: FlowNode[] = [];
  const edges: Edge[] = [];
  let y = 0;

  for (let i = 0; i < steps.length; i++) {
    const s = steps[i];
    const stepId = `step-${i}`;
    const label = formatNodeName(s.node ?? "");
    const summary = stepSummary(s, s.node);

    nodes.push({
      id: stepId,
      type: "stepNode",
      position: { x: 0, y },
      data: { label, summary, stepIndex: i },
    });
    y += PROCESS_NODE_HEIGHT + NODE_VERTICAL_GAP;

    const next = steps[i + 1];
    const nextType = next?.node;
    const isPlanning = s.node === "planning";
    const hasBranch = nextType === "search" || nextType === "web_search" || nextType === "final_answer";

    if (isPlanning && hasBranch && next) {
      const decisionId = `decision-${i}`;
      nodes.push({
        id: decisionId,
        type: "decisionNode",
        position: { x: 0, y },
        data: { label: formatNodeName(nextType) },
      });
      y += DECISION_NODE_SIZE + NODE_VERTICAL_GAP;
      edges.push({ id: `e-${stepId}-${decisionId}`, source: stepId, target: decisionId });
      edges.push({ id: `e-${decisionId}-step-${i + 1}`, source: decisionId, target: `step-${i + 1}` });
    } else if (next) {
      edges.push({ id: `e-${stepId}-step-${i + 1}`, source: stepId, target: `step-${i + 1}` });
    }
  }

  return { nodes, edges };
}

function StepNode({ data }: NodeProps<Node<StepNodeData>>) {
  return (
    <div
      style={{
        width: NODE_WIDTH,
        padding: "0.6rem 0.75rem",
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
      }}
    >
      <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
        {data.label}
      </div>
      {data.summary && (
        <div style={{ fontSize: "0.8rem", color: "var(--text)", marginTop: "0.35rem", lineHeight: 1.35, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {data.summary}
        </div>
      )}
    </div>
  );
}

function DecisionNode({ data }: NodeProps<Node<DecisionNodeData>>) {
  const size = DECISION_NODE_SIZE;
  return (
    <div
      style={{
        width: size,
        height: size,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--surface)",
        border: "2px solid var(--accent)",
        clipPath: "polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)",
        boxSizing: "border-box",
      }}
    >
      <span
        style={{
          fontSize: "0.65rem",
          fontWeight: 700,
          color: "var(--accent)",
          textAlign: "center",
          maxWidth: size * 0.7,
          lineHeight: 1.15,
          padding: 4,
        }}
      >
        {data.label}
      </span>
    </div>
  );
}

const nodeTypes = { stepNode: StepNode, decisionNode: DecisionNode };

function formatSearchQuery(raw: string): string {
  const s = raw.trim();
  if (!s) return s;
  const jsonMatch = s.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (jsonMatch) {
    try {
      const obj = JSON.parse(jsonMatch[1].trim());
      return typeof obj.search_query === "string" ? obj.search_query : s;
    } catch {
      return s;
    }
  }
  return s.replace(/\s*###\s*Observation:.*$/i, "").trim();
}

/** Strip markdown links [text](url) to plain text (keeps text only). */
function stripMarkdownLinks(text: string): string {
  if (!text?.trim()) return text ?? "";
  return text.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").trim();
}

function sourceDisplayTitle(src: RetrievedSource): string {
  if (src.title?.trim()) return src.title.trim();
  const content = stripMarkdownLinks(src.content ?? "");
  if (content.length > 0) return content.length > 80 ? content.slice(0, 80) + "..." : content;
  if (src.url) {
    try {
      const host = new URL(src.url).hostname.replace(/^www\./, "");
      return host;
    } catch {
      return src.url;
    }
  }
  return "Source";
}

function formatTrajectoryBlocks(text: string): ReactNode {
  if (!text?.trim()) return null;
  const raw = text.trim();
  const blocks = raw.split(/\s*###\s+/).filter(Boolean);
  const result: React.ReactNode[] = [];
  blocks.forEach((block, i) => {
    const firstNewline = block.indexOf("\n");
    const firstLine = firstNewline >= 0 ? block.slice(0, firstNewline) : block;
    const rest = firstNewline >= 0 ? block.slice(firstNewline + 1).trim() : "";
    const colonIdx = firstLine.indexOf(":");
    const label = colonIdx >= 0 ? firstLine.slice(0, colonIdx).trim() : firstLine.trim();
    const firstLineBody = colonIdx >= 0 ? firstLine.slice(colonIdx + 1).trim() : "";
    const body = rest ? (firstLineBody ? `${firstLineBody}\n${rest}` : rest) : firstLineBody;
        if (!label && !body) return;
        result.push(
          <div
            key={i}
            style={{
              marginBottom: i < blocks.length - 1 ? "0.75rem" : 0,
              paddingBottom: i < blocks.length - 1 ? "0.75rem" : 0,
              borderBottom: i < blocks.length - 1 ? "1px solid var(--border)" : "none",
            }}
          >
            {label ? (
              <div
                style={{
                  fontWeight: 600,
                  fontSize: "0.75rem",
                  color: "var(--accent)",
                  textTransform: "uppercase",
                  letterSpacing: "0.03em",
                  marginBottom: "0.25rem",
                }}
              >
                {label}
              </div>
            ) : null}
            <div style={{ fontSize: "0.875rem", lineHeight: 1.5, whiteSpace: "pre-wrap", color: "var(--text)" }}>
              {body || "\u00a0"}
            </div>
          </div>
        );
      });
  if (result.length === 0) return <span style={{ whiteSpace: "pre-wrap" }}>{text}</span>;
  return <>{result}</>;
}

type ConversationTurn = { question: string; answer: string };

export default function Home() {
  const [backendReady, setBackendReady] = useState<boolean | null>(null);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<"pdf" | "web">("pdf");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamSteps, setStreamSteps] = useState<StreamState[]>([]);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [conversationHistory, setConversationHistory] = useState<ConversationTurn[]>([]);
  const [chunkCount, setChunkCount] = useState<number | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [stepsView, setStepsView] = useState<"list" | "flow">("list");
  const [expandedRetrievedSteps, setExpandedRetrievedSteps] = useState<Set<number>>(new Set());
  const [flowGraphFromBackend, setFlowGraphFromBackend] = useState<{ nodes: FlowNode[]; edges: Edge[] } | null>(null);
  const flowData = useMemo(() => {
    if (flowGraphFromBackend?.nodes?.length) {
      const nodes = flowGraphFromBackend.nodes.map((n) => {
        if (n.type === "stepNode" && n.data && "stepIndex" in n.data && typeof n.data.stepIndex === "number") {
          const step = streamSteps[n.data.stepIndex];
          const summary = step ? stepSummary(step, step.node) : undefined;
          return { ...n, data: { ...n.data, summary } };
        }
        return n;
      });
      return { nodes, edges: flowGraphFromBackend.edges };
    }
    return stepsToFlow(streamSteps);
  }, [flowGraphFromBackend, streamSteps]);

  const fetchChunkCount = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      if (res.ok) {
        const data = (await res.json()) as { chunk_count: number };
        setChunkCount(data.chunk_count);
      } else {
        setChunkCount(null);
      }
    } catch {
      setChunkCount(null);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_URL}/ready`)
      .then((res) => {
        if (!cancelled) setBackendReady(res.ok);
      })
      .catch(() => {
        if (!cancelled) setBackendReady(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (backendReady !== true) return;
    fetchChunkCount();
  }, [backendReady, fetchChunkCount]);

  const handleUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setUploadError(null);
      setUploadLoading(true);
      try {
        const form = new FormData();
        form.append("file", file);
        const res = await fetch(`${API_URL}/documents`, {
          method: "POST",
          body: form,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error((err as { detail?: string }).detail || "Upload failed");
        }
        const data = (await res.json()) as { chunk_count: number };
        setChunkCount(data.chunk_count);
        setConversationHistory([]);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setUploadLoading(false);
        e.target.value = "";
      }
    },
    []
  );

  const reset = useCallback(() => {
    setStreamSteps([]);
    setResult(null);
    setError(null);
    setFlowGraphFromBackend(null);
    setExpandedRetrievedSteps(new Set());
  }, []);

  const handleAsk = useCallback(async () => {
    const q = question.trim();
    if (!q) return;
    reset();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          mode,
          conversation: conversationHistory.map((t) => ({ question: t.question, answer: t.answer })),
        }),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");
      const dec = new TextDecoder();
      let buf = "";
      const steps: StreamState[] = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n\n");
        buf = lines.pop() || "";
        for (const block of lines) {
          const eventMatch = block.match(/event:\s*(\w+)/);
          const dataMatch = block.match(/data:\s*(\{[\s\S]*\})/);
          if (eventMatch && dataMatch) {
            try {
              const data = JSON.parse(dataMatch[1]) as {
                node?: string;
                state?: StreamState["state"];
                flow_graph?: { nodes: FlowNode[]; edges: Edge[] };
                error?: string;
                final_answer?: string | null;
              };
              if (data.error) {
                setError(data.error);
                setLoading(false);
                return;
              }
              const eventName = eventMatch[1];
              if (eventName === "done" && data.flow_graph) {
                setFlowGraphFromBackend(data.flow_graph);
              }
              const statePayload = data.state ?? (eventName === "done" ? data : undefined);
              steps.push({
                node: data.node || eventName,
                state: statePayload,
              });
              setStreamSteps([...steps]);
            } catch {
              // skip malformed chunk
            }
          }
        }
      }
      const last = steps[steps.length - 1];
      if (last?.state) {
        const finalAnswer = last.state.final_answer ?? null;
        setResult({
          question: q,
          final_answer: finalAnswer,
          step_count: last.state.step_count ?? 0,
          gathered_evidence: last.state.gathered_evidence ?? [],
          trajectory: last.state.trajectory ?? "",
        });
        setConversationHistory((prev) => [...prev, { question: q, answer: finalAnswer ?? "" }]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [question, reset, conversationHistory]);

  if (backendReady === null) {
    return (
      <main style={{ minHeight: "100vh", maxWidth: 720, margin: "0 auto", padding: "2rem 1rem", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: "1rem" }}>
        <p style={{ color: "var(--muted)", fontSize: "0.95rem" }}>Checking backend...</p>
      </main>
    );
  }

  if (backendReady === false) {
    return (
      <main style={{ minHeight: "100vh", maxWidth: 720, margin: "0 auto", padding: "2rem 1rem" }}>
        <h1 style={{ fontWeight: 600, fontSize: "1.75rem", marginBottom: "0.5rem", letterSpacing: "-0.02em" }}>
          Agent-UniRAG
        </h1>
        <div
          style={{
            padding: "1.5rem",
            background: "rgba(248,113,113,0.1)",
            border: "1px solid var(--error)",
            borderRadius: 8,
            color: "var(--error)",
            marginTop: "1rem",
          }}
        >
          <p style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Agent not ready</p>
          <p style={{ fontSize: "0.9rem", lineHeight: 1.5, margin: 0 }}>
            The backend could not start the agent. Set <code style={{ background: "var(--surface)", padding: "0.2rem 0.4rem", borderRadius: 4 }}>OPENAI_API_KEY</code> in the server environment and restart the API. If using Docker, pass the key via env or an <code style={{ background: "var(--surface)", padding: "0.2rem 0.4rem", borderRadius: 4 }}>.env</code> file.
          </p>
        </div>
      </main>
    );
  }

  return (
    <main
      style={{
        minHeight: "100vh",
        maxWidth: 720,
        margin: "0 auto",
        padding: "2rem 1rem",
      }}
    >
      <h1
        style={{
          fontWeight: 600,
          fontSize: "1.75rem",
          marginBottom: "0.25rem",
          letterSpacing: "-0.02em",
        }}
      >
        Agent-UniRAG
        <a
          href="https://arxiv.org/abs/2505.22571"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Agent-UniRAG research paper (arXiv)"
          title="Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems"
          style={{
            fontSize: "0.65rem",
            fontWeight: 500,
            marginLeft: "0.5rem",
            color: "var(--muted)",
            textDecoration: "none",
            verticalAlign: "middle",
          }}
        >
          (paper)
        </a>
      </h1>
      <p style={{ color: "var(--muted)", fontSize: "0.95rem", marginBottom: "1rem" }}>
        Unified RAG agent for single-hop and multi-hop QA. Ask a question; the agent plans, searches, and reflects on evidence. Based on the research paper by the Agent-UniRAG authors.
      </p>

      <section
        style={{
          marginBottom: "1.5rem",
          padding: "1rem 1.25rem",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 8,
        }}
      >
        <h2 style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.75rem", color: "var(--muted)" }}>
          Mode
        </h2>
        <div style={{ display: "flex", alignItems: "center", gap: "1.25rem", flexWrap: "wrap" }}>
          <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer" }}>
            <input
              type="radio"
              name="retrieval-mode"
              checked={mode === "pdf"}
              onChange={() => setMode("pdf")}
              disabled={loading}
            />
            <span style={{ fontSize: "0.9rem", color: "var(--text)" }}>PDF (search your document)</span>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer" }}>
            <input
              type="radio"
              name="retrieval-mode"
              checked={mode === "web"}
              onChange={() => setMode("web")}
              disabled={loading}
            />
            <span style={{ fontSize: "0.9rem", color: "var(--text)" }}>Web search</span>
          </label>
        </div>
      </section>

      {mode === "pdf" && (
        <section style={{ marginBottom: "1.5rem" }}>
          <h2 style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.5rem", color: "var(--muted)" }}>
            Knowledge base (PDF)
          </h2>
          <p style={{ fontSize: "0.875rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
            Upload a PDF to search over. One file at a time; uploading replaces the previous document.
          </p>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
            <label
              style={{
                display: "inline-block",
                padding: "0.5rem 1rem",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: "0.875rem",
                cursor: uploadLoading ? "not-allowed" : "pointer",
              }}
            >
              <input
                type="file"
                accept="application/pdf"
                onChange={handleUpload}
                disabled={uploadLoading}
                style={{ display: "none" }}
              />
              {uploadLoading ? "Uploading..." : "Choose PDF"}
            </label>
            {chunkCount !== null && (
              <span style={{ fontSize: "0.875rem", color: "var(--muted)" }}>
                {chunkCount} chunk{chunkCount !== 1 ? "s" : ""} in index
              </span>
            )}
          </div>
          {uploadError && (
            <p style={{ marginTop: "0.5rem", fontSize: "0.875rem", color: "var(--error)" }}>
              {uploadError}
            </p>
          )}
        </section>
      )}

      {conversationHistory.length > 0 && (
        <section style={{ marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
            <h2 style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--muted)" }}>
              Conversation
            </h2>
            <button
              type="button"
              onClick={() => setConversationHistory([])}
              disabled={loading}
              style={{
                padding: "0.35rem 0.75rem",
                fontSize: "0.8rem",
                background: "transparent",
                border: "1px solid var(--border)",
                borderRadius: 6,
                color: "var(--muted)",
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              Clear
            </button>
          </div>
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {conversationHistory.map((turn, i) => (
              <li
                key={i}
                style={{
                  marginBottom: "0.75rem",
                  padding: "0.75rem 1rem",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                }}
              >
                <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--accent)", marginBottom: "0.25rem" }}>
                  Q
                </div>
                <div style={{ fontSize: "0.9rem", marginBottom: "0.5rem" }}>{turn.question}</div>
                <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--accent)", marginBottom: "0.25rem" }}>
                  A
                </div>
                <div style={{ fontSize: "0.9rem", color: "var(--text)", whiteSpace: "pre-wrap" }}>
                  {turn.answer || "(no answer)"}
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.5rem", color: "var(--muted)" }}>
          Example questions
        </h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {EXAMPLE_QUESTIONS.map(({ category, questions }) => (
            <div key={category}>
              <span
                style={{
                  display: "block",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  color: "var(--accent)",
                  marginBottom: "0.35rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
                }}
              >
                {category}
              </span>
              <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                {questions.map((q) => (
                  <li key={q}>
                    <button
                      type="button"
                      onClick={() => setQuestion(q)}
                      disabled={loading || (mode === "pdf" && uploadLoading)}
                      style={{
                        padding: "0.4rem 0.75rem",
                        fontSize: "0.875rem",
                        background: "var(--surface)",
                        border: "1px solid var(--border)",
                        borderRadius: 6,
                        color: "var(--text)",
                        textAlign: "left",
                        cursor: loading ? "not-allowed" : "pointer",
                      }}
                    >
                      {q}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
          marginBottom: "1.5rem",
        }}
      >
        <textarea
          placeholder={
            mode === "pdf" && uploadLoading
              ? "Uploading and processing document..."
              : conversationHistory.length > 0
                ? "Ask a follow-up or a new question..."
                : "Type or pick an example above, then click Ask"
          }
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading || (mode === "pdf" && uploadLoading)}
          rows={3}
          style={{
            width: "100%",
            padding: "0.75rem 1rem",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            color: "var(--text)",
            resize: "vertical",
          }}
        />
        <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={handleAsk}
            disabled={loading || !question.trim() || (mode === "pdf" && uploadLoading)}
            title={
              loading
                ? "Agent is running"
                : mode === "pdf" && uploadLoading
                  ? "Document is still being processed"
                  : !question.trim()
                    ? "Enter a question to ask"
                    : undefined
            }
            aria-busy={loading}
            style={{
              padding: "0.6rem 1.25rem",
              background: "var(--accent)",
              color: "var(--bg)",
              border: "none",
              borderRadius: 8,
              fontWeight: 600,
            }}
          >
            {loading ? "Running..." : "Ask"}
          </button>
        </div>
      </div>

      {error && (
        <div
          style={{
            padding: "1rem",
            background: "rgba(248,113,113,0.1)",
            border: "1px solid var(--error)",
            borderRadius: 8,
            color: "var(--error)",
            marginBottom: "1.5rem",
          }}
        >
          {error}
        </div>
      )}

      {streamSteps.length > 0 && (
        <section style={{ marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem", flexWrap: "wrap", gap: "0.5rem" }}>
            <h2 style={{ fontSize: "1rem", fontWeight: 600, color: "var(--muted)", margin: 0 }}>
              Steps
            </h2>
            <div style={{ display: "flex", gap: "0.25rem" }}>
              <button
                type="button"
                onClick={() => setStepsView("list")}
                aria-pressed={stepsView === "list"}
                aria-label="Show steps as list"
                style={{
                  padding: "0.35rem 0.65rem",
                  fontSize: "0.8rem",
                  background: stepsView === "list" ? "var(--accent)" : "var(--surface)",
                  color: stepsView === "list" ? "var(--bg)" : "var(--text)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  cursor: "pointer",
                }}
              >
                List
              </button>
              <button
                type="button"
                onClick={() => setStepsView("flow")}
                aria-pressed={stepsView === "flow"}
                aria-label="Show steps as flowchart"
                style={{
                  padding: "0.35rem 0.65rem",
                  fontSize: "0.8rem",
                  background: stepsView === "flow" ? "var(--accent)" : "var(--surface)",
                  color: stepsView === "flow" ? "var(--bg)" : "var(--text)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  cursor: "pointer",
                }}
              >
                Flow
              </button>
            </div>
          </div>
          {stepsView === "flow" ? (
            <div style={{ minHeight: 280, height: Math.min(520, 80 + flowData.nodes.length * (PROCESS_NODE_HEIGHT + NODE_VERTICAL_GAP)), width: "100%", border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden", background: "var(--bg)" }}>
              <ReactFlow
                nodes={flowData.nodes}
                edges={flowData.edges}
                nodeTypes={nodeTypes}
                fitView
                fitViewOptions={{ padding: 0.2 }}
                minZoom={0.2}
                maxZoom={1.5}
                nodesDraggable={false}
                nodesConnectable={false}
                elementsSelectable={false}
                proOptions={{ hideAttribution: true }}
              >
                <Background gap={12} size={1} color="var(--border)" />
                <Controls showInteractive={false} />
              </ReactFlow>
            </div>
          ) : (
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {streamSteps.map((s, i) => {
              const query = s.state?.current_search_query
                ? formatSearchQuery(s.state.current_search_query)
                : "";
              const evidenceCount = s.state?.gathered_evidence?.length ?? 0;
              const retrievedCount = s.state?.retrieved_sources?.length ?? 0;
              const finalAnswer = s.state?.final_answer?.trim();
              const trajectorySnippet = s.state?.trajectory?.trim();
              const showPlanningSnippet =
                (s.node === "planning" && !query && evidenceCount === 0 && trajectorySnippet) ?? false;
              const hasContent = query || evidenceCount > 0 || retrievedCount > 0 || finalAnswer || showPlanningSnippet;
              return (
                <li
                  key={i}
                  style={{
                    padding: "0.875rem 1rem",
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    marginBottom: "0.5rem",
                  }}
                >
                  <span
                    style={{
                      display: "inline-block",
                      fontWeight: 600,
                      fontSize: "0.8rem",
                      color: "var(--accent)",
                      textTransform: "uppercase",
                      letterSpacing: "0.04em",
                      marginBottom: hasContent ? "0.5rem" : 0,
                    }}
                  >
                    {formatNodeName(s.node ?? "")}
                  </span>
                  {query && (
                    <div
                      style={{
                        marginTop: "0.25rem",
                        fontSize: "0.9rem",
                        lineHeight: 1.5,
                        color: "var(--text)",
                        padding: "0.5rem 0.75rem",
                        background: "var(--bg)",
                        borderRadius: 6,
                        borderLeft: "3px solid var(--accent)",
                      }}
                    >
                      {query}
                    </div>
                  )}
                  {(s.node === "web_search" || s.node === "search") &&
                    s.state?.retrieved_sources &&
                    s.state.retrieved_sources.length > 0 && (
                      <div style={{ marginTop: "0.5rem" }}>
                        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--muted)", marginBottom: "0.25rem" }}>
                          Retrieved ({s.state.retrieved_sources.length})
                        </div>
                        <ul style={{ margin: 0, paddingLeft: "1.25rem", fontSize: "0.875rem", lineHeight: 1.5, color: "var(--text)", listStyle: "none" }}>
                          {(expandedRetrievedSteps.has(i) ? s.state.retrieved_sources : s.state.retrieved_sources.slice(0, 5)).map((src, j) => {
                            const linkLabel = sourceDisplayTitle(src);
                            const snippet = stripMarkdownLinks(src.content ?? "");
                            const showSnippet = snippet.length > 0 && snippet !== linkLabel && snippet.length > linkLabel.length;
                            return (
                              <li key={j} style={{ marginBottom: "0.6rem" }}>
                                {src.url ? (
                                  <a
                                    href={src.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    style={{ color: "var(--accent)", textDecoration: "none", fontWeight: 500 }}
                                  >
                                    {linkLabel}
                                  </a>
                                ) : (
                                  <span>{linkLabel}</span>
                                )}
                                {showSnippet && (
                                  <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.2rem", lineHeight: 1.4 }}>
                                    {snippet.length > 200 ? snippet.slice(0, 200) + "..." : snippet}
                                  </div>
                                )}
                              </li>
                            );
                          })}
                          {s.state.retrieved_sources.length > 5 && (
                            <li style={{ marginTop: "0.25rem" }}>
                              <button
                                type="button"
                                onClick={() => setExpandedRetrievedSteps((prev) => {
                                  const next = new Set(prev);
                                  if (next.has(i)) next.delete(i);
                                  else next.add(i);
                                  return next;
                                })}
                                aria-expanded={expandedRetrievedSteps.has(i)}
                                aria-label={expandedRetrievedSteps.has(i) ? "Collapse retrieved sources" : `Show ${s.state.retrieved_sources.length - 5} more retrieved sources`}
                                style={{
                                  background: "none",
                                  border: "none",
                                  padding: 0,
                                  fontSize: "0.8rem",
                                  color: "var(--accent)",
                                  cursor: "pointer",
                                  textDecoration: "underline",
                                }}
                              >
                                {expandedRetrievedSteps.has(i) ? "Show less" : `Show ${s.state.retrieved_sources.length - 5} more`}
                              </button>
                            </li>
                          )}
                        </ul>
                      </div>
                    )}
                  {showPlanningSnippet && trajectorySnippet && (
                    <div
                      style={{
                        marginTop: "0.25rem",
                        padding: "0.5rem 0.75rem",
                        background: "var(--bg)",
                        borderRadius: 6,
                        maxHeight: "10rem",
                        overflow: "auto",
                      }}
                    >
                      {formatTrajectoryBlocks(
                        (trajectorySnippet ?? "").length > 500
                          ? (trajectorySnippet ?? "").slice(-500).trim()
                          : (trajectorySnippet ?? "")
                      )}
                    </div>
                  )}
                  {finalAnswer && (
                    <div
                      style={{
                        marginTop: "0.25rem",
                        fontSize: "0.9rem",
                        lineHeight: 1.5,
                        color: "var(--text)",
                        padding: "0.5rem 0.75rem",
                        background: "var(--bg)",
                        borderRadius: 6,
                        borderLeft: "3px solid var(--accent)",
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {finalAnswer}
                    </div>
                  )}
                  {s.state?.gathered_evidence && s.state.gathered_evidence.length > 0 && (
                    <div style={{ marginTop: "0.5rem" }}>
                      <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--muted)", marginBottom: "0.25rem" }}>
                        Evidence ({s.state.gathered_evidence.length})
                      </div>
                      <ul style={{ margin: 0, paddingLeft: "1.25rem", fontSize: "0.875rem", lineHeight: 1.5, color: "var(--text)" }}>
                        {s.state.gathered_evidence.map((ev, j) => (
                          <li key={j} style={{ marginBottom: "0.25rem" }}>
                            {ev}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
          )}
        </section>
      )}

      {result && (
        <section>
          <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.75rem", color: "var(--muted)" }}>
            Answer
          </h2>
          <div
            style={{
              padding: "1rem",
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              whiteSpace: "pre-wrap",
            }}
          >
            {result.final_answer ?? "(no answer)"}
          </div>
          <p style={{ marginTop: "0.75rem", fontSize: "0.9rem", color: "var(--muted)" }}>
            Steps: {result.step_count}
          </p>
          {result.trajectory?.trim() && (
            <details style={{ marginTop: "1rem" }}>
              <summary style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--muted)", cursor: "pointer" }}>
                Planning trace
              </summary>
              <div
                style={{
                  marginTop: "0.5rem",
                  padding: "0.75rem 1rem",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  overflow: "auto",
                  maxHeight: "16rem",
                }}
              >
                {formatTrajectoryBlocks(result.trajectory)}
              </div>
            </details>
          )}
          {result.gathered_evidence.length > 0 && (
            <div style={{ marginTop: "1rem" }}>
              <h3 style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "0.5rem", color: "var(--muted)" }}>
                Evidence ({result.gathered_evidence.length})
              </h3>
              <ul
                style={{
                  listStyle: "none",
                  padding: 0,
                  margin: 0,
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.5rem",
                }}
              >
                {result.gathered_evidence.map((ev, i) => (
                  <li
                    key={i}
                    style={{
                      padding: "0.75rem 1rem",
                      background: "var(--surface)",
                      border: "1px solid var(--border)",
                      borderRadius: 8,
                      fontSize: "0.9rem",
                      lineHeight: 1.5,
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {ev}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      <footer
        style={{
          marginTop: "3rem",
          paddingTop: "1.5rem",
          borderTop: "1px solid var(--border)",
          textAlign: "center",
          fontSize: "0.875rem",
          color: "var(--muted)",
        }}
      >
        <p style={{ margin: 0, display: "flex", alignItems: "center", justifyContent: "center", gap: "0.35rem", flexWrap: "wrap" }}>
          Made for AI Community{" "}
          <a
            href="https://github.com/ashhadahsan"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--accent)", display: "inline-flex", alignItems: "center", gap: "0.35rem" }}
          >
            <svg
              viewBox="0 0 24 24"
              fill="currentColor"
              width={18}
              height={18}
              aria-hidden
            >
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
            github.com/ashhadahsan
          </a>
        </p>
      </footer>
    </main>
  );
}
