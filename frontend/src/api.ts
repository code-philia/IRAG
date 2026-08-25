import type { CandidateDetail, CandidateSummary, GenerationConfirmation, GenerationEvaluation, GenerationResult, GradientAttribution, ManualLinkResponse, SessionPayload, TokenPairAttribution, VisualizationGraph } from "./types";

async function fetchJson<T>(input: RequestInfo | URL, init?: RequestInit, timeoutMs = 60000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(input, { ...init, signal: controller.signal });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s: ${String(input)}`);
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function getExperiments(): Promise<{ tests: string[]; experiments: unknown[]; models?: Array<{ id: string; name: string; available?: boolean }> }> {
  return fetchJson("/api/experiments");
}

export async function loadSession(testId: string, model = "xsearch"): Promise<SessionPayload> {
  return fetchJson("/api/session/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ experimentId: "xsearch_user_study_python", model, testId, topK: 20, epoch: "final" })
  });
}

export async function loadSessionBootstrap(testId: string, model = "xsearch"): Promise<{ session: SessionPayload; candidate: CandidateDetail; graph: VisualizationGraph }> {
  return fetchJson("/api/session/bootstrap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ experimentId: "xsearch_user_study_python", model, testId, topK: 20, epoch: "final" })
  });
}

export async function loadCandidate(testId: string, candidateId: string, model = "xsearch"): Promise<CandidateDetail> {
  return fetchJson(`/api/candidates/${candidateId}?test_id=${encodeURIComponent(testId)}&model=${encodeURIComponent(model)}`);
}

export async function loadGraph(testId: string, candidateId: string, epoch = 4, model = "xsearch"): Promise<VisualizationGraph> {
  return fetchJson(
    `/api/visualize/graph?test_id=${encodeURIComponent(testId)}&candidate_id=${encodeURIComponent(candidateId)}&epoch=${epoch}&model=${encodeURIComponent(model)}`
  );
}

export function logEvent(eventType: string, eventData: Record<string, unknown>) {
  void fetch("/api/logs/events", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionId: "local_preview",
      eventType,
      eventData
    })
  });
}

export async function createManualLink(payload: {
  testId: string;
  candidateId: string;
  queryTokenIndex: number;
  codeTokenIndex: number;
  epoch: number;
  distance: number;
  color?: string;
}): Promise<ManualLinkResponse> {
  return fetchJson("/api/intervention/manual-link", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function applyDragRerank(payload: {
  model?: string;
  testId: string;
  candidateId: string;
  draggedNode: { id: string; type: "query_token" | "code_token"; tokenIndex: number };
  pairInterventions: Array<{
    queryTokenIndex: number;
    codeTokenIndex: number;
    originalProximity: number;
    currentProximity: number;
    proximityDelta: number;
    modelCosine: number;
  }>;
}): Promise<{
  status: "ok";
  candidates: CandidateSummary[];
  generalizedMatchesByCandidate?: Record<string, unknown>;
  localMatches?: {
    tokenMatches: Array<Record<string, unknown>>;
    lineMatches: Array<Record<string, unknown>>;
  };
  diagnostic: Record<string, string | number>;
}> {
  return fetchJson("/api/intervention/drag-rerank", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function resetInterventions(testId?: string, model = "xsearch"): Promise<{ status: "ok"; clearedMemories: number }> {
  return fetchJson("/api/intervention/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ testId, model })
  });
}

export async function loadTokenPairAttribution(payload: {
  testId: string;
  candidateId: string;
  queryTokenIndex: number;
  codeTokenIndex: number;
  epoch: number;
  topK?: number;
}): Promise<TokenPairAttribution> {
  return fetchJson("/api/diagnostics/token-pair-attribution", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function runGradientAttribution(payload: {
  testId: string;
  candidateId: string;
  queryTokenIndex: number;
  codeTokenIndex: number;
  mode?: "pull" | "push";
  lossScope?: "highlight_only" | "highlight_plus_cross_sample_batch";
  crossSampleWeight?: number;
  topK?: number;
  maxTrainSamples?: number;
  maxBatches?: number;
}): Promise<GradientAttribution> {
  return fetchJson("/api/diagnostics/token-pair-gradient-attribution", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  }, 120000);
}

export async function confirmReference(payload: {
  caseId: string;
  selectedCandidateId: string;
  selectedRank: number | null;
  selectedScore: number | null;
  interactionUsed: boolean;
  model: string;
}): Promise<GenerationConfirmation> {
  return fetchJson("/api/generation/confirm-reference", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function generateCode(payload: {
  caseId: string;
  selectionId?: string;
  condition: "no_rag" | "automatic_rag" | "interactive_rag";
}): Promise<GenerationResult> {
  return fetchJson("/api/generation/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  }, 130000);
}

export async function evaluateGeneration(generationId: string): Promise<GenerationEvaluation> {
  return fetchJson("/api/generation/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generationId })
  }, 30000);
}
