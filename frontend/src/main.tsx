import React, { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import { ArrowDown, ArrowLeftRight, ArrowUp, BookOpenCheck, CirclePlay, Loader2, Maximize2, Move, RefreshCw, ZoomIn, ZoomOut } from "lucide-react";
import { applyDragRerank, confirmReference, createManualLink, evaluateGeneration, finalizeReference, generateCode, getExperiments, loadCandidate, loadGenerationComparison, loadGraph, loadReferenceHint, loadSession, loadSessionBootstrap, loadTaskBrief, loadTokenPairAttribution, logEvent, resetInterventions, runGradientAttribution, setEventContext, startStudySession } from "./api";
import type {
  CandidateDetail,
  CandidateSummary,
  Concept,
  GenerationComparison,
  GenerationConfirmation,
  GenerationEvaluation,
  GenerationResult,
  GradientAttribution,
  GraphNode,
  ManualLink,
  ReferenceHint,
  SessionPayload,
  StudySession,
  TaskBrief,
  TokenPairAttribution,
  VisualizationGraph
} from "./types";
import "./styles.css";

const DEFAULT_TEST_ID = "";
type AppMode = "demo" | "study" | "study_dev" | "baseline";
const APP_MODE: AppMode = window.location.pathname.startsWith("/baseline")
  ? "baseline"
  : window.location.pathname.startsWith("/study_dev")
    ? "study_dev"
    : window.location.pathname.startsWith("/study")
      ? "study"
      : "demo";
const FOCUS_TEST_IDS = [DEFAULT_TEST_ID, "48", "1556", "1642", "2695", "3856", "954", "csn_9848", "csn_11087", "csn_11078", "csn_9406", "csn_400", "csn_13958", "csn_13527", "csn_8838", "csn_7664", "csn_2613", "csn_12213", "csn_11772", "csn_9388", "csn_2812", "csn_7727", "csn_4772", "csn_10023", "csn_2207", "csn_5340", "csn_10164", "csn_13655", "csn_14175", "csn_10643", "csn_12075"];
const GENERATION_CASE_IDS = new Set(["csn_8838", "csn_7664", "csn_2613", "csn_12213", "csn_11772", "csn_8884", "csn_3846", "csn_42", "csn_9388"]);
const AUTO_TOKEN_SUGGESTIONS_ENABLED = false;
const GRAPH_WIDTH = 880;
const GRAPH_HEIGHT = 560;
const SUPPORT_QUERY_EVIDENCE_THRESHOLD = 0.55;
const CONFLICT_QUERY_EVIDENCE_THRESHOLD = 0.35;
const CANDIDATE_PREFETCH_CONCURRENCY = 3;
const MAX_DISPLAY_SIMILARITY = 0.95;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function safeDisplaySimilarity(value: number | null | undefined, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > -1e6
    ? clamp(numeric, 0, MAX_DISPLAY_SIMILARITY)
    : clamp(fallback, 0, MAX_DISPLAY_SIMILARITY);
}

function displayToken(token: string | number | null | undefined) {
  const raw = String(token ?? "");
  const cleaned = raw
    .replace(/\\u0120/g, "")
    .replace(/\\u010a/g, "")
    .replace(/\u0120/g, "")
    .replace(/\u010a/g, "")
    .replace(/\\n/g, "")
    .replace(/^##/, "")
    .trim();
  if (!cleaned || /^G+$/.test(cleaned)) return "";
  return cleaned;
}

function tokenFamilyKey(token: string) {
  const key = displayToken(token).toLowerCase().replace(/[^a-z0-9]/g, "");
  if (key.endsWith("ies") && key.length > 4) return `${key.slice(0, -3)}y`;
  if (key.endsWith("s") && key.length > 4) return key.slice(0, -1);
  return key;
}

function repeatedCodeTokenKey(token: string) {
  return displayToken(token).trim().toLowerCase();
}

function isPunctuationToken(token: string | number | null | undefined) {
  const value = displayToken(token);
  return Boolean(value) && /^[^\p{L}\p{N}_]+$/u.test(value);
}

function displayConceptText(text: string | number | null | undefined) {
  return String(text ?? "")
    .replace(/\\u0120/g, " ")
    .replace(/\\u010a/g, " ")
    .replace(/\u0120/g, " ")
    .replace(/\u010a/g, " ")
    .replace(/\\n/g, " ")
    .split(/\s+/)
    .filter((part) => part && !/^G+$/.test(part))
    .join(" ");
}

function displayQueryOriginal(text: string | number | null | undefined) {
  const raw = displayToken(text).replace(/\r/g, "\n");
  const withoutCodeBlock = raw.split(/\.\.\s*code-block::|code-block::/i)[0];
  const firstParagraph = withoutCodeBlock.split(/\n\s*\n/)[0];
  return firstParagraph.replace(/\s+/g, " ").trim() || raw.replace(/\s+/g, " ").trim();
}

function withoutLeadingFunctionDocstring(code: string) {
  const lines = code.split("\n");
  let functionBodyStarted = false;
  for (let index = 0; index < lines.length; index += 1) {
    const trimmed = lines[index].trim();
    if (!functionBodyStarted) {
      if (trimmed.endsWith(":")) functionBodyStarted = true;
      continue;
    }
    if (!trimmed) continue;
    const opening = trimmed.match(/^(\"\"\"|''')/);
    if (!opening) return code;
    const quote = opening[1];
    const remaining = trimmed.slice(quote.length);
    let endIndex = index;
    if (!remaining.includes(quote)) {
      endIndex += 1;
      while (endIndex < lines.length && !lines[endIndex].includes(quote)) endIndex += 1;
      if (endIndex >= lines.length) return code;
    }
    return [...lines.slice(0, index), ...lines.slice(endIndex + 1)].join("\n");
  }
  return code;
}

function createClientId(prefix: string) {
  const randomUuid = globalThis.crypto?.randomUUID;
  if (typeof randomUuid === "function") return `${prefix}_${randomUuid.call(globalThis.crypto)}`;
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 12)}`;
}

function graphTokenLabel(token: string | number | null | undefined) {
  const cleaned = displayToken(token).trim();
  if (!cleaned || cleaned.startsWith("#") || /^[()[\]{}.,:;'"`=+\-*/%<>!|&]+$/.test(cleaned)) return "";
  return cleaned;
}

function selectedGraphTokenLabel(node: GraphNode, selected: boolean) {
  return graphTokenLabel(node.label) || (selected ? displayToken(node.label).trim() : "");
}

function isSemanticToken(token: string | number | null | undefined) {
  const value = displayToken(token);
  const lowInformationTokens = new Set([
    "self", "cls", "def", "class", "return", "none", "true", "false", "is",
    "when", "a", "an", "the", "this", "function", "and", "or", "not", "if", "else"
  ]);
  return Boolean(value)
    && !value.startsWith("#")
    && !/^[()[\]{}.,:;'"`=+\-*/%<>!|&]+$/.test(value)
    && !lowInformationTokens.has(value.toLowerCase());
}

function isAdjudicationUncoveredLine(line: CandidateDetail["codeLines"][number], matches: CandidateDetail["conceptMatches"]) {
  if (matches.length) return false;
  const semanticTokens = line.tokenIndices
    .map((index) => line.text && index)
    .filter((value): value is number => Number.isInteger(value));
  return semanticTokens.length > 0 && semanticTokens.length <= 4;
}

function isUnavailableTokenError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes("Code token has no valid model hidden slot")
    || message.includes("Code token has no valid subtoken slot")
    || message.includes("valid model hidden slot")
    || message.includes("valid subtoken slot");
}

function dragMatchKey(match: Pick<DragTokenMatch, "queryTokenIndex" | "codeTokenIndex">) {
  return `${match.queryTokenIndex}:${match.codeTokenIndex}`;
}

function isQuerySideEvidence(item: { querySimilarity?: number | null }, tone: "support" | "conflict") {
  const threshold = tone === "support" ? SUPPORT_QUERY_EVIDENCE_THRESHOLD : CONFLICT_QUERY_EVIDENCE_THRESHOLD;
  return Number(item.querySimilarity ?? 0) >= threshold;
}

function manualLinkKey(testId: string, candidateId: string) {
  return `${testId}::${candidateId}`;
}

function nodeDistance(a: GraphNode, b: GraphNode) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function projectionSimilarity(distance: number) {
  return 1 / (1 + distance / 120);
}

function selectedQueryCodePair(graph: VisualizationGraph | null, selectedTokenIds: string[]) {
  if (!graph) return null;
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const selectedNodes = selectedTokenIds.map((id) => nodeById.get(id)).filter((node): node is GraphNode => Boolean(node));
  const queryNode = [...selectedNodes].reverse().find((node) => node.type === "query_token");
  const codeNode = [...selectedNodes].reverse().find((node) => node.type === "code_token");
  if (!queryNode || !codeNode) return null;
  return { queryNode, codeNode };
}

type NeighborSignal = {
  status: "new" | "closer" | "farther" | "stable";
  distance: number;
  delta: number;
};

type TrailPoint = {
  x: number;
  y: number;
};

function nearestNeighbors(graph: VisualizationGraph | null, focusId: string | null, limit = 5) {
  if (!graph || !focusId) return [];
  const focus = graph.nodes.find((node) => node.id === focusId);
  if (!focus) return [];
  return graph.nodes
    .filter((node) => node.id !== focus.id)
    .map((node) => ({ node, distance: nodeDistance(focus, node), similarity: projectionSimilarity(nodeDistance(focus, node)) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, limit);
}

type DragTokenMatch = {
  queryTokenIndex: number;
  codeTokenIndex: number;
  similarity: number;
  baseline?: number;
  delta?: number;
  queryToken?: string;
  codeToken?: string;
  color: string;
  newlyConnected: boolean;
  source?: "local_drag" | "generalized";
  role?: "dragged" | "follower" | "generalized";
};

type DragLineMatch = {
  conceptId: number;
  lineNumber: number;
  previousLineNumber?: number;
  similarity: number;
  baseline: number;
  delta: number;
  color: string;
  source?: "local_drag" | "generalized";
};

type DragMatchBundle = {
  tokenMatches: DragTokenMatch[];
  lineMatches: DragLineMatch[];
};

type ExternalImpact = {
  queryTokenIndex: number;
  codeTokenIndex: number;
  delta: number;
  color: string;
  sourceCandidateId: string;
  targetQueryTokenIndices: number[];
};

type CanvasLevel = "block" | "line" | "line_tokens" | "token";
type QueryPointMode = "concept" | "tokens";

type HierarchyScoredNode = {
  id: string;
  conceptScores: Array<{ conceptId: number; similarity: number }>;
  displayConceptIds?: number[];
};

const HIERARCHY_CONCEPT_MATCH_THRESHOLD = 0.2;

function conceptWinnerMap<T extends HierarchyScoredNode>(nodes: T[], concepts: Concept[]) {
  const winners = new Map<string, Array<{ concept: Concept; similarity: number }>>();
  concepts.forEach((concept) => {
    const displayNodes = nodes.filter((node) => node.displayConceptIds?.includes(concept.conceptId));
    if (displayNodes.length) {
      displayNodes.forEach((node) => {
        const similarity = (Array.isArray(node.conceptScores) ? node.conceptScores : []).find((score) => score.conceptId === concept.conceptId)?.similarity ?? 0;
        winners.set(node.id, [...(winners.get(node.id) ?? []), { concept, similarity }]);
      });
      return;
    }
    const winner = nodes
      .map((node) => ({ node, score: (Array.isArray(node.conceptScores) ? node.conceptScores : []).find((score) => score.conceptId === concept.conceptId)?.similarity ?? -Infinity }))
      .sort((first, second) => second.score - first.score)[0];
    if (!winner || !Number.isFinite(winner.score) || winner.score < HIERARCHY_CONCEPT_MATCH_THRESHOLD) return;
    winners.set(winner.node.id, [...(winners.get(winner.node.id) ?? []), { concept, similarity: winner.score }]);
  });
  return winners;
}

function hierarchyFill(id: string, matches: Array<{ concept: Concept; similarity: number }>) {
  if (!matches.length) return "#cbd5e1";
  if (matches.length === 1) return matches[0].concept.color;
  return `url(#${id})`;
}

function trianglePoints(x: number, y: number, radius: number) {
  return `${x},${y - radius} ${x - radius * 0.9},${y + radius * 0.72} ${x + radius * 0.9},${y + radius * 0.72}`;
}

type HierarchySignal = { kind?: string; conceptId?: number; tokenIndex?: number };

function hierarchySignalTexts(signals: unknown, candidate: CandidateDetail, concepts: Map<number, Concept>) {
  const values = Array.isArray(signals) ? signals as HierarchySignal[] : [];
  const text = values.flatMap((signal) => {
    if (signal.kind === "weak_block") return ["weak concept evidence"];
    if (signal.kind === "uncovered_line") return ["uncovered code detail"];
    if (signal.kind !== "latent_token" || signal.tokenIndex == null || !Number.isInteger(signal.tokenIndex)) return [];
    const concept = signal.conceptId == null ? undefined : concepts.get(signal.conceptId);
    const token = displayToken(candidate.codeTokens[signal.tokenIndex]);
    return token ? [`inspect ${token} ↔ ${concept ? displayConceptText(concept.text) : "concept"}`] : [];
  });
  return [...new Set(text)].slice(0, 3);
}

type HierarchyAnnotation = {
  x: number;
  y: number;
  width: number;
  height: number;
  labelX: number;
  labelY: number;
  textAnchor: "start" | "middle" | "end";
  leaderX: number;
  leaderY: number;
};

type HierarchyLayoutBox = { x: number; y: number; width: number; height: number };

type GraphLabelLayoutItem = {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  priority: number;
};

type GraphLabelPlacement = {
  x: number;
  y: number;
  side: "left" | "right" | "top" | "bottom";
};

function graphLabelLayout(
  items: GraphLabelLayoutItem[],
  markerObstacles: HierarchyLayoutBox[]
) {
  const padding = 8;
  const expand = (box: HierarchyLayoutBox, amount = 3): HierarchyLayoutBox => ({
    x: box.x - amount,
    y: box.y - amount,
    width: box.width + amount * 2,
    height: box.height + amount * 2
  });
  const overlapArea = (first: HierarchyLayoutBox, second: HierarchyLayoutBox) => {
    const width = Math.max(0, Math.min(first.x + first.width, second.x + second.width) - Math.max(first.x, second.x));
    const height = Math.max(0, Math.min(first.y + first.height, second.y + second.height) - Math.max(first.y, second.y));
    return width * height;
  };
  const occupied = markerObstacles.map((box) => expand(box, 4));
  const layout = new Map<string, HierarchyAnnotation>();
  [...items]
    .sort((first, second) => second.priority - first.priority || first.id.localeCompare(second.id))
    .forEach((item) => {
      const placements: GraphLabelPlacement[] = [
        { x: item.x + 15, y: item.y - item.height / 2, side: "right" },
        { x: item.x - item.width - 15, y: item.y - item.height / 2, side: "left" },
        { x: item.x - item.width / 2, y: item.y - item.height - 15, side: "top" as const },
        { x: item.x - item.width / 2, y: item.y + 15, side: "bottom" as const }
      ];
      const candidates = placements.map((candidate) => ({
        ...candidate,
        x: clamp(candidate.x, padding, GRAPH_WIDTH - item.width - padding),
        y: clamp(candidate.y, padding, GRAPH_HEIGHT - item.height - padding),
        width: item.width,
        height: item.height
      }));
      const scored = candidates.map((candidate, index) => ({
        candidate,
        overlap: occupied.reduce((total, box) => total + overlapArea(candidate, box), 0),
        index
      })).sort((first, second) => first.overlap - second.overlap || first.index - second.index);
      const chosen = scored[0];
      // Labels stay adjacent to their node. If none of the four directions is
      // clear, ordinary labels yield instead of becoming distant annotations.
      if (!chosen || (chosen.overlap > 0 && item.priority < 1000)) return;
      const labelY = chosen.candidate.y + 13;
      const annotation: HierarchyAnnotation = chosen.candidate.side === "left"
        ? { ...chosen.candidate, labelX: chosen.candidate.x + chosen.candidate.width, labelY, textAnchor: "end", leaderX: chosen.candidate.x + chosen.candidate.width + 5, leaderY: item.y }
        : chosen.candidate.side === "right"
          ? { ...chosen.candidate, labelX: chosen.candidate.x, labelY, textAnchor: "start", leaderX: chosen.candidate.x - 5, leaderY: item.y }
          : { ...chosen.candidate, labelX: chosen.candidate.x + chosen.candidate.width / 2, labelY, textAnchor: "middle", leaderX: item.x, leaderY: chosen.candidate.side === "top" ? chosen.candidate.y + chosen.candidate.height + 5 : chosen.candidate.y - 5 };
      layout.set(item.id, annotation);
      occupied.push(expand(chosen.candidate));
    });
  return layout;
}

function hierarchyAnnotationLayout(
  items: Array<{ id: string; x: number; y: number; width: number; height: number; placement?: "side" | "free" }>,
  obstacles: HierarchyLayoutBox[] = []
) {
  const padding = 8;
  const expand = (box: HierarchyLayoutBox, amount = 5): HierarchyLayoutBox => ({
    x: box.x - amount,
    y: box.y - amount,
    width: box.width + amount * 2,
    height: box.height + amount * 2
  });
  const occupied: HierarchyLayoutBox[] = [
    ...obstacles.map((box) => expand(box)),
    ...items.map((item) => ({ x: item.x - 17, y: item.y - 17, width: 34, height: 34 }))
  ];
  const result = new Map<string, HierarchyAnnotation>();
  const overlapArea = (
    first: { x: number; y: number; width: number; height: number },
    second: { x: number; y: number; width: number; height: number }
  ) => {
    const width = Math.max(0, Math.min(first.x + first.width, second.x + second.width) - Math.max(first.x, second.x));
    const height = Math.max(0, Math.min(first.y + first.height, second.y + second.height) - Math.max(first.y, second.y));
    return width * height;
  };
  items.forEach((item) => {
    const offsets = [0, -28, 28, -56, 56, -84, 84, -112, 112, -140, 140];
    const sideCandidates = offsets.flatMap((verticalOffset) => [
      { x: item.x + 22, y: item.y - item.height / 2 + verticalOffset, side: "right" as const },
      { x: item.x - item.width - 22, y: item.y - item.height / 2 + verticalOffset, side: "left" as const }
    ]);
    const candidates = (item.placement === "side" ? sideCandidates : [
      ...sideCandidates,
      ...[-22, -50, -78, -106].flatMap((offset) => [
        { x: item.x - item.width / 2, y: item.y - item.height + offset, side: "top" as const },
        { x: item.x - item.width / 2, y: item.y - offset, side: "bottom" as const }
      ])
    ]).map((candidate) => ({
      x: clamp(candidate.x, padding, GRAPH_WIDTH - item.width - padding),
      y: clamp(candidate.y, padding, GRAPH_HEIGHT - item.height - padding),
      width: item.width,
      height: item.height,
      side: candidate.side
    }));
    const chosen = candidates
      .map((candidate, index) => ({
        candidate,
        score: occupied.reduce((sum, box) => sum + overlapArea(candidate, box), 0) * 100 + index * 0.02
      }))
      .sort((first, second) => first.score - second.score)[0].candidate;
    const labelY = chosen.y + 14;
    const annotation: HierarchyAnnotation = chosen.side === "left"
      ? { ...chosen, labelX: chosen.x + chosen.width, labelY, textAnchor: "end", leaderX: chosen.x + chosen.width + 6, leaderY: item.y }
      : chosen.side === "right"
        ? { ...chosen, labelX: chosen.x, labelY, textAnchor: "start", leaderX: chosen.x - 6, leaderY: item.y }
        : { ...chosen, labelX: chosen.x + chosen.width / 2, labelY, textAnchor: "middle", leaderX: item.x, leaderY: chosen.side === "top" ? chosen.y + chosen.height + 6 : chosen.y - 6 };
    result.set(item.id, annotation);
    occupied.push(expand(chosen));
  });
  return result;
}

function HierarchicalCanvas({
  candidate,
  session,
  graph,
  level,
  selectedBlockId,
  selectedLines,
  selectedConcepts,
  selectedTokenIds,
  onLevel,
  onBlock,
  onLine,
  onConcept,
  onToken
}: {
  candidate: CandidateDetail | null;
  session: SessionPayload | null;
  graph: VisualizationGraph | null;
  level: CanvasLevel;
  selectedBlockId: string | null;
  selectedLines: number[];
  selectedConcepts: number[];
  selectedTokenIds: string[];
  onLevel: (level: CanvasLevel) => void;
  onBlock: (id: string) => void;
  onLine: (lineNumber: number) => void;
  onConcept: (conceptId: number) => void;
  onToken: (id: string) => void;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const panRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const viewportRef = useRef({ zoom: 1, pan: { x: GRAPH_WIDTH / 2, y: GRAPH_HEIGHT / 2 } });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: GRAPH_WIDTH / 2, y: GRAPH_HEIGHT / 2 });
  const [hoveredBlockId, setHoveredBlockId] = useState<string | null>(null);
  const blocks = useMemo(() => graph?.hierarchy?.blocks ?? [], [graph]);
  const selectedBlock = blocks.find((block) => block.id === selectedBlockId);
  const displayLineNumberByRaw = useMemo(() => new Map(
    (candidate?.codeLines ?? []).filter((line) => line.tokenIndices.length > 0).map((line, index) => [line.lineNumber, index + 1])
  ), [candidate]);
  const displayLineNumber = (rawLineNumber: number) => displayLineNumberByRaw.get(rawLineNumber) ?? rawLineNumber;
  const visibleLines = level === "line" && selectedBlock ? graph?.hierarchy?.linesByBlock[selectedBlock.id] ?? [] : [];
  const selectedLine = selectedLines[0] ?? null;
  const visibleLineTokenIndices = level === "line_tokens" && selectedBlock && selectedLine != null
    ? new Set((graph?.hierarchy?.linesByBlock[selectedBlock.id] ?? []).find((line) => line.lineNumber === selectedLine)?.tokenIndices ?? [])
    : new Set<number>();
  useEffect(() => {
    setZoom(1);
    setPan({ x: GRAPH_WIDTH / 2, y: GRAPH_HEIGHT / 2 });
    panRef.current = null;
  }, [graph?.candidateId, graph?.epoch, level]);
  useEffect(() => {
    viewportRef.current = { zoom, pan };
  }, [zoom, pan]);
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return undefined;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = svg.getBoundingClientRect();
      const current = viewportRef.current;
      const width = GRAPH_WIDTH / current.zoom;
      const height = GRAPH_HEIGHT / current.zoom;
      const relX = rect.width ? (event.clientX - rect.left) / rect.width : 0.5;
      const relY = rect.height ? (event.clientY - rect.top) / rect.height : 0.5;
      const cursorX = current.pan.x - width / 2 + relX * width;
      const cursorY = current.pan.y - height / 2 + relY * height;
      const nextZoom = clamp(current.zoom * (event.deltaY < 0 ? 1.12 : 0.88), 0.65, 6);
      const nextWidth = GRAPH_WIDTH / nextZoom;
      const nextHeight = GRAPH_HEIGHT / nextZoom;
      const nextPan = { x: cursorX - (relX - 0.5) * nextWidth, y: cursorY - (relY - 0.5) * nextHeight };
      viewportRef.current = { zoom: nextZoom, pan: nextPan };
      setZoom(nextZoom);
      setPan(nextPan);
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, []);
  const viewWidth = GRAPH_WIDTH / zoom;
  const viewHeight = GRAPH_HEIGHT / zoom;
  const viewX = pan.x - viewWidth / 2;
  const viewY = pan.y - viewHeight / 2;
  const changeZoom = (factor: number) => setZoom((current) => clamp(current * factor, 0.65, 6));
  const resetViewport = () => {
    setZoom(1);
    setPan({ x: GRAPH_WIDTH / 2, y: GRAPH_HEIGHT / 2 });
  };
  const startPan = (event: React.MouseEvent<SVGSVGElement>) => {
    if ((event.target as SVGElement).closest(".hierarchy-node, .hierarchy-query-node")) return;
    panRef.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
  };
  const movePan = (event: React.MouseEvent<SVGSVGElement>) => {
    const current = panRef.current;
    if (!current || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    setPan({
      x: current.panX - ((event.clientX - current.x) / Math.max(1, rect.width)) * viewWidth,
      y: current.panY - ((event.clientY - current.y) / Math.max(1, rect.height)) * viewHeight
    });
  };
  if (!candidate || !session || !graph) return <section className="panel canvas-panel muted">waiting for projection</section>;
  const concepts = new Map(session.query.concepts.map((concept) => [concept.conceptId, concept]));
  const blockWinners = conceptWinnerMap(blocks, session.query.concepts);
  const lineNodes = visibleLines.map((line) => ({ ...line, id: `line_${line.lineNumber}` }));
  const lineWinners = conceptWinnerMap(lineNodes, session.query.concepts);
  const projectedQueryNodes = level === "line" && selectedBlock
    ? graph.hierarchy?.queryNodesByBlock?.[selectedBlock.id] ?? graph.hierarchy?.queryNodes ?? []
    : graph.hierarchy?.queryNodes ?? [];
  const queryNodes = projectedQueryNodes.filter((node) => {
    if (node.type === "query_concept") return Boolean(displayConceptText(node.label));
    const tokenId = node.id.replace("q_token_", "q_tok_");
    return Boolean(graphTokenLabel(node.label)) && selectedTokenIds.includes(tokenId);
  });
  const visibleCodeNodes = level === "block" ? blocks : lineNodes;
  const hierarchyLabel = (node: { label?: string; lineNumber?: number; text?: string }) => {
    if (level === "block") return node.label ?? "Block";
    const text = `L${displayLineNumber(node.lineNumber ?? 0)} ${(node.text ?? "").trim()}`;
    return text.length > 42 ? `${text.slice(0, 39)}...` : text;
  };
  const labelWidth = (text: string) => clamp(Math.ceil(text.length * 6.7 + 16), 110, 290);
  const queryLabelObstacles = queryNodes.map((node) => {
    const rawLabel = node.type === "query_concept" ? displayConceptText(node.label) : graphTokenLabel(node.label);
    const label = rawLabel.length > 24 ? `${rawLabel.slice(0, 21)}...` : rawLabel;
    const width = labelWidth(label);
    const placeLeft = node.labelPlacement === "left" || node.x > GRAPH_WIDTH - 190;
    return {
      x: placeLeft ? node.x - width - 15 : node.x + 15,
      y: node.y - 12,
      width,
      height: 22
    };
  });
  const annotationLayout = hierarchyAnnotationLayout([
    ...visibleCodeNodes.map((node) => ({ id: node.id, x: node.x, y: node.y, width: labelWidth(hierarchyLabel(node)), height: 22, placement: "free" as const }))
  ], queryLabelObstacles);
  const hoveredBlock = blocks.find((block) => block.id === hoveredBlockId);
  const hoveredBlockTooltipX = hoveredBlock ? clamp(hoveredBlock.x + 16, 12, GRAPH_WIDTH - 208) : 0;
  const hoveredBlockTooltipY = hoveredBlock ? clamp(hoveredBlock.y - 46, 12, GRAPH_HEIGHT - 48) : 0;
  const renderQueryNodes = () => queryNodes.map((node) => {
    const label = node.type === "query_concept" ? displayConceptText(node.label) : graphTokenLabel(node.label);
    const concept = node.conceptId == null ? undefined : concepts.get(node.conceptId);
    const tokenId = node.id.replace("q_token_", "q_tok_");
    const active = node.conceptId != null ? selectedConcepts.includes(node.conceptId) : selectedTokenIds.includes(tokenId);
    const placeLeft = node.labelPlacement === "left" || node.x > GRAPH_WIDTH - 190;
    return (
      <g
        key={node.id}
        className={active ? "hierarchy-query-node active" : "hierarchy-query-node"}
        onClick={() => node.conceptId != null ? onConcept(node.conceptId) : onToken(tokenId)}
      >
        <circle cx={node.x} cy={node.y} r="13" fill={concept?.color ?? "#64748b"} />
        <text x={node.x + (placeLeft ? -15 : 15)} y={node.y + 4} textAnchor={placeLeft ? "end" : "start"} className="hierarchy-query-label">{label.length > 24 ? `${label.slice(0, 21)}...` : label}</text>
      </g>
    );
  });
  return (
    <section className="panel canvas-panel hierarchy-canvas">
      <div className="canvas-head">
        <div>
          <div className="panel-title">Embedding Space</div>
          <div className="meta-line">{level === "block" ? "Semantic block overview" : level === "line" ? `Lines in ${selectedBlock?.label ?? "selected block"}` : "Token-level editing"}</div>
        </div>
        <div className="canvas-head-controls">
          <div className="hierarchy-breadcrumb">
            {(["block", "line", "line_tokens", "token"] as CanvasLevel[]).map((item) => (
              <button key={item} className={level === item ? "hierarchy-step active" : "hierarchy-step"} onClick={() => onLevel(item)} disabled={(item === "line" || item === "line_tokens") && !selectedBlock}>
                {item === "block" ? "Blocks" : item === "line" ? "Lines" : item === "line_tokens" ? "Line Tokens" : "All Tokens"}
              </button>
            ))}
            {level === "block" && selectedBlock ? <button className="hierarchy-open" onClick={() => onLevel("line")}>Open block</button> : null}
            {level === "line" && selectedLine != null ? <button className="hierarchy-open" onClick={() => onLevel("line_tokens")}>Open line</button> : null}
          </div>
          <div className="canvas-viewport-tools" aria-label="Canvas zoom controls">
            <button onClick={() => changeZoom(1.2)} title="Zoom in" aria-label="Zoom in"><ZoomIn size={15} /></button>
            <button onClick={() => changeZoom(1 / 1.2)} title="Zoom out" aria-label="Zoom out"><ZoomOut size={15} /></button>
            <button onClick={resetViewport} title="Reset view" aria-label="Reset view"><Maximize2 size={14} /></button>
          </div>
        </div>
      </div>
      <svg ref={svgRef} className="graph hierarchy-graph" viewBox={`${viewX} ${viewY} ${viewWidth} ${viewHeight}`} role="img" onMouseDown={startPan} onMouseMove={movePan} onMouseUp={() => { panRef.current = null; }} onMouseLeave={() => { panRef.current = null; }}>
        <defs>
          {[...blockWinners, ...lineWinners].filter(([, matches]) => matches.length > 1).map(([id, matches]) => (
            <linearGradient key={id} id={`hierarchy-gradient-${id}`} x1="0" x2="1">
              {matches.map((match, index) => <stop key={match.concept.conceptId} offset={`${(index / Math.max(1, matches.length - 1)) * 100}%`} stopColor={match.concept.color} />)}
            </linearGradient>
          ))}
        </defs>
        {level === "block" ? renderQueryNodes().concat(blocks.map((block) => {
          const matches = blockWinners.get(block.id) ?? [];
          const gradientId = `hierarchy-gradient-${block.id}`;
          const annotation = annotationLayout.get(block.id);
          return (
            <g
              key={block.id}
              className={selectedBlockId === block.id ? "hierarchy-node active" : "hierarchy-node"}
              onClick={() => onBlock(block.id)}
              onMouseEnter={() => setHoveredBlockId(block.id)}
              onMouseLeave={() => setHoveredBlockId((current) => current === block.id ? null : current)}
            >
              <polygon points={trianglePoints(block.x, block.y, 19)} fill={hierarchyFill(gradientId, matches)} />
              {annotation ? <><line className="hierarchy-label-leader" x1={block.x} y1={block.y} x2={annotation.leaderX} y2={annotation.leaderY} /><text x={annotation.labelX} y={annotation.labelY} textAnchor={annotation.textAnchor} className="hierarchy-code-label">{block.label}</text></> : null}
              <title>{block.label}</title>
            </g>
          );
        })) : null}
        {level === "block" && hoveredBlock ? (
          <g className="hierarchy-block-tooltip" pointerEvents="none">
            <rect x={hoveredBlockTooltipX} y={hoveredBlockTooltipY} width="196" height="38" rx="4" />
            <text x={hoveredBlockTooltipX + 9} y={hoveredBlockTooltipY + 15}>{hoveredBlock.label}</text>
            <text x={hoveredBlockTooltipX + 9} y={hoveredBlockTooltipY + 29}>Click to highlight · Open block for lines</text>
          </g>
        ) : null}
        {level === "line" && selectedBlock ? renderQueryNodes().concat(lineNodes.map((line) => {
          const matches = lineWinners.get(line.id) ?? [];
          const x = line.x;
          const y = line.y;
          const annotation = annotationLayout.get(line.id);
          return (
            <g key={line.lineNumber} className={selectedLine === line.lineNumber ? "hierarchy-node active" : "hierarchy-node"} onClick={() => onLine(line.lineNumber)}>
              <polygon points={trianglePoints(x, y, 13)} fill={hierarchyFill(`hierarchy-gradient-${line.id}`, matches)} />
              {annotation ? <><line className="hierarchy-label-leader" x1={x} y1={y} x2={annotation.leaderX} y2={annotation.leaderY} /><text x={annotation.labelX} y={annotation.labelY} textAnchor={annotation.textAnchor} className="hierarchy-code-label">{hierarchyLabel(line)}</text></> : null}
            </g>
          );
        })) : null}
        {level === "line_tokens" && selectedBlock ? graph.nodes.filter((node) => {
          const label = graphTokenLabel(node.label);
          return Boolean(label) && (node.type === "query_token" || visibleLineTokenIndices.has(node.tokenIndex));
        }).map((node) => (
          <g key={node.id} className={`${selectedTokenIds.includes(node.id) ? "hierarchy-node active" : "hierarchy-node"} ${node.type === "code_token" ? "code-token-node" : "query-token-node"}`} onClick={() => onToken(node.id)}>
            {node.type === "query_token"
              ? <circle cx={node.x} cy={node.y} r="10" fill={node.color} />
              : <polygon points={trianglePoints(node.x, node.y, 9)} fill={node.color} />}
            <text x={node.x + 12} y={node.y + 4}>{graphTokenLabel(node.label)}</text>
          </g>
        )) : null}
      </svg>
      <div className="hierarchy-hint">Select a node to coordinate with the side panels. Use Open block or Open line to inspect the next level.</div>
    </section>
  );
}

function externalImpactKey(impact: ExternalImpact) {
  return `${impact.sourceCandidateId}:${impact.queryTokenIndex}:${impact.codeTokenIndex}`;
}

function computeDragMatches(
  graph: VisualizationGraph,
  candidate: CandidateDetail,
  session: SessionPayload,
  positions: Record<string, { x: number; y: number }>,
  draggedNode: GraphNode,
  targetIds: string[] = []
) {
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const positionOf = (node: GraphNode) => positions[node.id] ?? { x: node.x, y: node.y };
  const originalPositionOf = (node: GraphNode) => ({ x: node.x, y: node.y });
  const semantic = new Map<string, number>();
  const semanticThreshold = graph.semanticLinkThreshold ?? 0.42;
  (graph.semanticLinks ?? []).forEach((link) => {
    const queryId = link.sourceType === "query_token" ? link.source : link.target;
    const codeId = link.sourceType === "code_token" ? link.source : link.target;
    semantic.set(`${queryId}:${codeId}`, Math.max(semantic.get(`${queryId}:${codeId}`) ?? 0, link.similarity));
  });
  const queryNodes = graph.nodes.filter((node) => node.type === "query_token" && Boolean(graphTokenLabel(node.label)));
  const codeNodes = graph.nodes.filter((node) => node.type === "code_token" && Boolean(graphTokenLabel(node.label)));
  const oppositeNodes = draggedNode.type === "query_token" ? codeNodes : queryNodes;
  const distances = oppositeNodes.map((node) => {
    const a = originalPositionOf(draggedNode);
    const b = originalPositionOf(node);
    return Math.hypot(a.x - b.x, a.y - b.y);
  }).sort((a, b) => a - b);
  const localScale = Math.max(40, distances[Math.floor(distances.length / 2)] ?? 120);
  const proximity = (a: GraphNode, b: GraphNode, original = false) => {
    const first = original ? originalPositionOf(a) : positionOf(a);
    const second = original ? originalPositionOf(b) : positionOf(b);
    return 1 / (1 + Math.hypot(first.x - second.x, first.y - second.y) / localScale);
  };
  const basePairSimilarity = (conceptId: number, queryIndex: number, codeIndex: number) => {
    const semanticBase = semantic.get(`q_tok_${queryIndex}:c_tok_${codeIndex}`);
    if (semanticBase != null) return semanticBase;
    return candidate.conceptMatches
      .filter((match) => match.conceptId === conceptId && match.queryTokenIndices.includes(queryIndex) && match.codeTokenIndices.includes(codeIndex))
      .reduce((best, match) => Math.max(best, match.similarity), 0);
  };
  const softmaxMean = (values: number[], temperature = 0.12) => {
    if (!values.length) return 0;
    const maxValue = Math.max(...values);
    const weights = values.map((value) => Math.exp((value - maxValue) / temperature));
    const total = weights.reduce((sum, value) => sum + value, 0);
    return values.reduce((sum, value, index) => sum + value * weights[index], 0) / total;
  };
  const tokenMatches: DragTokenMatch[] = [];
  const lineMatches: DragLineMatch[] = [];
  const pairInterventionMap = new Map<string, {
    queryTokenIndex: number;
    codeTokenIndex: number;
    originalProximity: number;
    currentProximity: number;
    proximityDelta: number;
    modelCosine: number;
  }>();
  const targetIdSet = new Set(targetIds);
  session.query.concepts.forEach((concept) => {
    candidate.codeLines.forEach((line) => {
      const lineNodes = line.tokenIndices
        .map((idx) => nodeById.get(`c_tok_${idx}`))
        .filter((node): node is GraphNode => Boolean(node && graphTokenLabel(node.label)));
      if (!lineNodes.length) return;
      const pairScores = concept.tokenIndices.flatMap((queryIndex) => {
        const queryNode = nodeById.get(`q_tok_${queryIndex}`);
        if (!queryNode) return [];
        return lineNodes.map((codeNode) => {
          const base = basePairSimilarity(concept.conceptId, queryIndex, codeNode.tokenIndex);
          const currentProximity = proximity(queryNode, codeNode);
          const originalProximity = proximity(queryNode, codeNode, true);
          const moved = positions[queryNode.id] != null || positions[codeNode.id] != null;
          const score = moved ? clamp(base + 0.65 * (currentProximity - originalProximity), 0, 1) : base;
          const involvesDraggedNode =
            (draggedNode.type === "query_token" && draggedNode.tokenIndex === queryIndex)
            || (draggedNode.type === "code_token" && draggedNode.tokenIndex === codeNode.tokenIndex);
          const targetSelected = draggedNode.type === "query_token"
            ? queryIndex === draggedNode.tokenIndex && targetIdSet.has(codeNode.id)
            : codeNode.tokenIndex === draggedNode.tokenIndex && targetIdSet.has(queryNode.id);
          if (targetSelected && moved) {
            const proximityDelta = currentProximity - originalProximity;
            const key = `${queryIndex}:${codeNode.tokenIndex}`;
            const previous = pairInterventionMap.get(key);
            if (!previous || Math.abs(proximityDelta) > Math.abs(previous.proximityDelta)) {
              pairInterventionMap.set(key, {
                queryTokenIndex: queryIndex,
                codeTokenIndex: codeNode.tokenIndex,
                originalProximity,
                currentProximity,
                proximityDelta,
                modelCosine: base
              });
            }
          }
          return { queryIndex, codeIndex: codeNode.tokenIndex, score, base, color: concept.color };
        });
      });
      if (!pairScores.length) return;
      pairScores.sort((a, b) => b.score - a.score).slice(0, 3).forEach((pair) => {
        if (pair.score >= 0.68) {
          tokenMatches.push({
            queryTokenIndex: pair.queryIndex,
            codeTokenIndex: pair.codeIndex,
            similarity: pair.score,
            baseline: pair.base,
            delta: pair.score - pair.base,
            color: pair.color,
            newlyConnected: pair.base < semanticThreshold && pair.score >= 0.68
          });
        }
      });
      const score = softmaxMean(pairScores.map((pair) => pair.score));
      const baseline = softmaxMean(pairScores.map((pair) => pair.base));
      lineMatches.push({
        conceptId: concept.conceptId,
        lineNumber: line.lineNumber,
        previousLineNumber: line.lineNumber,
        similarity: score,
        baseline,
        delta: score - baseline,
        color: concept.color
      });
    });
  });
  const uniqueTokenMatches = Array.from(new Map(tokenMatches.map((match) => [`${match.queryTokenIndex}:${match.codeTokenIndex}`, match])).values())
    .sort((a, b) => Math.abs((b.delta ?? 0) || b.similarity) - Math.abs((a.delta ?? 0) || a.similarity))
    .slice(0, 4)
    .map((match, index) => ({ ...match, source: "local_drag" as const, role: index === 0 ? "dragged" as const : "follower" as const }));
  const lineDeltaByLine = new Map<number, number[]>();
  lineMatches.forEach((match) => lineDeltaByLine.set(match.lineNumber, [...(lineDeltaByLine.get(match.lineNumber) ?? []), match.delta]));
  const candidateDelta = 0.08 * Array.from(lineDeltaByLine.values()).reduce((sum, deltas) => sum + deltas.reduce((a, b) => a + b, 0) / deltas.length, 0);
  const demo = session.generalizationDemo;
  const demoQueryIndex = demo?.queryTokenIndices?.[0];
  const demoCodeIndex = demo?.codeTokenIndex;
  const demoQueryNode = demoQueryIndex != null ? nodeById.get(`q_tok_${demoQueryIndex}`) : undefined;
  const demoCodeNode = demoCodeIndex != null ? nodeById.get(`c_tok_${demoCodeIndex}`) : undefined;
  const demoInvolvesDraggedNode = Boolean(
    demo?.enabled
    && demo.interactionCandidateId === candidate.id
    && demoQueryNode
    && demoCodeNode
    && (
      (draggedNode.type === "query_token" && draggedNode.tokenIndex === demoQueryIndex)
      || (draggedNode.type === "code_token" && draggedNode.tokenIndex === demoCodeIndex)
    )
  );
  const demoPairIntervention = (() => {
    if (!demoInvolvesDraggedNode || !demoQueryNode || !demoCodeNode) return null;
    const originalQuery = originalPositionOf(demoQueryNode);
    const originalCode = originalPositionOf(demoCodeNode);
    const currentQuery = draggedNode.type === "query_token" ? positionOf(demoQueryNode) : originalQuery;
    const currentCode = draggedNode.type === "code_token" ? positionOf(demoCodeNode) : originalCode;
    const originalProximity = 1 / (1 + Math.hypot(originalQuery.x - originalCode.x, originalQuery.y - originalCode.y) / localScale);
    const currentProximity = 1 / (1 + Math.hypot(currentQuery.x - currentCode.x, currentQuery.y - currentCode.y) / localScale);
    return {
      queryTokenIndex: demoQueryNode.tokenIndex,
      codeTokenIndex: demoCodeNode.tokenIndex,
      originalProximity,
      currentProximity,
      proximityDelta: currentProximity - originalProximity,
      modelCosine: basePairSimilarity(demoQueryNode.conceptId ?? -1, demoQueryNode.tokenIndex, demoCodeNode.tokenIndex)
    };
  })();
  const pairInterventions = (demoPairIntervention ? [demoPairIntervention] : [...pairInterventionMap.values()])
    .filter((item) => Math.abs(item.proximityDelta) >= 0.01)
    .sort((a, b) => Math.abs(b.proximityDelta) - Math.abs(a.proximityDelta))
    .slice(0, 12);
  return { tokenMatches: uniqueTokenMatches, lineMatches, localScale, candidateDelta, pairInterventions };
}

function tokenColorStyle(concepts: Concept[], activeConceptIds: Set<number>, lineSelected: boolean) {
  if (!concepts.length) {
    return lineSelected ? { background: "#eef2f7", borderColor: "#9aa7b7" } : undefined;
  }
  const activeConcepts = concepts.filter((concept) => activeConceptIds.has(concept.conceptId));
  const visibleConcepts = activeConcepts.length ? activeConcepts : concepts;
  const colors = visibleConcepts.map((concept) => concept.color);
  const borderColor = colors[0];
  const alpha = activeConcepts.length || lineSelected ? "aa" : "55";
  if (colors.length === 1) {
    return { background: `${colors[0]}${alpha}`, borderColor };
  }
  const step = 100 / colors.length;
  const background = `linear-gradient(90deg, ${colors
    .map((color, idx) => `${color}${alpha} ${idx * step}%, ${color}${alpha} ${(idx + 1) * step}%`)
    .join(", ")})`;
  return { background, borderColor };
}

function buildGeneralizedVisualMatches(details: Record<string, unknown> | undefined, session: SessionPayload): Record<string, DragMatchBundle> {
  if (!details) return {};
  const conceptsById = new Map(session.query.concepts.map((concept) => [concept.conceptId, concept]));
  const result: Record<string, DragMatchBundle> = {};
  Object.entries(details).forEach(([candidateId, raw]) => {
    const item = raw as {
      matches?: Array<{ conceptId: number; queryTokenIndices: number[]; codeTokenIndices: number[]; lineNumber: number; similarity: number }>;
      originalMatches?: Array<{ conceptId: number; queryTokenIndices: number[]; codeTokenIndices: number[]; lineNumber: number; similarity: number }>;
      tokenPairDeltas?: Array<{
        conceptId: number;
        queryTokenIndex: number;
        queryToken?: string;
        codeTokenIndex: number;
        codeToken?: string;
        originalSimilarity: number;
        generalizedSimilarity: number;
        delta: number;
        lineNumber?: number;
      }>;
    };
    const originalByConcept = new Map((item.originalMatches ?? []).map((match) => [Number(match.conceptId), match]));
    const tokenMatches: DragTokenMatch[] = [];
    const lineMatches: DragLineMatch[] = [];
    (item.matches ?? []).forEach((match) => {
      const concept = conceptsById.get(Number(match.conceptId));
      if (!concept) return;
      const original = originalByConcept.get(Number(match.conceptId));
      const baseline = safeDisplaySimilarity(Number(original?.similarity ?? match.similarity));
      const similarity = safeDisplaySimilarity(Number(match.similarity), baseline);
      const delta = similarity - baseline;
      if (Math.abs(delta) < 0.01) return;
      lineMatches.push({
        conceptId: Number(match.conceptId),
        lineNumber: Number(match.lineNumber),
        previousLineNumber: original?.lineNumber != null ? Number(original.lineNumber) : Number(match.lineNumber),
        similarity,
        baseline,
        delta,
        color: concept.color,
        source: "generalized"
      });
    });
    (item.tokenPairDeltas ?? []).forEach((pair) => {
      const concept = conceptsById.get(Number(pair.conceptId));
      if (!concept) return;
      const baseline = safeDisplaySimilarity(Number(pair.originalSimilarity));
      const similarity = safeDisplaySimilarity(Number(pair.generalizedSimilarity), baseline);
      const delta = similarity - baseline;
      if (Math.abs(delta) < 0.005) return;
      tokenMatches.push({
        queryTokenIndex: Number(pair.queryTokenIndex),
        codeTokenIndex: Number(pair.codeTokenIndex),
        queryToken: pair.queryToken,
        codeToken: pair.codeToken,
        similarity,
        baseline,
        delta,
        color: concept.color,
        newlyConnected: delta > 0,
        source: "generalized",
        role: "generalized"
      });
    });
    result[candidateId] = {
      tokenMatches: tokenMatches
        .sort((a, b) => Math.abs(b.delta ?? 0) - Math.abs(a.delta ?? 0))
        .slice(0, 8),
      lineMatches: lineMatches
        .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
        .slice(0, 3)
    };
  });
  return result;
}

function buildExternalImpacts(
  details: Record<string, unknown> | undefined,
  sourceCandidateId: string,
  targetIds: string[],
  session: SessionPayload
): Record<string, ExternalImpact[]> {
  if (!details) return {};
  const targetQueryTokenIndices = targetIds
    .filter((id) => id.startsWith("q_tok_"))
    .map((id) => Number(id.replace("q_tok_", "")))
    .filter((index) => Number.isInteger(index) && isSemanticToken(session.query.tokens[index]));
  const impacts: Record<string, ExternalImpact[]> = {};
  Object.entries(details).forEach(([candidateId, raw]) => {
    if (candidateId === sourceCandidateId) return;
    const item = raw as { tokenPairDeltas?: Array<{ queryTokenIndex: number; codeTokenIndex: number; delta: number; conceptId?: number; queryToken?: string; codeToken?: string }> };
    const entries = (item.tokenPairDeltas ?? [])
      .filter((pair) => isSemanticToken(pair.queryToken ?? session.query.tokens[Number(pair.queryTokenIndex)]) && isSemanticToken(pair.codeToken))
      .filter((pair) => Math.abs(Number(pair.delta ?? 0)) >= 0.005)
      .map((pair) => ({
        queryTokenIndex: Number(pair.queryTokenIndex),
        codeTokenIndex: Number(pair.codeTokenIndex),
        delta: Number(pair.delta),
        color: session.query.concepts.find((concept) => concept.conceptId === Number(pair.conceptId))?.color ?? "#7c3aed",
        sourceCandidateId,
        targetQueryTokenIndices: [Number(pair.queryTokenIndex)]
      }))
      .reduce<ExternalImpact[]>((unique, entry) => {
        const key = `${entry.queryTokenIndex}:${entry.codeTokenIndex}`;
        const previousIndex = unique.findIndex((item) => `${item.queryTokenIndex}:${item.codeTokenIndex}` === key);
        if (previousIndex < 0) unique.push(entry);
        else if (Math.abs(entry.delta) > Math.abs(unique[previousIndex].delta)) unique[previousIndex] = entry;
        return unique;
      }, [])
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
      .slice(0, 12);
    if (entries.length) impacts[candidateId] = entries;
  });
  return impacts;
}

function nodeColors(node: GraphNode) {
  return node.colors?.length ? node.colors : [node.color];
}

function nodeFill(node: GraphNode, selectedConceptSet: Set<number>) {
  const activeConceptId = node.conceptIds.find((id) => selectedConceptSet.has(id));
  if (activeConceptId != null) {
    const colorIndex = node.conceptIds.indexOf(activeConceptId);
    return node.colors?.[colorIndex] ?? node.color;
  }
  return nodeColors(node).length > 1 ? `url(#node_grad_${node.id})` : node.color;
}

function QueryPanel({
  session,
  candidate,
  selectedConcepts,
  selectedTokenIds,
  dragTokenMatches,
  dragLineMatches,
  selectedDragMatchKey,
  onConcept,
  onToken,
  onDragMatch,
  externalImpacts,
  selectedExternalImpactKey,
  onExternalImpact,
  queryPointMode
}: {
  session: SessionPayload | null;
  candidate: CandidateDetail | null;
  selectedConcepts: number[];
  selectedTokenIds: string[];
  dragTokenMatches: DragTokenMatch[];
  dragLineMatches: DragLineMatch[];
  selectedDragMatchKey: string | null;
  onConcept: (id: number) => void;
  onToken: (id: string) => void;
  onDragMatch: (match: DragTokenMatch) => void;
  externalImpacts: ExternalImpact[];
  selectedExternalImpactKey: string | null;
  onExternalImpact: (impact: ExternalImpact, conceptId?: number) => void;
  queryPointMode: QueryPointMode;
}) {
  const selectedConceptSet = useMemo(() => new Set(selectedConcepts), [selectedConcepts]);
  const selectedTokenSet = useMemo(() => new Set(selectedTokenIds), [selectedTokenIds]);
  const tokenConcepts = useMemo(() => {
    const map = new Map<number, Concept[]>();
    session?.query.concepts.forEach((concept) => {
      concept.tokenIndices.forEach((idx) => {
        const list = map.get(idx) ?? [];
        list.push(concept);
        map.set(idx, list);
      });
    });
    return map;
  }, [session]);

  if (!session) return <aside className="panel muted">No issue loaded.</aside>;
  const hasConceptFocus = selectedConcepts.length > 0 || selectedTokenIds.some((id) => id.startsWith("q_tok_"));
  const showLocalDragDiagnostics = externalImpacts.length === 0;
  const lineTextByNumber = new Map((candidate?.codeLines ?? []).map((line) => [line.lineNumber, line.text]));
  const conceptById = new Map(session.query.concepts.map((concept) => [concept.conceptId, concept]));
  const externalImpactDisplayItems = useMemo(() => {
    if (queryPointMode === "tokens") {
      return externalImpacts.map((impact) => ({ impact, concept: null }));
    }
    const grouped = new Map<string, { impact: ExternalImpact; concept: Concept }>();
    externalImpacts.forEach((impact) => {
      const concepts = session.query.concepts.filter((concept) => concept.tokenIndices.includes(impact.queryTokenIndex));
      concepts.forEach((concept) => {
        const key = `${concept.conceptId}:${impact.codeTokenIndex}`;
        const existing = grouped.get(key);
        if (!existing || Math.abs(impact.delta) > Math.abs(existing.impact.delta)) grouped.set(key, { impact, concept });
      });
    });
    return [...grouped.values()].sort((first, second) => Math.abs(second.impact.delta) - Math.abs(first.impact.delta));
  }, [externalImpacts, queryPointMode, session.query.concepts]);
  return (
    <aside className="panel query-panel">
      <div className="panel-title">Query</div>
      <div className="query-text">{session.query.rawText}</div>
      <div className="query-token-list">
        {session.query.tokens.map((token, index) => {
          const visibleToken = displayToken(token);
          if (!visibleToken) return null;
          const concepts = tokenConcepts.get(index) ?? [];
          const tokenId = `q_tok_${index}`;
          const active = selectedTokenSet.has(tokenId) || concepts.some((concept) => selectedConceptSet.has(concept.conceptId));
          const concept = concepts[0];
          return (
            <button
              key={`${token}_${index}`}
              className={`${active ? "query-token active" : "query-token"}${hasConceptFocus && !active ? " dimmed" : ""}`}
              style={concept ? { borderColor: concept.color, background: active ? `${concept.color}aa` : `${concept.color}33` } : undefined}
              onClick={() => onToken(tokenId)}
            >
              {visibleToken}
            </button>
          );
        })}
      </div>
      <div className="issue-concept-title">Issue Concepts</div>
      <div className="concept-list">
        {session.query.concepts.map((concept) => (
          <button
            key={concept.id}
            className={selectedConceptSet.has(concept.conceptId) ? "concept active" : "concept"}
            style={{ borderColor: concept.color, background: `${concept.color}55` }}
            onClick={() => onConcept(concept.conceptId)}
          >
            <span>{displayConceptText(concept.text) || "(empty concept)"}</span>
          </button>
        ))}
      </div>
      {showLocalDragDiagnostics && (dragLineMatches.length || dragTokenMatches.length) ? (
        <div className="drag-change-panel">
          <div className="panel-title">{session.model?.id === "codebert" ? "Representation-level What-if" : "Drag Similarity Explanation"}</div>
          {session.model?.id === "codebert" ? <div className="meta-line">Counterfactual update over frozen contextual token states; ranks use the re-aggregated adapter representation.</div> : null}
          {dragLineMatches.length ? (
            <div className="drag-change-section">
              <div className="drag-change-caption">Concept-Line Similarity Changes</div>
              <div className="drag-change-list compact">
                {dragLineMatches.slice(0, 4).map((match) => {
                  const concept = conceptById.get(match.conceptId);
                  const delta = match.delta ?? 0;
                  const lineText = lineTextByNumber.get(match.lineNumber) ?? `line ${match.lineNumber}`;
                  return (
                    <div key={`${match.conceptId}_${match.lineNumber}`} className="drag-line-entry">
                      <span className="drag-change-pair">
                        <strong>{displayToken(concept?.text) || `concept ${match.conceptId + 1}`}</strong>
                        <span>{"->"}</span>
                        <strong>{displayQueryOriginal(lineText)}</strong>
                      </span>
                      <span className="drag-change-score">
                        {safeDisplaySimilarity(match.baseline).toFixed(3)} -&gt; {safeDisplaySimilarity(match.similarity, match.baseline).toFixed(3)}
                      </span>
                      <span className={delta >= 0 ? "drag-change-delta positive" : "drag-change-delta negative"}>
                        {delta >= 0 ? "+" : ""}{delta.toFixed(3)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}
          {dragTokenMatches.length ? (
          <div className="drag-change-section">
          <div className="drag-change-caption">Token-Pair Cosine Diagnostics</div>
          <div className="drag-change-list">
            {dragTokenMatches.filter((match) => isSemanticToken(match.queryToken ?? session.query.tokens[match.queryTokenIndex]) && isSemanticToken(match.codeToken ?? candidate?.codeTokens[match.codeTokenIndex])).slice(0, 10).map((match) => {
              const key = dragMatchKey(match);
              const queryToken = displayToken(match.queryToken ?? session.query.tokens[match.queryTokenIndex]);
              const codeToken = displayToken(match.codeToken ?? candidate?.codeTokens[match.codeTokenIndex]);
              const delta = match.delta ?? 0;
              const baseline = match.baseline;
              return (
                <button
                  key={key}
                  className={selectedDragMatchKey === key ? "drag-change-entry active" : "drag-change-entry"}
                  onClick={() => onDragMatch(match)}
                >
                  <span className="drag-change-pair">
                    <strong>{queryToken || `q${match.queryTokenIndex}`}</strong>
                    <span>{"->"}</span>
                    <strong>{codeToken || `c${match.codeTokenIndex}`}</strong>
                  </span>
                  <span className="drag-change-score">
                    {baseline != null ? `${safeDisplaySimilarity(baseline).toFixed(3)} -> ` : ""}
                    {safeDisplaySimilarity(match.similarity, baseline ?? 0).toFixed(3)}
                    <small>{match.source === "generalized" ? " generalized" : " local"}</small>
                  </span>
                  <span className={delta >= 0 ? "drag-change-delta positive" : "drag-change-delta negative"}>
                    {delta >= 0 ? "+" : ""}{delta.toFixed(3)}
                  </span>
                </button>
              );
            })}
          </div>
          </div>
          ) : null}
        </div>
      ) : null}
      {externalImpacts.length ? (
        <div className="drag-change-panel external-impact-panel">
          <div className="panel-title">External Representation Changes</div>
          <div className="drag-change-caption">{queryPointMode === "concept" ? "Affected code tokens by issue concept" : "Affected code tokens by issue token"}</div>
          <div className="drag-change-list">
            {externalImpactDisplayItems.slice(0, 10).map(({ impact, concept }) => {
              const key = externalImpactKey(impact);
              const queryToken = concept ? displayConceptText(concept.text) : displayToken(session.query.tokens[impact.queryTokenIndex]);
              const codeToken = displayToken(candidate?.codeTokens[impact.codeTokenIndex]);
              return (
                <button
                  key={concept ? `${concept.conceptId}:${key}` : key}
                  className={selectedExternalImpactKey === key ? "drag-change-entry active external-impact-entry" : "drag-change-entry external-impact-entry"}
                  onClick={() => onExternalImpact(impact, concept?.conceptId)}
                >
                  <span className="drag-change-pair">
                    <strong>{queryToken || `q${impact.queryTokenIndex}`}</strong>
                    <span>-&gt;</span>
                    <strong>{codeToken || `c${impact.codeTokenIndex}`}</strong>
                  </span>
                  <span className="drag-change-score">code representation</span>
                  <span className={impact.delta >= 0 ? "drag-change-delta positive" : "drag-change-delta negative"}>
                    {impact.delta >= 0 ? "+" : ""}{impact.delta.toFixed(3)}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
      <div className="meta-line">{session.query.metadata.path}</div>
    </aside>
  );
}

function BaselineQueryPanel({ session }: { session: SessionPayload | null }) {
  return (
    <aside className="panel query-panel baseline-query-panel">
      <div className="panel-title">Query</div>
      {session ? (
        <>
          <div className="query-text">{session.query.rawText}</div>
          {session.query.metadata.path ? <div className="meta-line">{session.query.metadata.path}</div> : null}
        </>
      ) : <div className="meta-line">No issue loaded.</div>}
    </aside>
  );
}

function CandidatePanel({
  candidates,
  selectedId,
  onSelect,
  modelId,
  showGroundTruth = true,
  adjudicationMode = false,
  adjudicationIds,
  onAdjudicationToggle
}: {
  candidates: CandidateSummary[];
  selectedId: string | null;
  onSelect: (candidate: CandidateSummary) => void;
  modelId: string;
  showGroundTruth?: boolean;
  adjudicationMode?: boolean;
  adjudicationIds: string[];
  onAdjudicationToggle: (candidate: CandidateSummary) => void;
}) {
  const candidateListRef = useRef<HTMLDivElement | null>(null);
  // Rerank responses update each candidate's displayed rank, but not every
  // response path guarantees that the array itself is returned in that same
  // order. Keep presentation order derived from rank so React cannot retain a
  // former Rank 1 at the leading edge after it has moved to Rank 2.
  const orderedCandidates = useMemo(() => candidates
    .map((candidate, index) => ({ candidate, index }))
    .sort((first, second) => first.candidate.rank - second.candidate.rank
      || second.candidate.similarity - first.candidate.similarity
      || first.index - second.index)
    .map(({ candidate }) => candidate), [candidates]);
  const leadingCandidateId = orderedCandidates[0]?.id ?? null;

  useEffect(() => {
    // Browsers preserve the old first card as a horizontal scroll anchor when
    // React inserts a newly promoted Rank 1 before it. Always reveal the new
    // leader instead of leaving it just outside the viewport on the left.
    candidateListRef.current?.scrollTo({ left: 0, behavior: "smooth" });
  }, [leadingCandidateId]);

  return (
    <aside className="panel candidate-panel">
      <div className="panel-title">Ranked Candidates</div>
      <div ref={candidateListRef} className="candidate-list">
        {orderedCandidates.map((candidate) => (
          <div key={candidate.id} className="candidate-row">
            <button
              className={(adjudicationMode ? adjudicationIds.includes(candidate.id) : selectedId === candidate.id) ? "candidate active" : "candidate"}
              onClick={() => adjudicationMode ? onAdjudicationToggle(candidate) : onSelect(candidate)}
            >
            <span className="rank">#{candidate.rank}</span>
            <span className="candidate-main">
              <strong title={candidate.metadata.funcName || candidate.id}>{candidate.metadata.funcName || candidate.id}</strong>
              {candidate.originalRank != null && candidate.originalRank !== candidate.rank ? (
                <small className={candidate.rankDelta && candidate.rankDelta > 0 ? "delta up" : "delta neutral"}>
                  {candidate.rankDelta && candidate.rankDelta > 0 ? <ArrowUp size={12} /> : candidate.rankDelta && candidate.rankDelta < 0 ? <ArrowDown size={12} /> : null}
                  {`Rank ${candidate.originalRank} -> ${candidate.rank} (delta ${candidate.rankDelta && candidate.rankDelta > 0 ? "+" : ""}${candidate.rankDelta ?? 0})`}
                </small>
              ) : null}
              {showGroundTruth && modelId !== "codebert" && candidate.isGroundTruth ? <small className="ground-truth-label">Ground truth</small> : null}
            </span>
            <span className="score" title={`similarity ${safeDisplaySimilarity(candidate.similarity).toFixed(3)}`}>{safeDisplaySimilarity(candidate.similarity).toFixed(3)}</span>
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}

function AdjudicationPanel({
  session,
  candidates,
  summaries,
  showGroundTruth = true,
  showEvidence = true,
  showUncoveredEvidence = false,
  choosingReference = false,
  onChooseReference
}: {
  session: SessionPayload;
  candidates: CandidateDetail[];
  summaries: Record<string, CandidateSummary>;
  showGroundTruth?: boolean;
  showEvidence?: boolean;
  showUncoveredEvidence?: boolean;
  choosingReference?: boolean;
  onChooseReference: (candidate: CandidateDetail) => void;
}) {
  const conceptById = useMemo(() => new Map(session.query.concepts.map((concept) => [concept.conceptId, concept])), [session.query.concepts]);
  if (candidates.length !== 2) return null;
  return (
    <section className="panel adjudication-panel">
      <div className="adjudication-header">
        <div>
          <div className="panel-title">Candidate Adjudication</div>
          <small>{showEvidence ? (showUncoveredEvidence ? "Compare line-level concept evidence and uncovered implementation details." : "Compare line-level concept evidence before selecting one for generation.") : "Compare the two code references before selecting one for generation."}</small>
        </div>
      </div>
      <div className="adjudication-columns">
        {candidates.map((candidate) => {
          const summary = summaries[candidate.id];
          const lines = candidate.codeLines.filter((line) => line.tokenIndices.length > 0);
          return (
            <article key={candidate.id} className={`adjudication-candidate${showGroundTruth && session.model?.id !== "codebert" && summary?.isGroundTruth ? " ground-truth" : ""}`}>
              <div className="adjudication-candidate-title">
                <div>
                  <strong>{summary?.metadata.funcName || candidate.metadata.funcName || candidate.id}</strong>
                  <span>Rank {summary?.rank ?? "-"} · sim {safeDisplaySimilarity(summary?.similarity ?? candidate.similarity).toFixed(3)}</span>
                </div>
                <button className="adjudication-reference-action" onClick={() => onChooseReference(candidate)} disabled={choosingReference}><BookOpenCheck size={13} /> Use as reference</button>
              </div>
              <div className="adjudication-lines">
                {lines.map((line, index) => {
                  const matches = candidate.conceptMatches.filter((match) => match.codeTokenIndices.some((tokenIndex) => line.tokenIndices.includes(tokenIndex)));
                  const uncovered = showEvidence && showUncoveredEvidence && isAdjudicationUncoveredLine(line, matches);
                  const indentation = line.text.match(/^\s*/)?.[0].length ?? 0;
                  const matchColors = matches
                    .map((match) => conceptById.get(Number(match.conceptId))?.color)
                    .filter((color): color is string => Boolean(color));
                  return (
                    <div
                      key={line.lineNumber}
                      className={`adjudication-line${showEvidence && matches.length ? " aligned" : ""}${uncovered ? " uncovered" : ""}`}
                      style={showEvidence && matchColors.length && !uncovered ? {
                        borderLeftColor: matchColors[0],
                        background: matchColors.length === 1
                          ? `${matchColors[0]}1f`
                          : `linear-gradient(90deg, ${matchColors.map((color) => `${color}24`).join(", ")})`
                      } : undefined}
                    >
                      <span className="adjudication-line-number">L{index + 1}</span>
                      <code style={{ "--adjudication-indent": `${indentation}ch` } as React.CSSProperties}>{line.text.trimStart()}</code>
                      <span className="adjudication-evidence">
                        {uncovered ? <span className="adjudication-negative">uncovered by query concepts</span> : null}
                      </span>
                    </div>
                  );
                })}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function DiagnosticsPanel({
  graph,
  queryText,
  selectedTokenIds,
  selectedManualLinkId,
  manualLinks,
  neighborSignals,
  exitedNeighbors,
  attribution,
  attributionLoading,
  attributionError,
  gradientAttribution,
  gradientLoading,
  gradientError,
  crossSampleBatchLimit,
  onCrossSampleBatchLimit,
  onGradientTrace,
  onManualLink
}: {
  graph: VisualizationGraph | null;
  queryText: string;
  selectedTokenIds: string[];
  selectedManualLinkId: string | null;
  manualLinks: ManualLink[];
  neighborSignals: Record<string, NeighborSignal>;
  exitedNeighbors: string[];
  attribution: TokenPairAttribution | null;
  attributionLoading: boolean;
  attributionError: string | null;
  gradientAttribution: GradientAttribution | null;
  gradientLoading: boolean;
  gradientError: string | null;
  crossSampleBatchLimit: number;
  onCrossSampleBatchLimit: (limit: number) => void;
  onGradientTrace: (pair: { queryNode: GraphNode; codeNode: GraphNode }, lossScope?: GradientAttribution["lossScope"]) => void;
  onManualLink: (link: ManualLink) => void;
}) {
  const selectedNodes = useMemo(() => {
    if (!graph) return [];
    const ids = new Set(selectedTokenIds);
    return graph.nodes.filter((node) => ids.has(node.id));
  }, [graph, selectedTokenIds]);

  const pairs = useMemo(() => {
    const result = [];
    for (let i = 0; i < selectedNodes.length; i += 1) {
      for (let j = i + 1; j < selectedNodes.length; j += 1) {
        const distance = nodeDistance(selectedNodes[i], selectedNodes[j]);
        result.push({
          id: `${selectedNodes[i].id}_${selectedNodes[j].id}`,
          a: selectedNodes[i],
          b: selectedNodes[j],
          distance,
          similarity: projectionSimilarity(distance)
        });
      }
    }
    return result.sort((a, b) => a.distance - b.distance).slice(0, 6);
  }, [selectedNodes]);

  const neighbors = useMemo(() => nearestNeighbors(graph, selectedNodes[0]?.id ?? null, 5), [graph, selectedNodes]);
  const selectedPair = useMemo(() => selectedQueryCodePair(graph, selectedTokenIds), [graph, selectedTokenIds]);

  return (
    <section className="panel diagnostics-panel">
      <div className="panel-title">Token Diagnostics</div>
      <div className="diagnostic-grid">
        <div>
          <div className="diagnostic-label">Selected</div>
          <div className="chip-row">
            {selectedNodes.length ? selectedNodes.map((node) => <span key={node.id} className="small-chip">{displayToken(node.label)}</span>) : <span className="meta-line">No token selected</span>}
          </div>
        </div>
        <div>
          <div className="diagnostic-label">Manual Links</div>
          <div className="chip-row">
            {manualLinks.length ? manualLinks.slice(-3).map((link) => (
              <button
                key={link.id}
                className={selectedManualLinkId === link.id ? "small-chip link-chip active" : "small-chip link-chip"}
                style={{ borderColor: link.color, background: `${link.color}22` }}
                onClick={() => onManualLink(link)}
              >
                {`${displayToken(link.queryToken)} -> ${displayToken(link.codeToken)} · ${safeDisplaySimilarity(link.similarity).toFixed(2)}`}
              </button>
            )) : <span className="meta-line">No manual link</span>}
          </div>
        </div>
      </div>
      <div className="diagnostic-columns">
        <div>
          <div className="diagnostic-label">Pairwise Distance</div>
          {pairs.length ? pairs.map((pair) => (
            <div key={pair.id} className="metric-row">
              <span>{displayToken(pair.a.label)} / {displayToken(pair.b.label)}</span>
              <strong>{pair.distance.toFixed(1)} · sim {safeDisplaySimilarity(pair.similarity).toFixed(2)}</strong>
            </div>
          )) : <div className="meta-line">Select at least two tokens.</div>}
        </div>
        <div>
          <div className="diagnostic-label">Nearest Neighbors</div>
          {neighbors.length ? neighbors.map((item) => (
            <div key={item.node.id} className={`metric-row ${neighborSignals[item.node.id]?.status ?? ""}`}>
              <span>{displayToken(item.node.label)}</span>
              <strong>
                {neighborSignals[item.node.id]?.status && neighborSignals[item.node.id].status !== "stable" ? `${neighborSignals[item.node.id].status} ` : ""}
                {item.distance.toFixed(1)} · sim {safeDisplaySimilarity(item.similarity).toFixed(2)}
              </strong>
            </div>
          )) : <div className="meta-line">Select one token.</div>}
          {exitedNeighbors.length ? <div className="neighbor-exit">Exited: {exitedNeighbors.join(", ")}</div> : null}
        </div>
      </div>
      <TrainingTracePanel
        selectedPair={selectedPair}
        attribution={attribution}
        loading={attributionLoading}
        error={attributionError}
        gradientAttribution={gradientAttribution}
        gradientLoading={gradientLoading}
        gradientError={gradientError}
        queryText={queryText}
        crossSampleBatchLimit={crossSampleBatchLimit}
        onCrossSampleBatchLimit={onCrossSampleBatchLimit}
        onGradientTrace={onGradientTrace}
      />
    </section>
  );
}

function TrainingTracePanel({
  selectedPair,
  queryText,
  attribution,
  loading,
  error,
  gradientAttribution,
  gradientLoading,
  gradientError,
  crossSampleBatchLimit,
  onCrossSampleBatchLimit,
  onGradientTrace
}: {
  selectedPair: { queryNode: GraphNode; codeNode: GraphNode } | null;
  queryText: string;
  attribution: TokenPairAttribution | null;
  loading: boolean;
  error: string | null;
  gradientAttribution: GradientAttribution | null;
  gradientLoading: boolean;
  gradientError: string | null;
  crossSampleBatchLimit: number;
  onCrossSampleBatchLimit: (limit: number) => void;
  onGradientTrace: (pair: { queryNode: GraphNode; codeNode: GraphNode }, lossScope?: GradientAttribution["lossScope"]) => void;
}) {
  const pairMatchesAttribution =
    selectedPair &&
    attribution &&
    attribution.selectedPair.queryTokenIndex === selectedPair.queryNode.tokenIndex &&
    attribution.selectedPair.codeTokenIndex === selectedPair.codeNode.tokenIndex;
  const data = pairMatchesAttribution ? attribution : null;
  const support = data?.trainingEvidence.supportingSamples ?? [];
  const conflict = data?.trainingEvidence.conflictingSamples ?? [];
  return (
    <div className="training-trace">
      <div className="diagnostic-label">Training Trace</div>
      {!selectedPair ? (
        <div className="meta-line">Select one query token and one code token to inspect model evidence and training evidence.</div>
      ) : loading ? (
        <div className="trace-status"><Loader2 size={14} className="spin" /> Loading token-pair evidence</div>
      ) : error ? (
        <div className="trace-error">{error}</div>
      ) : data ? (
        <>
          <div className="trace-summary">
            <span className="small-chip">{displayToken(data.selectedPair.queryToken)}</span>
            <span className="trace-arrow">-&gt;</span>
            <span className="small-chip">{displayToken(data.selectedPair.codeToken)}</span>
            <strong>cos {data.selectedPair.modelCosine.toFixed(3)}</strong>
            {data.selectedPair.projectionDistance != null ? <strong>dist {data.selectedPair.projectionDistance.toFixed(1)}</strong> : null}
            {data.candidateContext.rank ? <strong>rank #{data.candidateContext.rank}</strong> : null}
            <button className="trace-action" onClick={() => selectedPair && onGradientTrace(selectedPair, "highlight_only")} disabled={gradientLoading}>
              {gradientLoading ? <Loader2 size={13} className="spin" /> : null}
              Run Gradient Trace
            </button>
            <button className="trace-action trace-action-secondary" onClick={() => selectedPair && onGradientTrace(selectedPair, "highlight_plus_cross_sample_batch")} disabled={gradientLoading}>
              {gradientLoading ? <Loader2 size={13} className="spin" /> : null}
              Run Cross-Sample Trace
            </button>
            <label className="batch-limit-control">
              <span>batches</span>
              <select value={crossSampleBatchLimit} onChange={(event) => onCrossSampleBatchLimit(Number(event.target.value))} disabled={gradientLoading}>
                <option value={3}>3</option>
                <option value={8}>8</option>
                <option value={16}>16</option>
              </select>
            </label>
          </div>
          <div className="trace-metrics">
            <div>
              <span>Query highlight</span>
              <strong className={data.selectedPair.queryHighlighted ? "trace-good" : "trace-muted"}>
                {data.selectedPair.queryHighlightScore != null ? data.selectedPair.queryHighlightScore.toFixed(3) : "n/a"}
              </strong>
            </div>
            <div>
              <span>Code highlight</span>
              <strong className={data.selectedPair.codeHighlighted ? "trace-good" : "trace-muted"}>
                {data.selectedPair.codeHighlightScore != null ? data.selectedPair.codeHighlightScore.toFixed(3) : "n/a"}
              </strong>
            </div>
            <div>
              <span>Training cache</span>
              <strong>{data.trainingEvidence.processedSamples}/{data.trainingEvidence.sampleLimit}</strong>
            </div>
          </div>
          <div className="trace-lists">
            <EvidenceList title="Supporting" samples={support} tone="support" />
            <EvidenceList title="Conflicting" samples={conflict} tone="conflict" />
          </div>
          <GradientTraceResult gradientAttribution={gradientAttribution} loading={gradientLoading} error={gradientError} />
          <div className="meta-line">{data.diagnosis.evidenceCaveat}</div>
        </>
      ) : (
        <div className="meta-line">Waiting for attribution result.</div>
      )}
    </div>
  );
}

function GradientTraceResult({
  gradientAttribution,
  loading,
  error
}: {
  gradientAttribution: GradientAttribution | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading) {
    return <div className="trace-status gradient-status"><Loader2 size={14} className="spin" /> Computing gradient influence</div>;
  }
  if (error) return <div className="trace-error gradient-status">{error}</div>;
  if (!gradientAttribution) return null;
  if (gradientAttribution.status !== "ok") {
    return <div className="meta-line gradient-status">{gradientAttribution.message || "Gradient trace is not available."}</div>;
  }
  const batchMode = gradientAttribution.lossScope === "highlight_plus_cross_sample_batch";
  return (
    <div className="gradient-trace">
      <div className="gradient-head">
        <span>{batchMode ? "Cross-Sample Batch Influence" : "Gradient Influence"}</span>
        <strong>{gradientAttribution.queryLoss?.type} · cos {gradientAttribution.queryLoss?.modelCosine.toFixed(3)}</strong>
        <small>
          {gradientAttribution.scope?.lossScope} · {gradientAttribution.scope?.evaluatedTrainSamples} samples
          {gradientAttribution.scope?.evaluatedBatches ? ` · ${gradientAttribution.scope.evaluatedBatches} batches` : ""}
          {gradientAttribution.scope?.cacheHits != null ? ` · cache ${gradientAttribution.scope.cacheHits}/${gradientAttribution.scope.evaluatedBatches ?? 0}` : ""}
        </small>
      </div>
      <div className="trace-lists">
        {batchMode ? (
          <>
            <GradientBatchList title="Helpful Batches" batches={gradientAttribution.helpfulBatches ?? []} tone="support" />
            <GradientBatchList title="Harmful Batches" batches={gradientAttribution.harmfulBatches ?? []} tone="conflict" />
          </>
        ) : (
          <>
            <GradientList title="Helpful" samples={gradientAttribution.helpfulSamples} tone="support" />
            <GradientList title="Harmful" samples={gradientAttribution.harmfulSamples} tone="conflict" />
          </>
        )}
      </div>
      {gradientAttribution.scope?.influenceMethod ? <div className="meta-line gradient-method">{gradientAttribution.scope.influenceMethod}</div> : null}
    </div>
  );
}

function GradientList({
  title,
  samples,
  tone
}: {
  title: string;
  samples: GradientAttribution["helpfulSamples"];
  tone: "support" | "conflict";
}) {
  return (
    <div className="evidence-list">
      <div className={`evidence-title ${tone}`}>{title}</div>
      {samples.length ? samples.map((sample) => (
        <details key={`${tone}_grad_${sample.trainIndex}`} className="evidence-card gradient-card">
          <summary>
            <span>{sample.funcName || `train ${sample.trainIndex}`} · batch {sample.batchIndex}</span>
            <strong>{sample.influence.toExponential(2)}</strong>
          </summary>
          <div className="evidence-reason">
            loss {sample.loss.toFixed(3)} · nl {sample.nlHighlightLoss.toFixed(3)} · code {sample.codeHighlightLoss.toFixed(3)}
          </div>
          <div className="evidence-meta">
            grad dot {sample.gradientDot.toExponential(2)} · norm infl {sample.normalizedInfluence.toFixed(3)} · samples {sample.batchStart}-{sample.batchEnd}
          </div>
          <div className="evidence-match">
            <span>{sample.docstring}</span>
          </div>
          <div className="evidence-meta">{sample.path}</div>
        </details>
      )) : <div className="meta-line">No samples in this direction.</div>}
    </div>
  );
}

function AlignmentHit({
  queryOriginal,
  conceptText,
  codeText,
  meta,
  tone
}: {
  queryOriginal?: string;
  conceptText: string;
  codeText?: string;
  meta: string;
  tone: "support" | "conflict";
}) {
  const displayConcept = displayToken(conceptText);
  return (
    <div className={`alignment-hit ${tone}`}>
      {queryOriginal ? (
        <div className="alignment-original">
          <div className="alignment-label">Query original</div>
          <div>{displayQueryOriginal(queryOriginal)}</div>
        </div>
      ) : null}
      <div className="alignment-flow">
        <div>
          <div className="alignment-label">Query concept</div>
          <div className="alignment-text concept-text">{displayConcept || "n/a"}</div>
        </div>
        <div className="alignment-arrow">-&gt;</div>
        <div>
          <div className="alignment-label">Code step</div>
          <pre className="alignment-code inline">{codeText || "No code span recorded."}</pre>
        </div>
      </div>
      <div className="alignment-meta">{meta}</div>
    </div>
  );
}

function GradientBatchList({
  title,
  batches,
  tone
}: {
  title: string;
  batches: NonNullable<GradientAttribution["helpfulBatches"]>;
  tone: "support" | "conflict";
}) {
  const source = tone === "support" ? "support" : "conflict";
  const visibleBatchEntries = batches
    .map((batch) => ({
      batch,
      visibleHits: (batch.selection?.sampleHits ?? [])
        .filter((hit) => hit.source === source && isQuerySideEvidence(hit, tone))
        .slice(0, 5)
    }))
    .filter((entry) => entry.visibleHits.length > 0);
  return (
    <div className="evidence-list">
      <div className={`evidence-title ${tone}`}>{title}</div>
      <div className="evidence-column-note">
        {tone === "support"
          ? "Similar query-side concept and code step appear in the same annotated training alignment."
          : "Similar query-side concept is aligned to a different code expression in training data."}
      </div>
      {visibleBatchEntries.length ? visibleBatchEntries.map(({ batch, visibleHits }) => (
        (() => {
          const titleHit = visibleHits[0];
          const titleText = titleHit?.docstring ? displayQueryOriginal(titleHit.docstring) : titleHit?.conceptText || `batch ${batch.batchIndex}`;
          return (
            <details key={`${tone}_batch_${batch.batchIndex}`} className="evidence-card gradient-card">
              <summary>
                <span>{displayToken(titleText)} · batch {batch.batchIndex}</span>
                <strong>{batch.influence.toExponential(2)}</strong>
              </summary>
              <div className="evidence-reason">
                total {batch.loss.toFixed(3)} · highlight {batch.highlightLoss.toFixed(3)} · cross {batch.crossSampleLoss.toFixed(3)} x {batch.crossSampleWeight.toFixed(1)}
              </div>
              <div className="evidence-meta">
                grad dot {batch.gradientDot.toExponential(2)} · norm infl {batch.normalizedInfluence.toFixed(3)}
                {batch.cacheHit != null ? ` · ${batch.cacheHit ? "cache hit" : "cache miss"}` : ""}
              </div>
              {batch.selection ? (
                <div className="evidence-meta">
                  selected score {batch.selection.selectionScore.toFixed(3)} · hits {batch.selection.hitCount}
                  {tone === "support" ? ` · support ${batch.selection.supportCount}` : ` · conflict ${batch.selection.conflictCount}`}
                </div>
              ) : null}
              <div className="alignment-stack">
                {visibleHits.map((hit) => (
                  <AlignmentHit
                    key={`${batch.batchIndex}_${hit.trainIndex}_${hit.source}_${hit.score}_${hit.conceptText}`}
                    queryOriginal={hit.docstring}
                    conceptText={hit.conceptText}
                    codeText={hit.stepCode}
                    meta={`${hit.source} · train ${hit.trainIndex} · score ${hit.score.toFixed(2)} · ${hit.funcName || hit.path}`}
                    tone={hit.source === "support" ? "support" : "conflict"}
                  />
                ))}
              </div>
              <details className="batch-member-details">
                <summary>Batch members</summary>
                <div className="batch-samples">
                  {batch.sampleSummaries.slice(0, 6).map((sample) => (
                    <div key={`${batch.batchIndex}_${sample.trainIndex}`} className="batch-sample">
                      <strong>{sample.trainIndex}</strong>
                      <span>{sample.funcName || sample.path || "training sample"}</span>
                      <small>{displayQueryOriginal(sample.docstring)}</small>
                    </div>
                  ))}
                </div>
              </details>
            </details>
          );
        })()
      )) : <div className="meta-line">No query-side alignments in this direction.</div>}
    </div>
  );
}

function EvidenceList({
  title,
  samples,
  tone
}: {
  title: string;
  samples: TokenPairAttribution["trainingEvidence"]["supportingSamples"];
  tone: "support" | "conflict";
}) {
  const visibleSamples = samples.filter((sample) => isQuerySideEvidence(sample, tone)).slice(0, 4);
  return (
    <div className="evidence-list">
      <div className={`evidence-title ${tone}`}>{title}</div>
      <div className="evidence-column-note">
        {tone === "support"
          ? "Similar query-side concept and code step appear in the same annotated training alignment."
          : "Similar query-side concept is aligned to a different code expression in training data."}
      </div>
      {visibleSamples.length ? visibleSamples.map((sample) => (
        <details key={`${tone}_${sample.trainIndex}_${sample.conceptId}_${sample.stepName}`} className="evidence-card">
          <summary>
            <span>{sample.docstring ? displayQueryOriginal(sample.docstring) : displayToken(sample.funcName || `train ${sample.trainIndex}`)}</span>
            <strong>{sample.score.toFixed(2)}</strong>
          </summary>
          <AlignmentHit
            conceptText={sample.conceptText}
            codeText={sample.stepCode}
            meta={`train ${sample.trainIndex} · ${sample.funcName || sample.path}`}
            tone={tone}
          />
          <div className="evidence-meta">
            {sample.path}
          </div>
        </details>
      )) : <div className="meta-line">No query-side evidence in the current cache.</div>}
    </div>
  );
}

function CodeViewer({
  candidate,
  session,
  graph,
  canvasLevel,
  selectedBlockId,
  selectedConcepts,
  selectedLines,
  selectedTokenIds,
  manualLinks,
  dragTokenMatches,
  dragLineMatches,
  displaySimilarity,
  allowPlainText = false,
  onBlock,
  onLine,
  onToken
}: {
  candidate: CandidateDetail | null;
  session: SessionPayload | null;
  graph: VisualizationGraph | null;
  canvasLevel: CanvasLevel;
  selectedBlockId: string | null;
  selectedConcepts: number[];
  selectedLines: number[];
  selectedTokenIds: string[];
  manualLinks: ManualLink[];
  dragTokenMatches: DragTokenMatch[];
  dragLineMatches: DragLineMatch[];
  displaySimilarity?: number;
  allowPlainText?: boolean;
  onBlock: (blockId: string) => void;
  onLine: (lineNumber: number) => void;
  onToken: (id: string) => void;
}) {
  const [plainTextMode, setPlainTextMode] = useState(false);
  const selectedConceptSet = useMemo(() => new Set(selectedConcepts), [selectedConcepts]);
  const selectedLineSet = useMemo(() => new Set(selectedLines), [selectedLines]);
  const selectedTokenSet = useMemo(() => new Set(selectedTokenIds), [selectedTokenIds]);
  const tokenConcepts = useMemo(() => {
    const map = new Map<number, Concept[]>();
    candidate?.conceptMatches.forEach((match) => {
      match.codeTokenIndices.forEach((idx) => {
        const concepts = map.get(idx) ?? [];
        concepts.push({
          id: match.id,
          conceptId: match.conceptId,
          tokenIndices: match.codeTokenIndices,
          text: match.codeText,
          color: match.color
        });
        map.set(idx, concepts);
      });
    });
    return map;
  }, [candidate]);
  const manualByCodeToken = useMemo(() => {
    const map = new Map<number, ManualLink>();
    manualLinks.forEach((link) => map.set(link.codeTokenIndex, link));
    return map;
  }, [manualLinks]);
  const hierarchyBlocks = graph?.hierarchy?.blocks ?? [];
  const conceptList = session?.query.concepts ?? [];
  const blockWinners = useMemo(() => conceptWinnerMap(hierarchyBlocks, conceptList), [graph?.hierarchy?.blocks, session?.query.concepts]);
  const allHierarchyLines = useMemo(() => Object.values(graph?.hierarchy?.linesByBlock ?? {}).flat().map((line) => ({ ...line, id: `line_${line.lineNumber}` })), [graph?.hierarchy?.linesByBlock]);
  const recommendedTokenIndices = useMemo(
    () => AUTO_TOKEN_SUGGESTIONS_ENABLED ? new Set(graph?.hierarchy?.recommendedTokenIndices ?? []) : new Set<number>(),
    [graph?.hierarchy?.recommendedTokenIndices]
  );
  const conceptById = useMemo(() => new Map(conceptList.map((concept) => [concept.conceptId, concept])), [conceptList]);
  const selectedBlockHierarchyLines = useMemo(() => {
    if (!selectedBlockId) return [];
    return (graph?.hierarchy?.linesByBlock?.[selectedBlockId] ?? []).map((line) => ({ ...line, id: `line_${line.lineNumber}` }));
  }, [graph?.hierarchy?.linesByBlock, selectedBlockId]);
  const selectedBlockLineWinners = useMemo(
    () => conceptWinnerMap(selectedBlockHierarchyLines, conceptList),
    [selectedBlockHierarchyLines, conceptList]
  );
  const blockByLine = useMemo(() => {
    const map = new Map<number, NonNullable<VisualizationGraph["hierarchy"]>["blocks"][number]>();
    hierarchyBlocks.forEach((block) => block.lineNumbers.forEach((lineNumber) => map.set(lineNumber, block)));
    return map;
  }, [hierarchyBlocks]);

  const displayCodeLines = useMemo(() => {
    // The viewer is an alignment surface: omit comments, docstrings, and
    // source lines that have no model code-token representation.
    const nonEmptyLines = (candidate?.codeLines ?? []).filter((line) => line.tokenIndices.length > 0);
    if (candidate?.testId !== "48" || candidate.id !== "code_846") return nonEmptyLines;
    const compacted: CandidateDetail["codeLines"] = [];
    nonEmptyLines.forEach((line) => {
      const standaloneCloser = /^[\]\)}]+[,;:]?$/.test(line.text.trim());
      const previous = compacted[compacted.length - 1];
      if (standaloneCloser && previous) {
        previous.text = `${previous.text.trimEnd()}${line.text.trim()}`;
        previous.tokenIndices = [...previous.tokenIndices, ...line.tokenIndices];
        return;
      }
      if (!line.tokenIndices.length) return;
      compacted.push({ ...line, tokenIndices: [...line.tokenIndices] });
    });
    return compacted;
  }, [candidate]);
  const displayLineNumberByRaw = useMemo(() => new Map(displayCodeLines.map((line, index) => [line.lineNumber, index + 1])), [displayCodeLines]);

  if (!candidate) return <section className="panel code-viewer muted">No candidate selected.</section>;
  const hasTokenFocus = selectedConcepts.length > 0 || selectedLines.length > 0 || selectedTokenIds.some((id) => id.startsWith("c_tok_")) || manualLinks.length > 0;
  return (
    <section className="panel code-viewer">
      <div className="code-viewer-header">
        <div className="panel-title">Code Viewer</div>
        {allowPlainText ? (
          <div className="code-view-mode-toggle" role="group" aria-label="Code display mode">
            <button
              type="button"
              className={!plainTextMode ? "active" : ""}
              aria-pressed={!plainTextMode}
              onClick={() => setPlainTextMode(false)}
            >Highlighted</button>
            <button
              type="button"
              className={plainTextMode ? "active" : ""}
              aria-pressed={plainTextMode}
              onClick={() => setPlainTextMode(true)}
            >Plain Text</button>
          </div>
        ) : null}
      </div>
      <div className="code-meta">
        <span>{candidate.metadata.funcName}</span>
        <span>similarity {safeDisplaySimilarity(displaySimilarity ?? candidate.similarity).toFixed(3)}</span>
      </div>
      {plainTextMode && allowPlainText ? (
        <pre className="baseline-code-content code-viewer-plain-content">{withoutLeadingFunctionDocstring(candidate.rawCode)}</pre>
      ) : <div className="code-scroll">
        <div className="code-lines">
          {displayCodeLines.map((line) => {
            const lineSelected = selectedLineSet.has(line.lineNumber);
            const tokenConceptSelected = line.tokenIndices.some((idx) => {
              const concepts = tokenConcepts.get(idx) ?? [];
              return concepts.some((concept) => selectedConceptSet.has(concept.conceptId));
            });
            const block = blockByLine.get(line.lineNumber);
            const blockMatches = block ? blockWinners.get(block.id) ?? [] : [];
            const lineMatches = selectedBlockLineWinners.get(`line_${line.lineNumber}`) ?? [];
            const isHierarchyDetail = canvasLevel === "line" || canvasLevel === "line_tokens";
            const belongsToSelectedBlock = Boolean(selectedBlockId && block?.id === selectedBlockId);
            // The code viewer uses the same concept winner map as the hierarchy
            // canvas, so line backgrounds and line-node colors always agree.
            const hierarchyMatches = canvasLevel === "block"
              ? blockMatches
              : isHierarchyDetail && belongsToSelectedBlock
                ? lineMatches
                : [];
            const hierarchyLine = allHierarchyLines.find((item) => item.lineNumber === line.lineNumber);
            const blockSignals = block && Array.isArray(block.signals) ? block.signals : [];
            const blockLineSignals = block
              ? allHierarchyLines
                .filter((item) => block.lineNumbers.includes(item.lineNumber))
                .flatMap((item) => Array.isArray(item.signals) ? item.signals : [])
              : [];
            const blockSignalTexts = [...new Set(hierarchySignalTexts([...blockSignals, ...blockLineSignals], candidate, conceptById))];
            const signalTexts = canvasLevel === "block"
              ? blockSignalTexts
              : canvasLevel === "line"
                ? hierarchySignalTexts(hierarchyLine?.signals, candidate, conceptById)
                : [];
            const hierarchyColors = hierarchyMatches.map((match) => match.concept.color);
            const blockSelected = canvasLevel === "block" && block?.id === selectedBlockId;
            const conceptSelected = canvasLevel === "token" || canvasLevel === "line_tokens"
              ? tokenConceptSelected || hierarchyMatches.some((match) => selectedConceptSet.has(match.concept.conceptId))
              : hierarchyMatches.some((match) => selectedConceptSet.has(match.concept.conceptId));
            const visibleBlockLineNumbers = (block?.lineNumbers ?? []).filter((lineNumber) => displayLineNumberByRaw.has(lineNumber));
            const blockLineNumbers = visibleBlockLineNumbers.length ? visibleBlockLineNumbers : block?.lineNumbers ?? [];
            const blockStart = blockLineNumbers[0] === line.lineNumber;
            const blockEnd = blockLineNumbers[blockLineNumbers.length - 1] === line.lineNumber;
            const levelActive = canvasLevel === "block"
              ? blockSelected || blockMatches.length > 0
              : isHierarchyDetail
                ? belongsToSelectedBlock && (lineSelected || hierarchyMatches.length > 0 || conceptSelected)
                : lineSelected || conceptSelected;
            const lineSelectionDimmed = selectedLines.length > 0 && !lineSelected;
            const indent = line.text.match(/^\s*/)?.[0] ?? "";
            const displayedLineNumber = displayLineNumberByRaw.get(line.lineNumber) ?? line.lineNumber;
            const displayedBlockStart = displayLineNumberByRaw.get(blockLineNumbers[0] ?? 0) ?? blockLineNumbers[0];
            const displayedBlockEnd = displayLineNumberByRaw.get(blockLineNumbers[blockLineNumbers.length - 1] ?? 0) ?? blockLineNumbers[blockLineNumbers.length - 1];
            return (
              <Fragment key={line.lineNumber}>
                {canvasLevel === "block" && blockStart && block ? (
                  <div className={`code-block-label${blockSelected ? " selected" : ""}`} style={hierarchyColors.length ? { borderColor: hierarchyColors[0] } : undefined}>
                    <span>{block.label}</span>
                    <span className="code-block-meta"><small>L{displayedBlockStart}-{displayedBlockEnd}</small>{signalTexts.map((signalText) => <span key={signalText} className={`hierarchy-code-signal ${signalText === "weak concept evidence" ? "weak" : signalText === "uncovered code detail" ? "uncovered" : "latent"}`}>{signalText}</span>)}</span>
                  </div>
                ) : null}
                <div
                  className={`${levelActive ? "code-line active" : "code-line"}${lineSelectionDimmed ? " line-dimmed" : ""}${canvasLevel === "block" ? " block-line" : ""}${isHierarchyDetail && belongsToSelectedBlock ? " hierarchy-detail-line" : ""}${isHierarchyDetail && selectedBlockId && !belongsToSelectedBlock ? " hierarchy-outside-scope" : ""}${blockStart ? " block-start" : ""}${blockEnd ? " block-end" : ""}${blockSelected ? " block-selected" : ""}`}
                  style={hierarchyColors.length ? {
                    borderLeftColor: hierarchyColors[0],
                    background: hierarchyColors.length === 1
                      ? `${hierarchyColors[0]}18`
                      : `linear-gradient(90deg, ${hierarchyColors.map((color) => `${color}24`).join(", ")})`
                  } : undefined}
                  onClick={() => onLine(line.lineNumber)}
                >
                  <span className="line-number">{displayedLineNumber}</span>
                  <span className="code-line-tokens">
                  <span className="code-indent">{indent}</span>
                  {line.tokenIndices.map((idx) => {
                    const concepts = tokenConcepts.get(idx) ?? [];
                    const manual = manualByCodeToken.get(idx);
                    const tokenId = `c_tok_${idx}`;
                    const showTokenHighlight = canvasLevel === "token" || canvasLevel === "line_tokens";
                    const inheritLineColor = canvasLevel === "line_tokens" && lineSelected && hierarchyMatches.length > 0;
                    const inheritedLineStyle = hierarchyColors.length === 1
                      ? { background: `${hierarchyColors[0]}33`, borderColor: hierarchyColors[0] }
                      : hierarchyColors.length > 1
                        ? { borderColor: hierarchyColors[0], background: `linear-gradient(90deg, ${hierarchyColors.map((color) => `${color}33`).join(", ")})` }
                        : undefined;
                    const recommended = recommendedTokenIndices.has(idx);
                    const inspectLabel = recommended ? `Inspect ${displayToken(candidate.codeTokens[idx])}` : undefined;
                    const selected = showTokenHighlight && (selectedTokenSet.has(tokenId) || concepts.some((concept) => selectedConceptSet.has(concept.conceptId)) || Boolean(manual));
                    return (
                      <Fragment key={idx}>
                      <button
                        className={`${selected ? "line-token active" : "line-token"}${recommended ? " inspect-token" : ""}${showTokenHighlight && manual ? " manual-token" : ""}${canvasLevel !== "line_tokens" && showTokenHighlight && hasTokenFocus && !selected && !lineSelected ? " dimmed" : ""}`}
                        title={inspectLabel}
                        aria-label={inspectLabel}
                        style={showTokenHighlight ? (
                          manual
                            ? { background: `${manual.color}22`, borderColor: manual.color, outlineColor: manual.color }
                            : inheritLineColor
                              ? inheritedLineStyle
                              : canvasLevel === "line_tokens"
                                ? undefined
                                : tokenColorStyle(concepts, selectedConceptSet, selectedLineSet.has(line.lineNumber))
                        ) : undefined}
                        onClick={(event) => {
                          event.stopPropagation();
                          onToken(tokenId);
                        }}
                      >
                        {displayToken(candidate.codeTokens[idx])}
                      </button>
                      </Fragment>
                    );
                  })}
                  {canvasLevel === "line" ? signalTexts.map((signalText) => <span key={signalText} className={`hierarchy-code-signal ${signalText === "weak concept evidence" ? "weak" : signalText === "uncovered code detail" ? "uncovered" : "latent"}`}>{signalText}</span>) : null}
                  </span>
                </div>
              </Fragment>
            );
          })}
        </div>
      </div>}
    </section>
  );
}

function BaselineCodeViewer({
  candidate,
  displaySimilarity
}: {
  candidate: CandidateDetail | null;
  displaySimilarity?: number;
}) {
  if (!candidate) return <section className="panel baseline-code-viewer muted">No candidate selected.</section>;
  return (
    <section className="panel baseline-code-viewer">
      <div className="panel-title">Code Viewer</div>
      <div className="code-meta">
        <span>{candidate.metadata.funcName || candidate.id}</span>
        <span>similarity {safeDisplaySimilarity(displaySimilarity ?? candidate.similarity).toFixed(3)}</span>
      </div>
      <pre className="baseline-code-content">{withoutLeadingFunctionDocstring(candidate.rawCode)}</pre>
    </section>
  );
}

type TokenCanvasProps = React.ComponentProps<typeof TokenVisualizationCanvas>;

function VisualizationCanvas({
  canvasLevel: _canvasLevel,
  selectedBlockId: _selectedBlockId,
  lineTokenScope: _lineTokenScope,
  queryPointMode,
  onCanvasLevel: _onCanvasLevel,
  onQueryPointModeChange,
  onBlock: _onBlock,
  onHierarchyLine: _onHierarchyLine,
  onHierarchyConcept,
  ...tokenProps
}: TokenCanvasProps & {
  canvasLevel: CanvasLevel;
  selectedBlockId: string | null;
  lineTokenScope: number | null;
  queryPointMode: QueryPointMode;
  onCanvasLevel: (level: CanvasLevel) => void;
  onQueryPointModeChange: (mode: QueryPointMode) => void;
  onBlock: (id: string) => void;
  onHierarchyLine: (lineNumber: number) => void;
  onHierarchyConcept: (conceptId: number) => void;
}) {
  const recommendedTokenIndices = AUTO_TOKEN_SUGGESTIONS_ENABLED
    ? new Set(tokenProps.graph?.hierarchy?.recommendedTokenIndices ?? [])
    : new Set<number>();
  const dragLinkedTokenIndices = (() => {
    if (!tokenProps.graph) return new Set<number>();
    const draggedSeeds = tokenProps.graph.nodes.filter((node) => {
      if (node.type !== "code_token") return false;
      const position = tokenProps.persistedPositions[node.id];
      return Boolean(position && Math.hypot(position.x - node.x, position.y - node.y) >= 2);
    });
    if (!draggedSeeds.length) return new Set<number>();
    const seedKeys = new Set(draggedSeeds.map((node) => repeatedCodeTokenKey(node.label)).filter(Boolean));
    return new Set(tokenProps.graph.nodes
      .filter((node) => node.type === "code_token" && !draggedSeeds.some((seed) => seed.id === node.id) && seedKeys.has(repeatedCodeTokenKey(node.label)))
      .map((node) => node.tokenIndex));
  })();
  const visibleNodeFilter = tokenProps.graph
    ? new Set(tokenProps.graph.nodes.filter((node) => {
        const selected = tokenProps.selectedTokenIds.includes(node.id);
        if (!selectedGraphTokenLabel(node, selected)) return false;
        if (node.type === "query_token") {
          const belongsToConcept = node.conceptIds.length > 0;
          return !belongsToConcept
            ? selected
            : queryPointMode === "tokens";
        }
        return true;
      }).map((node) => node.id))
    : null;
  return (
    <div className="token-canvas-shell">
      {dragLinkedTokenIndices.size ? <div className="related-suggestion-notice">Related token suggestion</div> : null}
      <TokenVisualizationCanvas
        {...tokenProps}
        visibleNodeFilter={visibleNodeFilter}
        canvasScope="all"
        queryPointMode={queryPointMode}
        onQueryConcept={onHierarchyConcept}
        showRecommendedTokens={false}
        linkedSuggestionTokenIndices={dragLinkedTokenIndices}
        nodeColorOverrides={{}}
        headerControls={<div className="token-level-switch hierarchy-breadcrumb">
        {AUTO_TOKEN_SUGGESTIONS_ENABLED && recommendedTokenIndices.size ? (
          <button
            type="button"
            className="hierarchy-open"
            onClick={() => undefined}
          >
            Suggestions
          </button>
        ) : null}
        <span className="query-point-mode" aria-label="Query point display">
          <button
            type="button"
            className={queryPointMode === "concept" ? "hierarchy-step active" : "hierarchy-step"}
            onClick={() => onQueryPointModeChange("concept")}
          >
            Concepts
          </button>
          <button
            type="button"
            className={queryPointMode === "tokens" ? "hierarchy-step active" : "hierarchy-step"}
            onClick={() => onQueryPointModeChange("tokens")}
          >
            Tokens
          </button>
        </span>
        </div>}
      />
    </div>
  );
}

function TokenVisualizationCanvas({
  candidate,
  session,
  graph,
  selectedConcepts,
  selectedLines,
  selectedTokenIds,
  selectedManualLinkId,
  manualLinks,
  neighborMode,
  neighborSignals,
  linkMode,
  linkDraft,
  loading,
  onNode,
  onDrop,
  dragMode,
  resetKey,
  dragTokenMatches,
  selectedDragMatchKey,
  dragTargetIds,
  onDragTargetChange,
  dragTargetsConfirmed,
  selectedExternalImpactKey,
  onDragTargetsConfirm,
  persistedPositions,
  onPositionsChange,
  persistedTrails,
  onTrailsChange,
  externalImpacts,
  visibleNodeFilter,
  canvasScope,
  queryPointMode = "tokens",
  onQueryConcept,
  showRecommendedTokens = false,
  linkedSuggestionTokenIndices = new Set<number>(),
  nodeColorOverrides = {},
  headerControls
}: {
  candidate: CandidateDetail | null;
  session: SessionPayload | null;
  graph: VisualizationGraph | null;
  selectedConcepts: number[];
  selectedLines: number[];
  selectedTokenIds: string[];
  selectedManualLinkId: string | null;
  manualLinks: ManualLink[];
  neighborMode: boolean;
  neighborSignals: Record<string, NeighborSignal>;
  linkMode: boolean;
  linkDraft: GraphNode | null;
  loading: boolean;
  onNode: (node: GraphNode) => void;
  onDrop: (payload: { node: GraphNode; updates: Array<{ nodeId: string; similarity: number }>; positions: Record<string, { x: number; y: number }> }) => void;
  dragMode: boolean;
  resetKey: number;
  dragTokenMatches: DragTokenMatch[];
  selectedDragMatchKey: string | null;
  dragTargetIds: string[];
  onDragTargetChange: (ids: string[]) => void;
  dragTargetsConfirmed: boolean;
  selectedExternalImpactKey: string | null;
  onDragTargetsConfirm: (confirmed: boolean) => void;
  persistedPositions: Record<string, { x: number; y: number }>;
  onPositionsChange: (positions: Record<string, { x: number; y: number }>) => void;
  persistedTrails: Record<string, TrailPoint[]>;
  onTrailsChange: (trails: Record<string, TrailPoint[]>) => void;
  externalImpacts: ExternalImpact[];
  visibleNodeFilter?: Set<string> | null;
  canvasScope?: "line" | "all";
  queryPointMode?: QueryPointMode;
  onQueryConcept?: (conceptId: number) => void;
  showRecommendedTokens?: boolean;
  linkedSuggestionTokenIndices?: Set<number>;
  nodeColorOverrides?: Record<string, string[]>;
  headerControls?: React.ReactNode;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const nodeDragRef = useRef<{
    node: GraphNode;
    startX: number;
    startY: number;
    origin: Record<string, { x: number; y: number }>;
  } | null>(null);
  const suppressClickRef = useRef(false);
  const viewportRef = useRef({ zoom: 1, pan: { x: GRAPH_WIDTH / 2, y: GRAPH_HEIGHT / 2 } });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: GRAPH_WIDTH / 2, y: GRAPH_HEIGHT / 2 });
  const [dragPositions, setDragPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [dragBaseline, setDragBaseline] = useState<Record<string, { x: number; y: number }>>({});
  const [dragTrails, setDragTrails] = useState<Record<string, TrailPoint[]>>({});
  const [compareMode, setCompareMode] = useState(false);
  const [dragSummary, setDragSummary] = useState<{
    node: GraphNode;
    phase: "moving" | "dropped";
    updates: Array<{ nodeId: string; similarity: number; distance: number }>;
  } | null>(null);
  const selectedConceptSet = useMemo(() => new Set(selectedConcepts), [selectedConcepts]);
  const selectedLineSet = useMemo(() => new Set(selectedLines), [selectedLines]);
  const selectedTokenSet = useMemo(() => new Set(selectedTokenIds), [selectedTokenIds]);
  const selectedTargetSet = useMemo(() => new Set(dragTargetIds), [dragTargetIds]);
  const recommendationTokenIndexSet = useMemo(
    () => new Set(graph?.hierarchy?.recommendedTokenIndices ?? []),
    [graph?.hierarchy?.recommendedTokenIndices]
  );
  const recommendationConceptIdSet = useMemo(
    () => new Set((graph?.hierarchy?.recommendedTokens ?? []).map((item) => item.conceptId)),
    [graph?.hierarchy?.recommendedTokens]
  );
  const nodeIsVisible = (nodeId: string) => {
    const node = nodeById.get(nodeId);
    return Boolean(node && selectedGraphTokenLabel(node, selectedTokenSet.has(node.id)))
      && (!visibleNodeFilter || visibleNodeFilter.has(nodeId));
  };
  const externalImpactByNode = useMemo(() => {
    const map = new Map<string, ExternalImpact[]>();
    externalImpacts.forEach((impact) => {
      const id = `c_tok_${impact.codeTokenIndex}`;
      map.set(id, [...(map.get(id) ?? []), impact]);
    });
    return map;
  }, [externalImpacts]);
  const nodeById = useMemo(() => {
    const map = new Map<string, GraphNode>();
    graph?.nodes.forEach((node) => map.set(node.id, node));
    return map;
  }, [graph]);
  const conceptDisplayNodes = useMemo(() => {
    if (!graph || !session || queryPointMode !== "concept") return [];
    return session.query.concepts.flatMap((concept) => {
      const members = concept.tokenIndices
        .map((tokenIndex) => nodeById.get(`q_tok_${tokenIndex}`))
        .filter((node): node is GraphNode => Boolean(node));
      if (!members.length) return [];
      const x = members.reduce((sum, node) => sum + node.x, 0) / members.length;
      const y = members.reduce((sum, node) => sum + node.y, 0) / members.length;
      return [{
        id: `q_concept_display_${concept.conceptId}`,
        type: "query_token" as const,
        label: displayConceptText(concept.text),
        conceptId: concept.conceptId,
        conceptIds: [concept.conceptId],
        color: concept.color,
        colors: [concept.color],
        highlightScore: 1,
        tokenIndex: -concept.conceptId - 1,
        x,
        y,
        memberIds: members.map((node) => node.id)
      }];
    });
  }, [graph, session, queryPointMode, nodeById]);
  const conceptDisplayMembers = useMemo(
    () => new Map(conceptDisplayNodes.map((node) => [node.id, node.memberIds])),
    [conceptDisplayNodes]
  );
  const conceptDisplayNodeById = useMemo(
    () => new Map(conceptDisplayNodes.map((node) => [node.conceptId, node])),
    [conceptDisplayNodes]
  );
  const externalImpactConnections = useMemo(() => {
    if (queryPointMode === "tokens" || !session) {
      return externalImpacts.map((impact) => ({ impact, queryNode: nodeById.get(`q_tok_${impact.queryTokenIndex}`) }));
    }
    const grouped = new Map<string, { impact: ExternalImpact; queryNode: GraphNode }>();
    externalImpacts.forEach((impact) => {
      const concepts = session.query.concepts.filter((concept) => concept.tokenIndices.includes(impact.queryTokenIndex));
      concepts.forEach((concept) => {
        const queryNode = conceptDisplayNodeById.get(concept.conceptId);
        if (!queryNode) return;
        const key = `${concept.conceptId}:${impact.codeTokenIndex}`;
        const existing = grouped.get(key);
        if (!existing || Math.abs(impact.delta) > Math.abs(existing.impact.delta)) grouped.set(key, { impact, queryNode });
      });
    });
    return [...grouped.values()];
  }, [externalImpacts, queryPointMode, session, nodeById, conceptDisplayNodeById]);
  const queryConceptTargetGroups = useMemo(() => {
    if (!session) return [];
    return session.query.concepts.map((concept) => ({
      id: `q_concept_target_${concept.conceptId}`,
      label: displayConceptText(concept.text),
      memberIds: concept.tokenIndices.map((tokenIndex) => `q_tok_${tokenIndex}`)
    }));
  }, [session]);
  const queryConceptMembersByTokenId = useMemo(() => {
    const membersByTokenId = new Map<string, string[]>();
    if (queryPointMode !== "concept") return membersByTokenId;
    queryConceptTargetGroups.forEach((group) => {
      group.memberIds.forEach((memberId) => membersByTokenId.set(memberId, group.memberIds));
    });
    return membersByTokenId;
  }, [queryConceptTargetGroups, queryPointMode]);
  const recommendationByCodeToken = useMemo(() => {
    const curated = graph?.hierarchy?.recommendedTokens ?? [];
    const signals = [
      ...(graph?.hierarchy?.blocks ?? []).flatMap((block) => Array.isArray(block.signals) ? block.signals : []),
      ...Object.values(graph?.hierarchy?.linesByBlock ?? {}).flat().flatMap((line) => Array.isArray(line.signals) ? line.signals : [])
    ] as Array<{ kind?: string; tokenIndex?: number; conceptId?: number }>;
    const result = new Map<number, number>();
    curated.forEach((item) => result.set(item.tokenIndex, item.conceptId));
    signals.forEach((signal) => {
      if (signal.kind !== "latent_token" || signal.tokenIndex == null || signal.conceptId == null || result.has(signal.tokenIndex)) return;
      result.set(signal.tokenIndex, signal.conceptId);
    });
    return result;
  }, [graph?.hierarchy]);
  const targetsCanBeConfirmed = dragTargetIds.length > 0;
  const targetDisplayItems = useMemo(() => {
    const remaining = new Set(dragTargetIds);
    const items: Array<{ id: string; label: string; type: "query_token" | "code_token" }> = [];
    if (queryPointMode === "concept") queryConceptTargetGroups.forEach((conceptGroup) => {
      const members = conceptGroup.memberIds;
      if (members.length && members.every((id) => remaining.has(id))) {
        members.forEach((id) => remaining.delete(id));
        items.push({ id: conceptGroup.id, label: conceptGroup.label, type: "query_token" });
      }
    });
    remaining.forEach((id) => {
      const node = nodeById.get(id);
      if (node) items.push({ id, label: node.label, type: node.type });
    });
    return items;
  }, [dragTargetIds, queryConceptTargetGroups, nodeById, queryPointMode]);
  const focusId = selectedTokenIds[0] ?? null;
  const focusNode = focusId ? graph?.nodes.find((node) => node.id === focusId) ?? null : null;
  const neighborItems = useMemo(() => nearestNeighbors(graph, focusId, 5), [graph, focusId]);
  const localFocusIds = useMemo(() => {
    if (dragMode && dragTargetIds.length) return new Set<string>();
    if (selectedDragMatchKey || !focusId || !graph) return new Set<string>();
    const ids = new Set<string>([focusId]);
    nearestNeighbors(graph, focusId, zoom > 2.2 ? 8 : 4).forEach((item) => ids.add(item.node.id));
    return ids;
  }, [graph, focusId, zoom, selectedDragMatchKey]);
  const visibleDragMatches = useMemo(
    () => dragTokenMatches.filter((match) => dragMatchKey(match) === selectedDragMatchKey),
    [dragTokenMatches, selectedDragMatchKey]
  );
  const dragColorsByNode = useMemo(() => {
    const colors = new Map<string, string[]>();
    visibleDragMatches.forEach((match) => {
      const queryId = `q_tok_${match.queryTokenIndex}`;
      const codeId = `c_tok_${match.codeTokenIndex}`;
      colors.set(queryId, [...(colors.get(queryId) ?? []), match.color]);
      colors.set(codeId, [...(colors.get(codeId) ?? []), match.color]);
    });
    return colors;
  }, [visibleDragMatches]);
  const draggedTrailIds = useMemo(() => new Set([
    ...(dragSummary?.node ? [dragSummary.node.id] : []),
    ...dragTokenMatches.filter((match) => match.role === "dragged").flatMap((match) => [`q_tok_${match.queryTokenIndex}`, `c_tok_${match.codeTokenIndex}`])
  ]), [dragSummary?.node?.id, dragTokenMatches]);
  const followerTrailIds = useMemo(() => new Set([
    ...dragTokenMatches.filter((match) => match.role !== "dragged").flatMap((match) => [`q_tok_${match.queryTokenIndex}`, `c_tok_${match.codeTokenIndex}`])
  ]), [dragTokenMatches]);

  useEffect(() => {
    setZoom(1);
    setPan({ x: GRAPH_WIDTH / 2, y: GRAPH_HEIGHT / 2 });
    setDragPositions(persistedPositions);
    setDragBaseline({});
    setDragTrails({});
    setCompareMode(false);
    setDragSummary(null);
    dragRef.current = null;
    nodeDragRef.current = null;
  }, [graph?.candidateId, graph?.epoch, resetKey]);

  useEffect(() => {
    nodeDragRef.current = null;
    dragRef.current = null;
    suppressClickRef.current = false;
    if (!dragMode) setDragSummary(null);
  }, [dragMode]);

  useEffect(() => {
    if (graph?.candidateId) setDragTrails(persistedTrails);
  }, [graph?.candidateId]);

  useEffect(() => {
    viewportRef.current = { zoom, pan };
  }, [zoom, pan]);

  const isNodeActive = (node: GraphNode) => {
    const conceptActive = node.conceptIds?.some((id) => selectedConceptSet.has(id)) || false;
    const lineActive = node.type === "code_token" && node.lineNumber != null && selectedLineSet.has(node.lineNumber);
    return conceptActive || lineActive || selectedTokenSet.has(node.id) || linkDraft?.id === node.id || Boolean(neighborSignals[node.id]);
  };
  const manualLinksForGraph = useMemo(
    () => manualLinks.filter((link) => !graph || link.candidateId === graph.candidateId),
    [manualLinks, graph?.candidateId]
  );
  const hasFocus = selectedConcepts.length > 0 || selectedLines.length > 0 || selectedTokenIds.length > 0 || manualLinksForGraph.length > 0 || Boolean(linkDraft) || Boolean(selectedDragMatchKey) || compareMode;
  const viewWidth = GRAPH_WIDTH / zoom;
  const viewHeight = GRAPH_HEIGHT / zoom;
  const viewX = pan.x - viewWidth / 2;
  const viewY = pan.y - viewHeight / 2;
  const pointFor = (node: GraphNode) => dragPositions[node.id] ?? { x: node.x, y: node.y };
  const recommendationLinks = showRecommendedTokens && session
    ? [...recommendationByCodeToken.entries()].flatMap(([codeTokenIndex, conceptId]) => {
        const codeNode = nodeById.get(`c_tok_${codeTokenIndex}`);
        const concept = session.query.concepts.find((item) => item.conceptId === conceptId);
        if (!codeNode || !concept || !nodeIsVisible(codeNode.id)) return [];
        return concept.tokenIndices
          .map((queryTokenIndex) => nodeById.get(`q_tok_${queryTokenIndex}`))
          .filter((queryNode): queryNode is GraphNode => queryNode !== undefined && nodeIsVisible(queryNode.id))
          .map((queryNode) => ({ codeNode, queryNode, color: concept.color }));
      })
    : [];
  const zoomEmphasis = clamp((zoom - 1.4) / 3.2, 0, 1);
  const shouldShowGraphNodeLabel = (node: GraphNode) => {
    if (!selectedGraphTokenLabel(node, selectedTokenSet.has(node.id))) return false;
    const suggested = node.type === "code_token" && (
      (showRecommendedTokens && recommendationTokenIndexSet.has(node.tokenIndex))
      || linkedSuggestionTokenIndices.has(node.tokenIndex)
    );
    return canvasScope === "line"
      // At rest, reserve the label budget for the non-symbol tokens on
      // concept-aligned code lines. The layout engine will hide only labels
      // that genuinely collide, instead of truncating every token first.
      || (canvasScope === "all" && node.type === "code_token" && node.conceptIds.length > 0)
      || (canvasScope === "all" && node.type === "query_token" && queryPointMode === "tokens")
      || zoom > 1.85
      || selectedTokenSet.has(node.id)
      || selectedTargetSet.has(node.id)
      || isNodeActive(node)
      || suggested
      || (!hasFocus && node.type === "query_token");
  };
  const graphLabelAnnotations = useMemo(() => {
    if (!graph) return new Map<string, HierarchyAnnotation>();
    const isVisible = (nodeId: string) => nodeIsVisible(nodeId);
    const markerObstacles = [
      ...graph.nodes.filter((node) => isVisible(node.id)).map((node) => {
        const point = dragPositions[node.id] ?? { x: node.x, y: node.y };
        const radius = node.type === "query_token" ? 15 : 11;
        return { x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2 };
      }),
      ...conceptDisplayNodes.map((node) => ({ x: node.x - 15, y: node.y - 15, width: 30, height: 30 }))
    ];
    const items: GraphLabelLayoutItem[] = graph.nodes.flatMap((node) => {
      if (!isVisible(node.id) || !shouldShowGraphNodeLabel(node)) return [];
      const point = dragPositions[node.id] ?? { x: node.x, y: node.y };
      const selected = selectedTokenSet.has(node.id) || selectedTargetSet.has(node.id);
      const active = isNodeActive(node);
      const suggested = showRecommendedTokens && (
        (node.type === "code_token" && recommendationTokenIndexSet.has(node.tokenIndex))
        || (node.type === "query_token" && node.conceptIds.some((id) => recommendationConceptIdSet.has(id))));
      const linked = (canvasScope === "line" || canvasScope === "all")
        && node.type === "code_token"
        && linkedSuggestionTokenIndices.has(node.tokenIndex);
      const priority = suggested || linked
        ? 1200
        : selected || active || canvasScope === "line"
        ? 1000
        : canvasScope === "all" && node.type === "query_token" && queryPointMode === "tokens"
          ? 1100
        : node.type === "query_token"
            ? 800
        : node.conceptIds.length
          ? 1080
          : suggested || linked
            ? 900
            : externalImpactByNode.has(node.id)
              ? 850
              : 200;
      const label = selectedGraphTokenLabel(node, selectedTokenSet.has(node.id));
      return [{ id: node.id, x: point.x, y: point.y, width: Math.max(34, label.length * 6.4 + 8), height: 18, priority }];
    });
    conceptDisplayNodes.forEach((node) => {
      const members = conceptDisplayMembers.get(node.id) ?? [];
      const selected = selectedConceptSet.has(node.conceptId) || members.some((id) => selectedTargetSet.has(id));
      const label = displayConceptText(node.label);
      if (!label) return;
      items.push({
        id: node.id,
        x: node.x,
        y: node.y,
        width: Math.max(44, label.length * 6.8 + 10),
        height: 20,
        priority: selected ? 1120 : 1000
      });
    });
    return graphLabelLayout(items, markerObstacles);
  }, [
    graph,
    visibleNodeFilter,
    dragPositions,
    conceptDisplayNodes,
    conceptDisplayMembers,
    selectedConceptSet,
    selectedTokenSet,
    selectedTargetSet,
    selectedLineSet,
    selectedConcepts,
    selectedLines,
    selectedTokenIds,
    linkDraft,
    neighborSignals,
    showRecommendedTokens,
    canvasScope,
    recommendationTokenIndexSet,
    recommendationConceptIdSet,
    linkedSuggestionTokenIndices,
    externalImpactByNode,
    queryPointMode,
    zoom,
    hasFocus
  ]);

  function semanticNeighborsFor(nodeId: string) {
    if (!graph) return [];
    const threshold = graph.semanticLinkThreshold ?? 0.42;
    return (graph.semanticLinks ?? [])
      .filter((link) => (link.source === nodeId || link.target === nodeId) && link.similarity >= threshold)
      .map((link) => ({ node: nodeById.get(link.source === nodeId ? link.target : link.source), similarity: link.similarity }))
      .filter((item): item is { node: GraphNode; similarity: number } => Boolean(item.node));
  }

  function topologyUpdatesFor(node: GraphNode, positions: Record<string, { x: number; y: number }>) {
    const draggedPoint = positions[node.id] ?? pointFor(node);
    const neighbors = semanticNeighborsFor(node.id)
      .map(({ node: neighbor }) => ({
        node: neighbor,
        distance: Math.hypot(
          (positions[neighbor.id]?.x ?? neighbor.x) - draggedPoint.x,
          (positions[neighbor.id]?.y ?? neighbor.y) - draggedPoint.y
        )
      }))
      .sort((a, b) => a.distance - b.distance);
    if (!neighbors.length) return [];
    const robustScale = Math.max(24, neighbors[Math.floor(neighbors.length / 2)].distance);
    return neighbors.map((item) => ({
      nodeId: item.node.id,
      distance: item.distance,
      similarity: clamp(1 / (1 + item.distance / robustScale), 0, 1)
    }));
  }

  function updateDragSummary(node: GraphNode, phase: "moving" | "dropped", positions: Record<string, { x: number; y: number }>) {
    setDragSummary({ node, phase, updates: topologyUpdatesFor(node, positions) });
  }

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return undefined;
    const handleNativeWheel = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      const rect = svg.getBoundingClientRect();
      const { zoom: currentZoom, pan: currentPan } = viewportRef.current;
      const currentViewWidth = GRAPH_WIDTH / currentZoom;
      const currentViewHeight = GRAPH_HEIGHT / currentZoom;
      const currentViewX = currentPan.x - currentViewWidth / 2;
      const currentViewY = currentPan.y - currentViewHeight / 2;
      const relX = rect.width ? (event.clientX - rect.left) / rect.width : 0.5;
      const relY = rect.height ? (event.clientY - rect.top) / rect.height : 0.5;
      const cursorX = currentViewX + relX * currentViewWidth;
      const cursorY = currentViewY + relY * currentViewHeight;
      const factor = event.deltaY < 0 ? 1.12 : 0.88;
      const nextZoom = clamp(currentZoom * factor, 0.65, 6);
      const nextViewWidth = GRAPH_WIDTH / nextZoom;
      const nextViewHeight = GRAPH_HEIGHT / nextZoom;
      const nextPan = {
        x: cursorX - (relX - 0.5) * nextViewWidth,
        y: cursorY - (relY - 0.5) * nextViewHeight
      };
      viewportRef.current = { zoom: nextZoom, pan: nextPan };
      setZoom(nextZoom);
      setPan(nextPan);
    };
    svg.addEventListener("wheel", handleNativeWheel, { passive: false });
    return () => svg.removeEventListener("wheel", handleNativeWheel);
  }, []);

  function handleMouseDown(event: React.MouseEvent<SVGSVGElement>) {
    if (nodeDragRef.current) return;
    event.preventDefault();
    if ((event.target as SVGElement).closest(".graph-node")) return;
    dragRef.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
  }

  function handleMouseMove(event: React.MouseEvent<SVGSVGElement>) {
    const nodeDrag = nodeDragRef.current;
    if (nodeDrag && svgRef.current) {
      const rect = svgRef.current.getBoundingClientRect();
      const dx = ((event.clientX - nodeDrag.startX) / Math.max(1, rect.width)) * viewWidth;
      const dy = ((event.clientY - nodeDrag.startY) / Math.max(1, rect.height)) * viewHeight;
      if (Math.abs(dx) + Math.abs(dy) > 3) suppressClickRef.current = true;
      const draggedOrigin = nodeDrag.origin[nodeDrag.node.id];
      const nextPositions: Record<string, { x: number; y: number }> = {
        ...nodeDrag.origin,
        [nodeDrag.node.id]: { x: draggedOrigin.x + dx, y: draggedOrigin.y + dy }
      };
      semanticNeighborsFor(nodeDrag.node.id)
        .filter(({ node }) => node.type === "code_token" && dragTargetIds.includes(node.id))
        .forEach(({ node, similarity }) => {
        const threshold = graph?.semanticLinkThreshold ?? 0.42;
        const normalizedSimilarity = clamp((similarity - threshold) / (1 - threshold), 0, 1);
        const strength = 0.34 * Math.pow(normalizedSimilarity, 0.75);
        const origin = nodeDrag.origin[node.id];
        if (origin) nextPositions[node.id] = { x: origin.x + dx * strength, y: origin.y + dy * strength };
        });
      setDragPositions(nextPositions);
      onPositionsChange(nextPositions);
      setDragTrails((current) => {
        const next = { ...current };
        Object.entries(nextPositions).forEach(([nodeId, point]) => {
          const trail = next[nodeId] ?? [nodeDrag.origin[nodeId]];
          const previous = trail[trail.length - 1];
          if (!previous || Math.hypot(point.x - previous.x, point.y - previous.y) >= 2) {
            next[nodeId] = [...trail, point];
          }
        });
        onTrailsChange(next);
        return next;
      });
      updateDragSummary(nodeDrag.node, "moving", nextPositions);
      return;
    }
    const drag = dragRef.current;
    if (!drag || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const dx = ((event.clientX - drag.x) / Math.max(1, rect.width)) * viewWidth;
    const dy = ((event.clientY - drag.y) / Math.max(1, rect.height)) * viewHeight;
    setPan({ x: drag.panX - dx, y: drag.panY - dy });
  }

  function stopPanning() {
    dragRef.current = null;
  }

  function handleNodeMouseDown(event: React.MouseEvent<SVGGElement>, node: GraphNode) {
    if (!dragMode || linkMode) return;
    event.preventDefault();
    event.stopPropagation();
    if (!dragTargetsConfirmed) {
      suppressClickRef.current = true;
      const conceptMembers = conceptDisplayMembers.get(node.id)
        ?? (queryPointMode === "concept" && node.type === "query_token" ? queryConceptMembersByTokenId.get(node.id) : undefined);
      if (conceptMembers) {
        const allSelected = conceptMembers.every((id) => dragTargetIds.includes(id));
        onDragTargetChange(allSelected
          ? dragTargetIds.filter((id) => !conceptMembers.includes(id))
          : [...new Set([...dragTargetIds, ...conceptMembers])]
        );
        return;
      }
      onDragTargetChange(
        dragTargetIds.includes(node.id)
          ? dragTargetIds.filter((id) => id !== node.id)
          : [...dragTargetIds, node.id]
      );
      return;
    }
    if (!dragTargetsConfirmed || dragTargetIds.includes(node.id) || conceptDisplayMembers.has(node.id)) return;
    const origin: Record<string, { x: number; y: number }> = {};
    graph?.nodes.forEach((item) => {
      origin[item.id] = pointFor(item);
    });
    const resetOrigin: Record<string, { x: number; y: number }> = {};
    graph?.nodes.forEach((item) => {
      resetOrigin[item.id] = { x: item.x, y: item.y };
    });
    setDragBaseline(resetOrigin);
    const affectedIds = new Set<string>([node.id]);
    semanticNeighborsFor(node.id).forEach(({ node: neighbor }) => affectedIds.add(neighbor.id));
    setDragTrails((current) => {
      const next = { ...current };
      affectedIds.forEach((nodeId) => {
        if (!next[nodeId] && resetOrigin[nodeId]) next[nodeId] = [resetOrigin[nodeId]];
      });
      onTrailsChange(next);
      return next;
    });
    setCompareMode(true);
    nodeDragRef.current = { node, startX: event.clientX, startY: event.clientY, origin };
  }

  function handleMouseUp() {
    const nodeDrag = nodeDragRef.current;
    if (!nodeDrag) {
      stopPanning();
      return;
    }
    if (suppressClickRef.current && graph) {
      const updates = topologyUpdatesFor(nodeDrag.node, dragPositions);
      updateDragSummary(nodeDrag.node, "dropped", dragPositions);
      setDragTrails((current) => {
        const next = { ...current };
        Object.keys(next).forEach((nodeId) => {
          const point = dragPositions[nodeId];
          if (!point) return;
          const previous = next[nodeId][next[nodeId].length - 1];
          if (!previous || Math.hypot(point.x - previous.x, point.y - previous.y) >= 2) {
            next[nodeId] = [...next[nodeId], point];
          }
        });
        onTrailsChange(next);
        return next;
      });
      onDrop({ node: nodeDrag.node, updates, positions: dragPositions });
    }
    nodeDragRef.current = null;
    dragRef.current = null;
  }

  return (
    <main className="panel canvas-panel">
      <div className="canvas-head">
        <div>
          <div className="panel-title">Embedding Space</div>
          <div className="meta-line">
            {graph
              ? `${visibleNodeFilter?.size ?? graph.nodes.length} tokens · ${canvasScope === "line" ? "selected line" : "snapshot"}${dragMode ? (dragTargetIds.length ? ` · ${dragTargetIds.length} target${dragTargetIds.length > 1 ? "s" : ""} selected` : " · select target token(s)") : ""}`
              : "waiting for projection"}
          </div>
        </div>
        <div className="canvas-head-controls">
        {headerControls}
        <button
            className={compareMode ? "canvas-action active" : "canvas-action"}
            onClick={() => setCompareMode((value) => !value)}
            disabled={!Object.keys(dragPositions).length}
            title="显示拖拽前后的局部位置变化"
          >
            <ArrowLeftRight size={14} />
            Compare
        </button>
        {loading && (
          <div className="loading">
            <Loader2 size={16} className="spin" />
            Embedding running
          </div>
        )}
        </div>
      </div>
      <svg
        ref={svgRef}
        className="graph"
        viewBox={`${viewX} ${viewY} ${viewWidth} ${viewHeight}`}
        role="img"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <defs>
          <marker id="external-impact-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M0,0 L7,3.5 L0,7 z" fill="#7c3aed" />
          </marker>
          {graph?.nodes.map((node) => {
            const colors = nodeColorOverrides[node.id]?.length ? nodeColorOverrides[node.id] : nodeColors(node);
            if (colors.length <= 1) return null;
            const step = 100 / colors.length;
            return (
              <linearGradient key={`grad_${node.id}`} id={`${nodeColorOverrides[node.id]?.length ? "line_node_grad" : "node_grad"}_${node.id}`} x1="0%" y1="0%" x2="100%" y2="0%">
                {colors.map((color, idx) => (
                  <React.Fragment key={`${node.id}_${color}_${idx}`}>
                    <stop offset={`${idx * step}%`} stopColor={color} />
                    <stop offset={`${(idx + 1) * step}%`} stopColor={color} />
                  </React.Fragment>
                ))}
              </linearGradient>
            );
          })}
        </defs>
        {compareMode && <rect x={viewX} y={viewY} width={viewWidth} height={viewHeight} fill="rgba(248,250,252,0.68)" />}
        {compareMode && Object.entries(dragTrails).map(([nodeId, trail]) => {
          const original = trail[0] ?? dragBaseline[nodeId];
          const finalPoint = dragPositions[nodeId] ?? trail[trail.length - 1];
          if (!original || !finalPoint || Math.hypot(finalPoint.x - original.x, finalPoint.y - original.y) < 1) return null;
          const isDraggedTrail = draggedTrailIds.has(nodeId);
          const isFollowerTrail = followerTrailIds.has(nodeId) || (!isDraggedTrail && trail.length > 0);
          if (!isDraggedTrail && !isFollowerTrail) return null;
          const lastTrailPoint = trail[trail.length - 1];
          const points = lastTrailPoint && Math.hypot(finalPoint.x - lastTrailPoint.x, finalPoint.y - lastTrailPoint.y) >= 2
            ? [...trail, finalPoint]
            : trail.length > 1 ? trail : [original, finalPoint];
          return (
            <g key={`trail_${nodeId}`} className={isDraggedTrail ? "motion-trail primary" : "motion-trail follower"}>
              <polyline points={points.map((point) => `${point.x},${point.y}`).join(" ")} />
              <circle cx={original.x} cy={original.y} r={isDraggedTrail ? 4 : 2.6} className="trail-origin" />
              <circle cx={finalPoint.x} cy={finalPoint.y} r={isDraggedTrail ? 5.5 : 3.4} className="trail-end" />
            </g>
          );
        })}
        {recommendationLinks.map(({ codeNode, queryNode, color }) => {
          const codePoint = pointFor(codeNode);
          const queryPoint = pointFor(queryNode);
          return (
            <line
              key={`recommendation_${codeNode.id}_${queryNode.id}`}
              x1={codePoint.x}
              y1={codePoint.y}
              x2={queryPoint.x}
              y2={queryPoint.y}
              className="recommendation-link"
              stroke={color}
            />
          );
        })}
        {!compareMode && !neighborMode && !((canvasScope === "all" || canvasScope === "line") && !hasFocus) && graph?.edges.map((edge) => {
          if (!nodeIsVisible(edge.source) || !nodeIsVisible(edge.target)) return null;
          const source = nodeById.get(edge.source);
          const target = nodeById.get(edge.target);
          if (!source || !target) return null;
          const sourcePoint = pointFor(source);
          const targetPoint = pointFor(target);
          const active = selectedConceptSet.has(edge.conceptId);
          return (
            <line
              key={edge.id}
              x1={sourcePoint.x}
              y1={sourcePoint.y}
              x2={targetPoint.x}
              y2={targetPoint.y}
              stroke={edge.color}
              strokeWidth={active ? 4 : 1.4 + edge.similarity}
              opacity={active ? 0.9 : hasFocus ? 0.06 : 0.16}
            />
          );
        })}
        {!compareMode && !neighborMode && manualLinksForGraph.map((link) => {
          const source = nodeById.get(`q_tok_${link.queryTokenIndex}`);
          const target = nodeById.get(`c_tok_${link.codeTokenIndex}`);
          if (!source || !target || !nodeIsVisible(source.id) || !nodeIsVisible(target.id)) return null;
          const sourcePoint = pointFor(source);
          const targetPoint = pointFor(target);
          const active = selectedManualLinkId === link.id || (selectedTokenSet.has(source.id) && selectedTokenSet.has(target.id));
          return (
            <line
              key={link.id}
              x1={sourcePoint.x}
              y1={sourcePoint.y}
              x2={targetPoint.x}
              y2={targetPoint.y}
              stroke={link.color}
              strokeDasharray="6 4"
              strokeWidth={active ? 5 : 3}
              opacity={active ? 1 : 0.78}
            />
          );
        })}
        {!neighborMode && visibleDragMatches.map((match) => {
          const source = nodeById.get(`q_tok_${match.queryTokenIndex}`);
          const target = nodeById.get(`c_tok_${match.codeTokenIndex}`);
          if (!source || !target || !nodeIsVisible(source.id) || !nodeIsVisible(target.id)) return null;
          const sourcePoint = pointFor(source);
          const targetPoint = pointFor(target);
          const isGeneralized = match.source === "generalized";
          const isNegative = (match.delta ?? 0) < 0;
          return (
            <line
              key={`drag_match_${match.queryTokenIndex}_${match.codeTokenIndex}`}
              x1={sourcePoint.x}
              y1={sourcePoint.y}
              x2={targetPoint.x}
              y2={targetPoint.y}
              stroke={match.color}
              strokeDasharray={isGeneralized ? (isNegative ? "2 5" : "7 4") : "5 5"}
              strokeWidth={compareMode ? 4.8 : match.role === "dragged" ? 4.2 : isGeneralized ? 2.8 : 2.4}
              opacity={compareMode ? 0.98 : match.role === "dragged" ? 0.96 : 0.72}
            />
          );
        })}
        {!neighborMode && externalImpactConnections.map(({ impact, queryNode }, index) => {
          const queryTokenIndex = impact.queryTokenIndex;
          const codeNode = nodeById.get(`c_tok_${impact.codeTokenIndex}`);
          if (!codeNode || !queryNode || !nodeIsVisible(codeNode.id)) return null;
          const sourcePoint = pointFor(codeNode);
          const targetPoint = pointFor(queryNode);
          const dx = targetPoint.x - sourcePoint.x;
          const dy = targetPoint.y - sourcePoint.y;
          const length = Math.max(18, Math.min(90, 20 + Math.abs(impact.delta) * 320));
          const norm = Math.hypot(dx, dy) || 1;
          const arrowLength = Math.min(length, Math.max(18, norm * 0.72));
          const selected = selectedExternalImpactKey === externalImpactKey(impact)
            && queryTokenIndex === impact.queryTokenIndex;
          const arrowStart = sourcePoint;
          const arrowDirection = impact.delta >= 0 ? 1 : -1;
          const arrowEnd = {
            x: arrowStart.x + (dx / norm) * arrowLength * arrowDirection,
            y: arrowStart.y + (dy / norm) * arrowLength * arrowDirection
          };
          return (
            <line
              key={`external_impact_${impact.sourceCandidateId}_${impact.codeTokenIndex}_${queryTokenIndex}_${index}`}
              x1={arrowStart.x}
              y1={arrowStart.y}
              x2={arrowEnd.x}
              y2={arrowEnd.y}
              className={selected ? "external-impact-arrow active" : "external-impact-arrow"}
              stroke={impact.color}
              strokeDasharray={`${Math.max(4, length / 5)} ${Math.max(4, length / 7)}`}
              markerEnd="url(#external-impact-arrow)"
            />
          );
        })}
        {neighborMode && focusNode && neighborItems.map((item) => {
          const signal = neighborSignals[item.node.id];
          const focusPoint = pointFor(focusNode);
          const neighborPoint = pointFor(item.node);
          return (
            <line
              key={`neighbor_${item.node.id}`}
              x1={focusPoint.x}
              y1={focusPoint.y}
              x2={neighborPoint.x}
              y2={neighborPoint.y}
              className={`neighbor-line ${signal?.status ?? "stable"}`}
            />
          );
        })}
        {graph?.nodes
          .filter((node) => nodeIsVisible(node.id))
          .slice()
          .sort((a, b) => {
            const priority = (node: GraphNode) => {
              const suggestion = showRecommendedTokens && (
                (node.type === "code_token" && recommendationTokenIndexSet.has(node.tokenIndex)) ||
                (node.type === "query_token" && node.conceptIds.some((id) => recommendationConceptIdSet.has(id))));
              const linkedSuggestion = (canvasScope === "line" || canvasScope === "all") && node.type === "code_token" && linkedSuggestionTokenIndices.has(node.tokenIndex);
              if (selectedTokenSet.has(node.id) || selectedTargetSet.has(node.id) || draggedTrailIds.has(node.id) || suggestion || linkedSuggestion) return 3;
              if (localFocusIds.has(node.id)) return 2;
              if (dragColorsByNode.has(node.id) || followerTrailIds.has(node.id) || externalImpactByNode.has(node.id)) return 1;
              return 0;
            };
            return priority(a) - priority(b);
          })
          .map((node) => {
          const active = isNodeActive(node);
          const isQuery = node.type === "query_token";
          const signal = neighborSignals[node.id];
          const selectedDirectly = selectedTokenSet.has(node.id);
          const selectedTarget = selectedTargetSet.has(node.id);
          const localNeighbor = localFocusIds.has(node.id) && !selectedDirectly;
          // Line Tokens is an inspection scope, not a global overview. Every
          // token on the selected line must remain equally readable; only the
          // label layout may suppress overlapping annotations.
          const lowPriority = canvasScope !== "line" && !hasFocus && node.type === "code_token" && !node.conceptIds.length;
          const tokenFocus = selectedTokenIds.length > 0;
          const point = pointFor(node);
          const isDragging = Boolean(nodeDragRef.current?.node.id === node.id);
          const isDraggedMarker = draggedTrailIds.has(node.id);
          const isFollowerMarker = followerTrailIds.has(node.id);
          const externalImpactsForNode = externalImpactByNode.get(node.id) ?? [];
          const externallyImpacted = node.type === "code_token" && externalImpactsForNode.length > 0;
          const recommendedSuggestion = showRecommendedTokens && node.type === "code_token" && recommendationByCodeToken.has(node.tokenIndex);
          const dragLinkedSuggestion = (canvasScope === "line" || canvasScope === "all") && node.type === "code_token" && linkedSuggestionTokenIndices.has(node.tokenIndex);
          const motionRelevant = draggedTrailIds.has(node.id)
            || followerTrailIds.has(node.id)
            || dragColorsByNode.has(node.id)
            || externallyImpacted
            || recommendedSuggestion
            || dragLinkedSuggestion;
          const dimmed = canvasScope !== "line" && ((hasFocus && !active && !motionRelevant) || lowPriority);
          const nodeOpacity = canvasScope === "line"
            ? 1
            : selectedDirectly || selectedTarget
            ? 1
            : localNeighbor
              ? 0.66 + zoomEmphasis * 0.22
              : dimmed
                ? 0.34 + zoomEmphasis * 0.22
                : 0.68 + zoomEmphasis * 0.22;
          const label = selectedGraphTokenLabel(node, selectedDirectly);
          const annotation = graphLabelAnnotations.get(node.id);
          const markerScale = selectedDirectly || selectedTarget ? 1.08 : localNeighbor ? 1.04 + zoomEmphasis * 0.12 : 1 + zoomEmphasis * 0.08;
          const queryRadius = (active ? 10 : 7) * markerScale;
          const codeRadius = (active ? 9 : 6) * markerScale;
          const overrideColors = nodeColorOverrides[node.id];
          const nodeFillColor = overrideColors?.length
            ? overrideColors.length === 1 ? overrideColors[0] : `url(#line_node_grad_${node.id})`
            : nodeFill(node, selectedConceptSet);
          return (
            <g
              key={node.id}
              onMouseDown={(event) => handleNodeMouseDown(event, node)}
              onClick={() => {
                if (suppressClickRef.current) {
                  suppressClickRef.current = false;
                  return;
                }
                onNode(node);
              }}
              className={`${active ? "graph-node active" : "graph-node"} ${isQuery ? "query-token-node" : "code-token-node"}${dimmed ? " dimmed" : ""}${signal ? ` neighbor-${signal.status}` : ""}${node.id === focusId ? " focus-node" : ""}${localNeighbor ? " local-neighbor" : ""}${isDragging ? " dragging" : ""}${isDraggedMarker ? " dragged-marker" : ""}${isFollowerMarker ? " follower-marker" : ""}`}
              style={{ opacity: canvasScope === "line" || tokenFocus || zoom > 1.4 ? nodeOpacity : undefined }}
            >
              {externallyImpacted ? (
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={(isQuery ? queryRadius : codeRadius) + 8}
                  className="external-impact-glow"
                  fill={externalImpactsForNode[0]?.color ?? "#7c3aed"}
                />
              ) : null}
              {recommendedSuggestion ? (
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={codeRadius + 5}
                  className="recommendation-glow"
                  fill="#f59e0b"
                />
              ) : null}
              {dragLinkedSuggestion ? (
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={codeRadius + 3}
                  className="drag-linked-suggestion-glow"
                  fill="#fbbf24"
                />
              ) : null}
              {isQuery ? (
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={queryRadius}
                  fill={dragColorsByNode.has(node.id) ? dragColorsByNode.get(node.id)![0] : nodeFillColor}
                  stroke="#17202a"
                  strokeWidth={selectedDirectly || selectedTarget ? 3.2 : active ? 2.5 : 1}
                />
              ) : (
                <polygon
                  points={`${point.x},${point.y - codeRadius} ${point.x - codeRadius},${point.y + codeRadius * 0.88} ${point.x + codeRadius},${point.y + codeRadius * 0.88}`}
                  fill={dragColorsByNode.has(node.id) ? dragColorsByNode.get(node.id)![0] : nodeFillColor}
                  stroke="#17202a"
                  strokeWidth={selectedDirectly || selectedTarget ? 3.2 : active ? 2.5 : 1}
                />
              )}
              {annotation ? <>
                <line className="hierarchy-label-leader" x1={point.x} y1={point.y} x2={annotation.leaderX} y2={annotation.leaderY} />
                <text x={annotation.labelX} y={annotation.labelY} textAnchor={annotation.textAnchor}>{label}</text>
              </> : null}
              <title>{displayToken(node.label)}</title>
            </g>
          );
        })}
        {conceptDisplayNodes.map((node) => {
          const members = conceptDisplayMembers.get(node.id) ?? [];
          const active = selectedConceptSet.has(node.conceptId) || members.some((id) => selectedTargetSet.has(id));
          const point = { x: node.x, y: node.y };
          const label = displayConceptText(node.label);
          const annotation = graphLabelAnnotations.get(node.id);
          return (
            <g
              key={node.id}
              className={active ? "graph-node active concept-display-node query-concept-node" : "graph-node concept-display-node query-concept-node"}
              onMouseDown={(event) => handleNodeMouseDown(event, node)}
              onClick={() => {
                if (suppressClickRef.current) {
                  suppressClickRef.current = false;
                  return;
                }
                onQueryConcept?.(node.conceptId);
              }}
            >
              <circle
                cx={point.x}
                cy={point.y}
                r={active ? 13 : 11}
                fill={node.color}
                stroke="#17202a"
                strokeWidth={active ? 2.8 : 1.4}
              />
              {label && annotation ? <>
                <line className="hierarchy-label-leader" x1={point.x} y1={point.y} x2={annotation.leaderX} y2={annotation.leaderY} />
                <text x={annotation.labelX} y={annotation.labelY} textAnchor={annotation.textAnchor}>{label}</text>
              </> : null}
              <title>{label}</title>
            </g>
          );
        })}
      </svg>
      {dragMode ? (
        <div className="target-selection-popover">
          {dragTargetIds.length ? (
            <>
              <div className="target-selection-title">Selected target</div>
              <div className="target-selection-list">
                {targetDisplayItems.map((target) => (
                  <div key={target.id} className="target-selection-token">
                    {target.type === "query_token" ? "Q" : "C"} · {graphTokenLabel(target.label) || target.label}
                  </div>
                ))}
              </div>
              <button
                type="button"
                className="target-selection-action"
                disabled={!dragTargetsConfirmed && !targetsCanBeConfirmed}
                onClick={() => onDragTargetsConfirm(!dragTargetsConfirmed)}
              >
                {dragTargetsConfirmed ? "Edit targets" : "Confirm targets"}
              </button>
            </>
          ) : (
            <div className="target-selection-empty">Select a target token</div>
          )}
          {!dragTargetsConfirmed && dragTargetIds.length > 0 && !targetsCanBeConfirmed ? (
            <div className="target-selection-warning">Select at least one target</div>
          ) : null}
        </div>
      ) : null}
    </main>
  );
}

function TaskBriefOverview() {
  return <section className="assignment-section assignment-overview-section">
    <div className="assignment-section-label">Task Overview</div>
    <p>你不需要直接编写代码。请先阅读任务背景，然后从 20 个候选代码片段中选择一个最有助于后续生成目标实现的 Reference。</p>
    <p>请根据任务需要判断哪段代码最值得参考；目标实现本身不在候选列表中。</p>
  </section>;
}

function TaskBriefReferenceGoal({ points }: { points: string[] }) {
  return <section className="assignment-section assignment-reference-goal-section">
    <div className="assignment-section-label">Reference Goal</div>
    <p>系统将提供 20 个候选代码片段，目标实现不在候选中。请选择 ONE Reference，作为后续代码生成的参考。</p>
    <div className="assignment-reference-points">
      <span>一个有价值的 Reference 应帮助你理解：</span>
      <ul>{points.map((point) => <li key={point}>{point}</li>)}</ul>
    </div>
  </section>;
}

function TaskBriefPanel({ brief, onConfirm, onClose, ready = true }: { brief: TaskBrief; onConfirm: () => void; onClose?: () => void; ready?: boolean }) {
  const [acknowledged, setAcknowledged] = useState(false);
  if (brief.caseId === "csn_12226") {
    return <section className={`panel task-brief-panel assignment-task-brief api-result-task-brief ${onClose ? "task-brief-readonly" : ""}`}>
      <div className="assignment-brief-header">
        <div>
          <div className="assignment-brief-kicker">Task Brief · CSN 12226</div>
          <h2>处理删除 alarm 的 API response</h2>
        </div>
        {onClose ? <button onClick={onClose}>Close</button> : null}
      </div>

      <div className="assignment-brief-content">
        <TaskBriefOverview />
        <section className="assignment-section assignment-task-section">
          <div className="assignment-section-label">Task</div>
          <p>你正在维护一个删除 alarm 的 CLI command。上游已经完成 API 请求；当前任务是根据 response 的状态决定是否向终端输出内容。</p>
        </section>

        <div className="assignment-contract-grid api-result-contract-grid">
          <section className="assignment-section assignment-input-section">
            <div className="assignment-section-label">Input</div>
            <h3>API response result</h3>
            <p>当前结果包含 HTTP status 与 response body。删除操作成功时不需要额外反馈；失败时 response body 会说明错误原因。</p>
          </section>
          <section className="assignment-section assignment-output-section">
            <div className="assignment-section-label">Output</div>
            <h3>Terminal output</h3>
            <p><code>200</code> 时保持安静。非 <code>200</code> 时向终端输出带颜色的 JSON error response。</p>
          </section>
        </div>

        <section className="assignment-section assignment-example-section">
          <div className="assignment-section-label">Result scenarios</div>
          <div className="api-result-scenarios" aria-label="API response result scenarios">
            <article className="api-result-card success">
              <header><strong>Deletion succeeds</strong><code>HTTP 200</code></header>
              <div><span>response body</span><code>{'{"status": "deleted"}'}</code></div>
              <div className="api-terminal-output"><span>stdout</span><code>&lt;empty&gt;</code></div>
            </article>
            <article className="api-result-card error">
              <header><strong>Target not found</strong><code>HTTP 404</code></header>
              <div><span>response body</span><code>{'{"error": "Alarm not found"}'}</code></div>
              <div className="api-terminal-output colorized"><span>stdout</span><code>{'{"error": "Alarm not found"}'}</code></div>
            </article>
            <article className="api-result-card error">
              <header><strong>Server error</strong><code>HTTP 500</code></header>
              <div><span>response body</span><code>{'{"error": "Internal server error"}'}</code></div>
              <div className="api-terminal-output colorized"><span>stdout</span><code>{'{"error": "Internal server error"}'}</code></div>
            </article>
          </div>
        </section>

        <section className="assignment-section assignment-query-section">
          <div className="assignment-section-label">Query</div>
          <p>处理当前 API response：成功删除时不输出；失败时将错误 JSON response 以终端颜色展示。</p>
          <blockquote>{brief.taskQuery}</blockquote>
        </section>
        <TaskBriefReferenceGoal points={["CLI 如何读取 API response 的状态和 body。", "成功与失败结果如何走向不同的终端输出行为。", "JSON response 如何被处理为终端展示内容。"]} />
      </div>

      {!onClose ? <div className="issue-brief-actions">
        <label className="task-brief-ack"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /> 我已阅读任务背景，并理解需要选择一个最有助于后续生成的 Reference。</label>
        <button className={`task-brief-enter ${acknowledged && ready ? "ready" : ""}`} onClick={onConfirm} disabled={!acknowledged || !ready}>{ready ? acknowledged ? "进入任务" : "确认理解后进入任务" : "正在准备检索工作区..."}</button>
      </div> : null}
    </section>;
  }
  if (brief.caseId === "csn_3846") {
    return <section className={`panel task-brief-panel assignment-task-brief decorator-task-brief ${onClose ? "task-brief-readonly" : ""}`}>
      <div className="assignment-brief-header">
        <div>
          <div className="assignment-brief-kicker">Task Brief · CSN 3846</div>
          <h2>判断 decorator 是否引用当前方法名称</h2>
        </div>
        {onClose ? <button onClick={onClose}>Close</button> : null}
      </div>

      <div className="assignment-brief-content">
        <TaskBriefOverview />
        <section className="assignment-section assignment-task-section">
          <div className="assignment-section-label">任务背景介绍</div>
          <p>你正在维护与 Python method decorator 相关的判定逻辑。当前任务是判断：某个 dotted decorator 是否引用了与当前 method 相同的名称。</p>
          <div className="ast-dead-code-note"><strong>Dotted decorator</strong><span>Dotted decorator 是使用点号连接的 decorator 表达式，例如 <code>@x.setter</code>。其中 <code>x</code> 是点号左侧被引用的名称，<code>setter</code> 是在该名称上访问的 decorator 属性。</span></div>
        </section>

        <div className="assignment-contract-grid decorator-contract-grid">
          <section className="assignment-section assignment-input-section">
            <div className="assignment-section-label">Input</div>
            <h3>Current method node</h3>
            <p>当前对象表示一个带有方法名和 decorators 的 Python method。decorator 可以是普通名称，也可以是引用某个名称的 dotted form。</p>
          </section>
          <section className="assignment-section assignment-output-section">
            <div className="assignment-section-label">Output</div>
            <h3>Boolean result <small>True / False</small></h3>
            <p>当某个 dotted decorator 左侧引用的名称与当前方法名相同时，返回 <code>True</code>；否则返回 <code>False</code>。</p>
          </section>
        </div>

        <section className="assignment-section assignment-example-section">
          <div className="assignment-section-label">Decision examples</div>
          <div className="decorator-rule-grid" aria-label="Decorator redefinition decision examples">
            <article><code>@property</code><code>def x(...)</code><strong className="decision-false">False</strong><span>普通 decorator 名称</span></article>
            <article><code>@x.setter</code><code>def x(...)</code><strong className="decision-true">True</strong><span>引用相同名称</span></article>
            <article><code>@x.deleter</code><code>def x(...)</code><strong className="decision-true">True</strong><span>引用相同名称</span></article>
            <article><code>@y.setter</code><code>def x(...)</code><strong className="decision-false">False</strong><span>引用不同名称</span></article>
            <article><code>@staticmethod</code><code>def x(...)</code><strong className="decision-false">False</strong><span>普通 decorator 名称</span></article>
          </div>
        </section>

        <section className="assignment-section assignment-query-section">
          <div className="assignment-section-label">Query</div>
          <p>检查当前 method 的 decorators，判断其中引用的名称是否与当前 method name 相同。</p>
          <blockquote>{brief.taskQuery}</blockquote>
        </section>
        <TaskBriefReferenceGoal points={["函数的 decorators 如何被访问并逐项检查。", "dotted decorator 如何表示它所引用的名称。", "如何从 AST 信息判断该名称是否与当前方法名相同。"]} />
      </div>

      {!onClose ? <div className="issue-brief-actions">
        <label className="task-brief-ack"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /> 我已阅读任务背景，并理解需要选择一个最有助于后续生成的 Reference。</label>
        <button className={`task-brief-enter ${acknowledged && ready ? "ready" : ""}`} onClick={onConfirm} disabled={!acknowledged || !ready}>{ready ? acknowledged ? "进入任务" : "确认理解后进入任务" : "正在准备检索工作区..."}</button>
      </div> : null}
    </section>;
  }
  if (brief.caseId === "csn_42") {
    return <section className={`panel task-brief-panel assignment-task-brief cloud-sql-task-brief ${onClose ? "task-brief-readonly" : ""}`}>
      <div className="assignment-brief-header">
        <div>
          <div className="assignment-brief-kicker">Task Brief · CSN 42</div>
          <h2>等待 Cloud SQL 数据库删除完成</h2>
        </div>
        {onClose ? <button onClick={onClose}>Close</button> : null}
      </div>

      <div className="assignment-brief-content">
        <TaskBriefOverview />
        <section className="assignment-section assignment-task-section">
          <div className="assignment-section-label">任务背景介绍</div>
          <p>你正在删除 Cloud SQL 中的一个数据库。Cloud SQL 的删除接口采用异步操作：请求返回只表示删除任务已被接受，并不代表数据库已经删除。</p>
          <div className="ast-region-list" aria-label="Cloud SQL asynchronous deletion flow">
            <span>发送删除请求</span><span>获得本次 Operation</span><span>等待该 Operation 完成</span><span>成功后返回 None</span>
          </div>
          <p>函数必须跟踪这次删除请求返回的 Operation；只有确认该 Operation 成功完成后才能正常返回。如果 Operation 失败，则应报告相应错误。</p>
        </section>

        <div className="assignment-contract-grid">
          <section className="assignment-section assignment-input-section">
            <div className="assignment-section-label">Input</div>
            <h3>要删除的 Cloud SQL database</h3>
            <div className="format-term-list" aria-label="Cloud SQL delete input">
              <div><strong>project_id</strong><span>Cloud SQL 项目的名称</span><code>demo-project</code></div>
              <div><strong>instance</strong><span>承载 database 的数据库实例</span><code>prod-db</code></div>
              <div><strong>database</strong><span>要删除的 database 名称</span><code>analytics</code></div>
            </div>
          </section>
          <section className="assignment-section assignment-output-section">
            <div className="assignment-section-label">Output</div>
            <h3>成功时返回 None</h3>
            <p>删除操作成功完成后，函数返回 <code>None</code>；此时目标数据库已不存在。如果后台 Operation 失败，应报告相应错误，而不是正常返回。</p>
          </section>
        </div>

        <section className="assignment-section assignment-example-section">
          <div className="assignment-section-label">测试场景（输入与输出）</div>
          <div className="behavior-scenario-grid" aria-label="Cloud SQL database deletion behavior examples">
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 1 · 请求完成后删除完成</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p><code>demo-project / prod-db / analytics</code><br />后台删除操作已完成。</p></div>
              <div className="scenario-example-row"><span>期望输出</span><p>返回 <code>None</code>；<code>analytics</code> 已不存在。</p></div>
            </article>
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 2 · 请求已接受，删除仍在后台进行</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p>删除请求已返回，但短时间内仍可找到 <code>analytics</code>。</p></div>
              <div className="scenario-example-row"><span>期望输出</span><p>函数继续等待后台操作完成；此时不应返回。</p></div>
            </article>
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 3 · 等待本次请求对应的 Operation</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p>删除请求返回 <code>Operation B</code>，同时系统中存在其他 <code>Operation A</code>。</p></div>
              <div className="scenario-example-row"><span>期望输出</span><p>只跟踪并等待 <code>Operation B</code> 完成，然后返回 <code>None</code>。</p></div>
            </article>
            <article className="behavior-scenario-card scenario-fail">
              <header><strong>错误输出示例</strong><span>不能通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p>删除请求已被接受，但 <code>analytics</code> 仍存在。</p></div>
              <div className="scenario-example-row"><span>错误输出</span><p>函数立刻返回 <code>None</code>，没有等待后台删除完成。</p></div>
            </article>
          </div>
        </section>

        <section className="assignment-section assignment-query-section">
          <div className="assignment-section-label">Query</div>
          <p>从 Cloud SQL instance 删除一个 database，并确保函数结束时删除已完成。</p>
          <blockquote>{brief.taskQuery}</blockquote>
        </section>
        <TaskBriefReferenceGoal points={["类似云端 database 删除请求如何实际发送。", "请求返回后如何确认后台操作已经真正完成。", "如何区分请求已被接受与 database 已经删除。"]} />
      </div>

      {!onClose ? <div className="issue-brief-actions">
        <label className="task-brief-ack"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /> 我已阅读任务背景，并理解需要选择一个最有助于后续生成的 Reference。</label>
        <button className={`task-brief-enter ${acknowledged && ready ? "ready" : ""}`} onClick={onConfirm} disabled={!acknowledged || !ready}>{ready ? acknowledged ? "进入任务" : "确认理解后进入任务" : "正在准备检索工作区..."}</button>
      </div> : null}
    </section>;
  }
  if (brief.caseId === "csn_9388") {
    return <section className={`panel task-brief-panel assignment-task-brief url-task-brief ${onClose ? "task-brief-readonly" : ""}`}>
      <div className="assignment-brief-header">
        <div>
          <div className="assignment-brief-kicker">Task Brief · CSN 9388</div>
          <h2>从 URL 中移除查询参数</h2>
        </div>
        {onClose ? <button onClick={onClose}>Close</button> : null}
      </div>

      <div className="assignment-brief-content">
        <TaskBriefOverview />
        <section className="assignment-section assignment-task-section">
          <div className="assignment-section-label">任务背景介绍</div>
          <p>你正在维护一个处理 URL 的小工具。一个 URL 可以包含 scheme、host、path、query 和 fragment。当前任务只移除 query（查询参数），其他部分保持不变。</p>
          <div className="url-component-legend" aria-label="URL component terminology">
            <span><code>scheme</code>协议</span><span><code>host</code>主机地址</span><span><code>path</code>路径</span><span><code>query</code>查询部分</span><span><code>fragment</code>片段标识</span>
          </div>
        </section>

        <div className="assignment-contract-grid">
          <section className="assignment-section assignment-input-section">
            <div className="assignment-section-label">Input</div>
            <h3>一个 URL 字符串</h3>
            <p>URL 的 query component 位于 <code>?</code> 之后、<code>#</code> 之前。例如在 <code>?page=2#summary</code> 中，query component 是 <code>page=2</code>；<code>summary</code> 属于 fragment。</p>
            <div className="url-segment-example" aria-label="URL components to preserve or remove">
              <div className="url-segment preserve"><code>https://example.com/report.csv</code><span>保留</span></div>
              <b className="url-segment-separator">|</b>
              <div className="url-segment remove"><code>?page=2</code><span>删除</span></div>
              <b className="url-segment-separator">|</b>
              <div className="url-segment preserve"><code>#summary</code><span>保留</span></div>
            </div>
          </section>
          <section className="assignment-section assignment-output-section">
            <div className="assignment-section-label">Output</div>
            <h3>移除 query component 后的 URL</h3>
            <p>保留 scheme、网站地址、路径和 fragment，只让 <code>?</code> 后的查询参数消失。</p>
            <div className="url-result-example"><code>https://example.com/report.csv</code><code>#summary</code></div>
          </section>
        </div>

        <section className="assignment-section assignment-example-section">
          <div className="assignment-section-label">测试场景（输入与输出）</div>
          <div className="behavior-scenario-grid format-scenario-grid" aria-label="URL query removal behavior examples">
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 1 · 含查询参数和 fragment</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p><code>https://example.com/report.csv?page=2#summary</code></p></div>
              <div className="scenario-example-row"><span>期望输出</span><p><code>https://example.com/report.csv#summary</code></p></div>
            </article>
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 2 · 含查询参数、不含 fragment</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p><code>http://example.test/v1/items;latest?limit=10</code></p></div>
              <div className="scenario-example-row"><span>期望输出</span><p><code>http://example.test/v1/items;latest</code></p></div>
            </article>
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 3 · 没有查询参数</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p><code>https://example.com/landing#top</code></p></div>
              <div className="scenario-example-row"><span>期望输出</span><p>保持不变：<code>https://example.com/landing#top</code></p></div>
            </article>
            <article className="behavior-scenario-card scenario-fail">
              <header><strong>错误输出示例</strong><span>不能通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p><code>https://example.com/report.csv?page=2#summary</code></p></div>
              <div className="scenario-example-row"><span>错误输出</span><p>连路径或 <code>#summary</code> 也被删除，或仍保留 <code>?page=2</code>。</p></div>
            </article>
          </div>
        </section>

        <section className="assignment-section assignment-query-section">
          <div className="assignment-section-label">Query</div>
          <p>读取 URL 的不同部分，重新组合出一个不含 query component 的 URL。</p>
          <blockquote>{brief.taskQuery}</blockquote>
        </section>
        <TaskBriefReferenceGoal points={["代码如何读取和重新组合 URL 的不同部分。", "如何只改变 query component 而保留其他 URL 信息。", "同一代码库中已有的 URL 处理约定。"]} />
      </div>

      {!onClose ? <div className="issue-brief-actions">
        <label className="task-brief-ack"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /> 我已阅读任务背景，并理解需要选择一个最有助于后续生成的 Reference。</label>
        <button className={`task-brief-enter ${acknowledged && ready ? "ready" : ""}`} onClick={onConfirm} disabled={!acknowledged || !ready}>{ready ? acknowledged ? "进入任务" : "确认理解后进入任务" : "正在准备检索工作区..."}</button>
      </div> : null}
    </section>;
  }
  if (brief.caseId === "csn_8884") {
    return <section className={`panel task-brief-panel assignment-task-brief ast-task-brief ${onClose ? "task-brief-readonly" : ""}`}>
      <div className="assignment-brief-header">
        <div>
          <div className="assignment-brief-kicker">Task Brief · CSN 8884</div>
          <h2>清理 Try AST 中的 dead code</h2>
        </div>
        {onClose ? <button onClick={onClose}>Close</button> : null}
      </div>

      <div className="assignment-brief-content">
        <TaskBriefOverview />
        <section className="assignment-section assignment-task-section">
          <div className="assignment-section-label">任务背景介绍</div>
          <p>你正在维护一个 Python AST optimizer，需要处理 <code>try</code> statement 对应的 AST node。当前任务是清理其中不可到达的 dead code，同时保持 <code>try / except</code> 的控制流结构和源码位置信息有效。</p>
          <div className="ast-dead-code-note"><strong>Dead code</strong><span>在某个执行区域中，若控制流已被 <code>return</code>、<code>raise</code> 等语句终止，后续 statements 可能不可到达。</span></div>
        </section>

        <div className="assignment-contract-grid ast-contract-grid">
          <section className="assignment-section assignment-input-section">
            <div className="assignment-section-label">Input</div>
            <h3>Try AST node <small>node: ast.Try</small></h3>
            <p>当前输入表示一个 Python <code>try</code> statement 的 AST 节点，并包含多个执行区域。</p>
            <div className="ast-region-list" aria-label="Try execution regions">
              <span>Try body</span><span>Except handler(s)</span><span>Else branch</span><span>Finally branch</span>
            </div>
          </section>
          <section className="assignment-section assignment-output-section">
            <div className="assignment-section-label">Output</div>
            <h3>Optimized Try AST <small>Optional[ast.AST]</small></h3>
            <ul className="ast-output-list">
              <li>移除不可到达的 statements</li>
              <li>保持原有 <code>try / except</code> 控制流结构</li>
              <li>保持 source-location metadata 有效</li>
            </ul>
          </section>
        </div>

        <section className="assignment-section assignment-example-section">
          <div className="assignment-section-label">Before and after</div>
          <div className="ast-before-after-diagram" aria-label="Dead code removal preserves Try AST regions and source location metadata">
            <article className="ast-diagram-state">
              <div className="ast-diagram-title">Before Optimization · Try AST</div>
              <div className="ast-diagram-regions">
                <div><strong>Try body</strong><span>Return</span></div>
                <div><strong>Except handler(s)</strong><span>...</span></div>
                <div><strong>Else branch</strong><span>Return</span><em><span>Statement (dead)</span><b aria-label="removed">×</b></em></div>
                <div><strong>Finally branch</strong><span>Raise</span><em><span>Statement (dead)</span><b aria-label="removed">×</b></em></div>
              </div>
            </article>
            <div className="ast-diagram-arrow" aria-hidden="true"><span>Remove unreachable statements</span><b>→</b></div>
            <article className="ast-diagram-state ast-diagram-output">
              <div className="ast-diagram-title">After Optimization · Try AST</div>
              <div className="ast-diagram-regions">
                <div><strong>Try body</strong><span>Return</span><small>preserved</small></div>
                <div><strong>Except handler(s)</strong><span>...</span><small>preserved</small></div>
                <div><strong>Else branch</strong><span>Return</span><small>preserved</small></div>
                <div><strong>Finally branch</strong><span>Raise</span><small>preserved</small></div>
              </div>
            </article>
            <div className="ast-diagram-preservation">
              <span className="ast-dead-code-removed-note">Dead statements appear only before optimization.</span>
              <strong>Preserve:</strong>
              <span>try / except branch structure</span>
              <span>source-location metadata</span>
            </div>
          </div>
        </section>

        <section className="assignment-section assignment-example-section">
          <div className="assignment-section-label">测试场景（输入与输出）</div>
          <div className="behavior-scenario-grid" aria-label="Try AST optimization behavior examples">
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 1 · 清理 Try body</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p><code>Try body: Return → Statement (dead)</code><br /><code>Except handler(s): ...</code></p></div>
              <div className="scenario-example-row"><span>期望输出</span><p><code>Try body: Return</code><br />后续 dead statement 被移除；<code>except</code> handler 仍保留。</p></div>
            </article>
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 2 · 清理所有区域</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p><code>Else: Return → Statement (dead)</code><br /><code>Finally: Raise → Statement (dead)</code></p></div>
              <div className="scenario-example-row"><span>期望输出</span><p><code>Else: Return</code><br /><code>Finally: Raise</code><br />两个区域中的 dead statement 都被移除。</p></div>
            </article>
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 3 · 保持结构有效</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p><code>2 except handlers + else + finally</code><br />所有 statements 都可到达。</p></div>
              <div className="scenario-example-row"><span>期望输出</span><p>输出保留两个 handlers、<code>else</code>、<code>finally</code> 和所有 statements；source location 仍有效。</p></div>
            </article>
            <article className="behavior-scenario-card scenario-fail">
              <header><strong>错误输出示例</strong><span>不能通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p><code>Else: Return → Statement (dead)</code><br /><code>Finally: Raise → Statement (dead)</code></p></div>
              <div className="scenario-example-row"><span>错误输出</span><p>只清理 <code>Try body</code>，但 <code>else</code> 或 <code>finally</code> 中仍有 dead statement。</p></div>
            </article>
          </div>
        </section>

        <section className="assignment-section assignment-query-section">
          <div className="assignment-section-label">Query</div>
          <p>对当前 <code>Try</code> AST，清理各执行区域中不可到达的代码。</p>
          <blockquote>{brief.taskQuery}</blockquote>
        </section>
        <TaskBriefReferenceGoal points={["项目如何遍历并重写包含多个 statement list 的 AST 节点。", "如何处理 control-flow node 中需要清理的 statement regions。", "AST 变换后如何保持节点结构和 source-location information 有效。"]} />
      </div>

      {!onClose ? <div className="issue-brief-actions">
        <label className="task-brief-ack"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /> 我已阅读任务背景，并理解需要选择一个最有助于后续生成的 Reference。</label>
        <button className={`task-brief-enter ${acknowledged && ready ? "ready" : ""}`} onClick={onConfirm} disabled={!acknowledged || !ready}>{ready ? acknowledged ? "进入任务" : "确认理解后进入任务" : "正在准备检索工作区..."}</button>
      </div> : null}
    </section>;
  }
  if (brief.caseId === "csn_11772") {
    return <section className={`panel task-brief-panel assignment-task-brief ${onClose ? "task-brief-readonly" : ""}`}>
      <div className="assignment-brief-header">
        <div>
          <div className="assignment-brief-kicker">Task Brief · CSN 11772</div>
          <h2>为处理后的文件确定正确的后缀</h2>
        </div>
        {onClose ? <button onClick={onClose}>Close</button> : null}
      </div>

      <div className="assignment-brief-content">
        <TaskBriefOverview />
        <section className="assignment-section assignment-task-section">
          <div className="assignment-section-label">Task</div>
          <p>有些文件会先经过工具处理，处理后的文件格式可能与原文件不同。当前能够读取到的文件不包含原始文件路径，因此不能从类似 <code>example.less</code> 的文件名读取后缀；需要根据处理后的文件格式，为当前文件推断应使用的后缀。</p>
        </section>

        <div className="assignment-contract-grid">
          <section className="assignment-section assignment-input-section">
            <div className="assignment-section-label">Input</div>
            <h3>一个 MIME type</h3>
            <p>输入只有处理工具给出的 <strong>MIME type</strong>。它是一段用来说明“处理结果是什么类型文件”的文字标签，例如 <code>text/coffeescript</code>。</p>
            <div className="format-term-list" aria-label="Format term definitions">
              <div><strong>MIME type</strong><span>处理结果的文件类型标签</span><code>text/coffeescript</code></div>
            </div>
            <p>项目环境中已经保存了处理结果格式与文件后缀之间的对应关系；任务需要利用这个既有信息完成推断。</p>
          </section>
          <section className="assignment-section assignment-output-section">
            <div className="assignment-section-label">Output</div>
            <h3>处理结果应使用的文件后缀 <small>Optional[str]</small></h3>
            <p>文件后缀是文件名最后一个点号后的短标记，例如 <code>.coffee</code>。输出是与输入 MIME type 对应的后缀；若项目不知道这个 MIME type 对应什么后缀，则输出 <code>None</code>，表示不添加后缀。</p>
          </section>
        </div>

        <section className="assignment-section assignment-example-section">
          <div className="assignment-section-label">测试场景（输入与输出）</div>
          <div className="behavior-scenario-grid format-scenario-grid" aria-label="Compiled asset extension behavior examples">
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 1 · 已知格式</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p>处理工具报告 MIME type：<code>text/coffeescript</code></p></div>
              <div className="scenario-example-row"><span>期望输出</span><p><code>.coffee</code></p></div>
            </article>
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 2 · 另一种已知格式</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p>处理工具报告 MIME type：<code>text/x-template</code></p></div>
              <div className="scenario-example-row"><span>期望输出</span><p><code>.tmpl</code></p></div>
            </article>
            <article className="behavior-scenario-card scenario-pass">
              <header><strong>测试 3 · 没有已知对应</strong><span>通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p>处理工具报告的 MIME type 在项目环境中没有已知对应关系。</p></div>
              <div className="scenario-example-row"><span>期望输出</span><p><code>None</code>，表示不添加后缀。</p></div>
            </article>
            <article className="behavior-scenario-card scenario-fail">
              <header><strong>错误输出示例</strong><span>不能通过</span></header>
              <div className="scenario-example-row"><span>输入</span><p>处理工具报告 MIME type：<code>text/coffeescript</code></p></div>
              <div className="scenario-example-row"><span>错误输出</span><p>返回 <code>None</code> 或任意无关后缀，而不是已知对应的 <code>.coffee</code>。</p></div>
            </article>
          </div>
        </section>

        <section className="assignment-section assignment-query-section">
          <div className="assignment-section-label">Query</div>
          <p>读取处理工具报告的 MIME type，并找出处理结果应使用的文件后缀。</p>
          <blockquote>{brief.taskQuery}</blockquote>
        </section>
        <TaskBriefReferenceGoal points={["如何取得处理工具报告的 MIME type。", "项目如何记录 MIME type 与文件后缀的对应关系。", "如何从 MIME type 找到处理结果应使用的后缀。"]} />
      </div>

      {!onClose ? <div className="issue-brief-actions">
        <label className="task-brief-ack"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /> 我已阅读任务背景，并理解需要选择一个最有助于后续生成的 Reference。</label>
        <button className={`task-brief-enter ${acknowledged && ready ? "ready" : ""}`} onClick={onConfirm} disabled={!acknowledged || !ready}>{ready ? acknowledged ? "进入任务" : "确认理解后进入任务" : "正在准备检索工作区..."}</button>
      </div> : null}
    </section>;
  }
  if (brief.caseId === "csn_584") {
    return <section className={`panel task-brief-panel issue-grounded-brief ${onClose ? "task-brief-readonly" : ""}`}>
      <div className="issue-brief-header">
        <div>
          <div className="issue-brief-kicker">Task Brief · CSN 584</div>
          <h2>理解 RandomPerspective 所需的几何参数，并选择一段可帮助后续实现的参考代码。</h2>
        </div>
        {onClose ? <button onClick={onClose}>Close</button> : null}
      </div>

      <div className="issue-brief-stages">
        <section className="issue-brief-stage">
          <div className="issue-stage-number">01</div>
          <div className="github-issue-mock">
            <div className="github-repo">pytorch <span>/</span> vision</div>
            <div className="github-issue-title">[Feature request] Add perspective transform <span>#779</span></div>
            <div className="github-issue-meta"><span className="issue-status">Closed</span><span>fmassa opened this issue · Mar 8, 2019</span><span>enhancement · help wanted</span></div>
            <div className="github-comment github-opening-comment">
              <div className="github-comment-head"><span className="github-avatar avatar-yumike">f</span><strong>fmassa</strong><span>opened this issue</span></div>
              <p>TorchVision does not yet expose a perspective image transform. Pillow already provides the underlying operation, so torchvision needs an interface that fits its transform pipeline.</p>
            </div>
            <div className="github-comment">
              <div className="github-comment-head"><span className="github-avatar avatar-rafales">f</span><strong>fmassa</strong><span>reviewed PR #781</span></div>
              <p>Passing low-level perspective coefficients directly would make the transform difficult to reason about. A clearer interface is to describe the transformation with two quadrilaterals and let the functional implementation handle the lower-level coefficients.</p>
            </div>
            <div className="github-comment">
              <div className="github-comment-head"><span className="github-avatar avatar-yumike">s</span><strong>surgan12</strong><span>replied</span></div>
              <p>The perspective interface can use two quadrilaterals as its geometric input.</p>
            </div>
            <div className="github-comment">
              <div className="github-comment-head"><span className="github-avatar avatar-yumike">s</span><strong>surgan12</strong><span>follow-up</span></div>
              <p>For the random transform, the transform layer can generate random destination vertices and pass the resulting geometry to the functional perspective operation.</p>
            </div>
            <a href="https://github.com/pytorch/vision/issues/779" target="_blank" rel="noreferrer">View original issue ↗</a>
            <a href="https://github.com/pytorch/vision/pull/781" target="_blank" rel="noreferrer">View implementation PR ↗</a>
          </div>
        </section>

        <section className="issue-brief-stage">
          <div className="issue-stage-number">02</div>
          <h3>同一案例中的项目知识</h3>
          <p>下面沿用同一个 RandomPerspective 场景，说明开发者在进入代码检索前需要掌握的最低限度项目知识。</p>
          <div className="issue-knowledge-list">
            <div><span>Perspective geometry</span><code>original quadrilateral → transformed quadrilateral</code><small>一次 perspective transform 可以通过变换前和变换后的两个四边形来描述。</small></div>
            <div><span>Perspective parameters</span><code>startpoints · endpoints</code><small>startpoints 表示原始几何，endpoints 表示变换后的几何。</small></div>
            <div><span>Point structure</span><code>4 × (x, y)</code><small>每组数据包含四个二维角点，并采用 TL → TR → BR → BL 的顺序。</small></div>
            <div><span>RandomPerspective</span><code>random destination geometry</code><small>它生成随机的 destination geometry，再交给后续 perspective operation。</small></div>
          </div>
          <p className="issue-muted">这些信息定义了 RandomPerspective 需要产生的数据类型与高层几何语义；具体如何根据图像尺寸和随机强度构造一组有效参数，需要通过检索项目已有代码进一步判断。</p>
        </section>

        <section className="issue-brief-stage">
          <div className="issue-stage-number">03</div>
          <h3>你的开发任务</h3>
          <p>你正在接手 TorchVision 中 RandomPerspective 的相关功能实现。现有 perspective API 已使用 startpoints 和 endpoints 表示变换前后的几何。对于 RandomPerspective，还需要在调用后续 perspective operation 之前，根据图像尺寸和随机变换设置生成这两组参数。</p>
          <p>其中 <code>distortion_scale</code> 表示允许的随机透视变化强度。为了理解项目已有代码对这些参数的结构和使用方式有什么要求，你决定先检索代码库中的相关实现，并选择一段可用于后续代码生成的 Reference。</p>
          <div className="issue-query-label">Issue</div>
          <blockquote>{brief.taskQuery}</blockquote>
          <p>系统将提供 20 个候选代码片段；目标实现不在候选中。请选择一段最值得作为后续代码生成参考的 Reference，而不是寻找现成答案。</p>
        </section>
      </div>

      {!onClose ? <div className="issue-brief-actions">
        <label className="task-brief-ack"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /> 我已阅读任务背景，并理解需要选择一个最有助于后续生成的 Reference。</label>
        <button className={`task-brief-enter ${acknowledged && ready ? "ready" : ""}`} onClick={onConfirm} disabled={!acknowledged || !ready}>{ready ? acknowledged ? "进入任务" : "确认理解后进入任务" : "正在准备检索工作区..."}</button>
      </div> : null}
    </section>;
  }
  return <section className="panel task-brief-panel">
    <div className="generation-header">
      <div><div className="panel-title">Task Brief</div><p>{brief.role}</p></div>
      {onClose ? <button onClick={onClose}>Close</button> : null}
    </div>
    {brief.sections?.length ? brief.sections.map((section) => <section key={section.heading}>
      <h3>{section.heading}</h3>
      {section.content?.length ? section.content.map((item, index) => item.type === "paragraph"
        ? <p key={`${section.heading}-paragraph-${index}`}>{item.text}</p>
        : <pre key={`${section.heading}-code-${index}`} className="task-brief-code"><code>{item.content}</code></pre>
      ) : <>
        {section.paragraphs?.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
        {section.codeBlocks?.map((block) => <pre key={block.content} className="task-brief-code"><code>{block.content}</code></pre>)}
      </>}
      {section.bullets?.length ? <ul className="task-brief-bullets">{section.bullets.map((item) => <li key={item}>{item}</li>)}</ul> : null}
    </section>) : <>
      <section><h3>系统背景</h3><p>{brief.systemContext}</p></section>
      <section><h3>领域对象</h3><div className="task-brief-objects">{brief.domainObjects.map((item) => <div key={item.name}><strong>{item.name}</strong><span>{item.description}</span></div>)}</div></section>
      <section><h3>需要理解的信息</h3><ol>{brief.essentialDomainKnowledge.map((item) => <li key={item}>{item}</li>)}</ol></section>
      <section><h3>当前 Query</h3><p className="task-brief-query">{brief.taskQuery}</p></section>
      <section><h3>Reference 选择</h3><p>{brief.referenceSelectionInstruction}</p></section>
    </>}
    {!onClose ? <label className="task-brief-ack"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /> 我已阅读任务背景，并理解需要选择一个最有助于后续生成的 Reference。</label> : null}
    {!onClose ? <button className={`task-brief-enter ${acknowledged && ready ? "ready" : ""}`} onClick={onConfirm} disabled={!acknowledged || !ready}>{ready ? acknowledged ? "进入任务" : "确认理解后进入任务" : "正在准备检索工作区..."}</button> : null}
  </section>;
}

function ParticipantGate({ condition, onStart }: { condition: "baseline" | "irag"; onStart: (participantId: string) => Promise<void> }) {
  const [participantId, setParticipantId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function submit() {
    setSubmitting(true); setError(null);
    try { await onStart(participantId); } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setSubmitting(false); }
  }
  return <main className="participant-gate"><section className="panel"><div className="panel-title">Interactive RAG Study</div><p>Enter your assigned anonymous participant ID to begin.</p><label>Participant ID<input value={participantId} onChange={(event) => setParticipantId(event.target.value)} maxLength={64} autoFocus /></label>{error ? <div className="error">{error}</div> : null}<button className="primary" onClick={submit} disabled={submitting || !participantId.trim()}>{submitting ? "Starting..." : `Start ${condition === "irag" ? "IRAG" : "Baseline"}`}</button></section></main>;
}

function PostTaskMeasures({ onSubmit }: { onSubmit: (confidence: number, difficulty: number, reason: string) => Promise<void> }) {
  const [confidence, setConfidence] = useState(4);
  const [difficulty, setDifficulty] = useState(4);
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function submit() {
    setSubmitting(true);
    setError(null);
    try {
      await onSubmit(confidence, difficulty, reason);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }
  return <section className="generation-reference-analysis post-task-measures"><div className="panel-title">Task questionnaire</div><label>How confident are you that this reference will help complete the task? <small>1 = not confident at all; 7 = extremely confident</small><select value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} disabled={submitting}>{[1, 2, 3, 4, 5, 6, 7].map((value) => <option key={value} value={value}>{value}</option>)}</select></label><label>How difficult was it to select a useful reference? <small>1 = not difficult at all; 7 = extremely difficult</small><select value={difficulty} onChange={(event) => setDifficulty(Number(event.target.value))} disabled={submitting}>{[1, 2, 3, 4, 5, 6, 7].map((value) => <option key={value} value={value}>{value}</option>)}</select></label><label>What was the main reason for your selection? <textarea value={reason} onChange={(event) => setReason(event.target.value)} disabled={submitting} /></label>{error ? <div className="error">{error}</div> : null}<button className="primary" onClick={submit} disabled={submitting}>{submitting ? "Submitting..." : "Submit response"}</button></section>;
}

function GenerationPanel({
  confirmation,
  result,
  comparison,
  evaluation,
  loading,
  showInternal,
  hint,
  hintLoading,
  onHint,
  showMeasures,
  onSubmitMeasures,
  onGenerate,
  onEvaluate,
  onBack,
  finalized,
  onFinalize,
  showFinalization
}: {
  confirmation: GenerationConfirmation;
  result: GenerationResult | null;
  comparison: GenerationComparison | null;
  evaluation: GenerationEvaluation | null;
  loading: boolean;
  showInternal: boolean;
  hint: ReferenceHint | null;
  hintLoading: boolean;
  onHint: () => void;
  showMeasures: boolean;
  onSubmitMeasures: (confidence: number, difficulty: number, reason: string) => Promise<void>;
  onGenerate: (condition: "no_rag" | "interactive_rag") => void;
  onEvaluate: () => void;
  onBack: () => void;
  finalized: boolean;
  onFinalize: () => void;
  showFinalization: boolean;
}) {
  const { task, selection } = confirmation;
  return (
    <section className="panel generation-panel">
      <div className="generation-header">
        <div>
          <div className="panel-title">Generate From Selected Reference</div>
          <p>{task.query}</p>
          {task.functionSignature ? <pre className="generation-signature">{task.functionSignature}</pre> : null}
        </div>
        {!finalized ? <button onClick={onBack}>Back to Retrieval</button> : null}
      </div>
      <div className="generation-context">
        <div className="panel-title">Selected Reference</div>
        <div className="generation-context-meta">
          <strong>{selection.candidate.metadata.funcName || selection.selectedCandidateId}</strong>
          <span>Rank {selection.selectedRank ?? "-"} · score {selection.selectedScore?.toFixed(3) ?? "-"}</span>
          <span>{selection.interactionUsed ? "interaction used" : "no interaction"}</span>
        </div>
        <details open>
          <summary>View selected retrieved code</summary>
          <pre className="generation-context-code">{withoutLeadingFunctionDocstring(selection.candidate.rawCode)}</pre>
        </details>
      </div>
      {APP_MODE === "demo" ? <section className="generation-reference-analysis" aria-label="Reference hint">
        {!hint ? <div className="generation-reference-analysis-header"><div className="panel-title">Need help understanding this reference?</div><button onClick={onHint} disabled={hintLoading}>{hintLoading ? "Loading hint..." : "Show Hint"}</button></div> : <><div className="panel-title">Reference Hint</div><div className="reference-hint"><strong>What it does</strong><p>{hint.whatItDoes}</p><strong>Useful clue</strong><p>{hint.usefulClue}</p></div></>}
      </section> : null}
      <div className="generation-actions">
        <button className="primary" onClick={() => onGenerate("interactive_rag")} disabled={loading}>{loading ? <Loader2 size={16} className="spin" /> : <CirclePlay size={16} />} Generate With Selected Reference</button>
        {showInternal ? <button onClick={() => onGenerate("no_rag")} disabled={loading}>Generate Without Reference</button> : null}
        {showInternal && result && task.evaluationAvailable ? <button onClick={onEvaluate} disabled={loading}>Re-run Evaluation</button> : null}
        {showFinalization && (!finalized ? <button onClick={onFinalize} disabled={loading}>Confirm Reference</button> : <span className="generation-finalized">Reference confirmed for this task.</span>)}
      </div>
      {result ? <div className="generation-output">
        <div className="panel-title">Generation Result</div>
        <div className="generation-result-meta">{result.condition === "no_rag" ? "No retrieval context" : "Selected reference context"} · {result.curatedGeneration ? "Curated representative 5-run result" : "Live generation"} · {result.model} · {result.promptVersion} · {result.generationTime.toFixed(2)}s</div>
        {showInternal && comparison ? <div className="generation-comparison">
          <section>
            <div className="comparison-title">Generated Code</div>
            <pre>{comparison.generatedCode}</pre>
          </section>
          <section>
            <div className="comparison-title">Hidden Ground-Truth Implementation</div>
            <div className="comparison-meta">{comparison.groundTruth.functionName || comparison.groundTruth.candidateId} · {comparison.groundTruth.path}</div>
            <pre>{comparison.groundTruth.rawCode}</pre>
          </section>
        </div> : <pre>{result.generatedCode}</pre>}
      </div> : null}
      {evaluation ? <div className={`generation-evaluation ${evaluation.status}`}>
        <div className="panel-title">Evaluation</div>
        <div className="evaluation-summary">
          {evaluation.status === "ok" ? <strong>Functional tests: {evaluation.testsPassed} / {evaluation.testsTotal} passed · {Math.round((evaluation.passRate ?? 0) * 100)}%</strong> : <span>Functional tests: unavailable on this server. {evaluation.message}</span>}
        </div>
        {showInternal && evaluation.apiPrecision != null && evaluation.apiRecall != null ? <div className="evaluation-metrics">
          <div><span>API precision</span><strong>{(evaluation.apiPrecision * 100).toFixed(1)}%</strong></div>
          <div><span>API recall</span><strong>{(evaluation.apiRecall * 100).toFixed(1)}%</strong></div>
          <div><span>API F1</span><strong>{((evaluation.apiF1 ?? 0) * 100).toFixed(1)}%</strong></div>
        </div> : null}
        {showInternal && (evaluation.generatedApis?.length || evaluation.groundTruthApis?.length) ? <div className="evaluation-api-sets">
          <span>Generated APIs: {evaluation.generatedApis?.join(", ") || "none"}</span>
          <span>GT APIs: {evaluation.groundTruthApis?.join(", ") || "none"}</span>
        </div> : null}
        {evaluation.testResults?.length ? <div className="evaluation-tests">
          {evaluation.testResults.map((test) => <div key={test.name} className={test.passed ? "evaluation-test passed" : "evaluation-test failed"}>
            <strong>{test.passed ? "Pass" : "Fail"}</strong><span>{test.name}</span>{test.error ? <small>{test.error}</small> : null}
          </div>)}
        </div> : null}
        {showInternal && evaluation.matchedApis?.length ? <div className="evaluation-api-match">Matched APIs: {evaluation.matchedApis.join(", ")}</div> : null}
      </div> : null}
      {showMeasures && finalized ? <PostTaskMeasures onSubmit={onSubmitMeasures} /> : null}
    </section>
  );
}

function GenerationOpeningPanel({
  candidate,
  query,
  onBack
}: {
  candidate: CandidateDetail | null;
  query: string;
  onBack: () => void;
}) {
  return (
    <section className="panel generation-panel generation-opening-panel">
      <div className="generation-header">
        <div>
          <div className="panel-title">Generate From Selected Reference</div>
          <p>{query}</p>
        </div>
        <button onClick={onBack}>Back to Retrieval</button>
      </div>
      {candidate ? <div className="generation-context">
        <div className="panel-title">Selected Reference</div>
        <div className="generation-context-meta"><strong>{candidate.metadata.funcName || candidate.id}</strong></div>
        <details open>
          <summary>View selected retrieved code</summary>
          <pre className="generation-context-code">{withoutLeadingFunctionDocstring(candidate.rawCode)}</pre>
        </details>
      </div> : null}
      <div className="generation-opening-status"><Loader2 size={16} className="spin" /> Registering this selection...</div>
    </section>
  );
}

function App() {
  const appMode = APP_MODE;
  const isBaseline = appMode === "baseline";
  const isDemo = appMode === "demo";
  const studyCondition = isBaseline ? "baseline" : "irag";
  const [tests, setTests] = useState<string[]>([]);
  const [testId, setTestId] = useState(DEFAULT_TEST_ID);
  const [modelId, setModelId] = useState("xsearch");
  const [session, setSession] = useState<SessionPayload | null>(null);
  const [candidate, setCandidate] = useState<CandidateDetail | null>(null);
  const [activeCandidateId, setActiveCandidateId] = useState<string | null>(null);
  const [graph, setGraph] = useState<VisualizationGraph | null>(null);
  const [selectedConcepts, setSelectedConcepts] = useState<number[]>([]);
  const [selectedLines, setSelectedLines] = useState<number[]>([]);
  const [selectedTokenIds, setSelectedTokenIds] = useState<string[]>([]);
  const [selectedManualLinkId, setSelectedManualLinkId] = useState<string | null>(null);
  const [manualLinks, setManualLinks] = useState<ManualLink[]>([]);
  const [manualLinkStore, setManualLinkStore] = useState<Record<string, ManualLink[]>>({});
  const [linkMode, setLinkMode] = useState(false);
  const [linkDraft, setLinkDraft] = useState<GraphNode | null>(null);
  const [candidates, setCandidates] = useState<CandidateSummary[]>([]);
  const [adjudicationMode, setAdjudicationMode] = useState(false);
  const [adjudicationIds, setAdjudicationIds] = useState<string[]>([]);
  const [adjudicationCandidates, setAdjudicationCandidates] = useState<Record<string, CandidateDetail>>({});
  const candidateDetailCacheRef = useRef<Record<string, CandidateDetail>>({});
  const graphCacheRef = useRef<Record<string, VisualizationGraph>>({});
  const candidateRequestCacheRef = useRef<Record<string, Promise<CandidateDetail>>>({});
  const graphRequestCacheRef = useRef<Record<string, Promise<VisualizationGraph>>>({});
  const cacheGenerationRef = useRef(0);
  const [playing, setPlaying] = useState(false);
  const [focusPaneOrder, setFocusPaneOrder] = useState<"graph-first" | "code-first">("code-first");
  const [codePaneWidth, setCodePaneWidth] = useState(960);
  const [resizingFocusPanes, setResizingFocusPanes] = useState(false);
  const focusPanesRef = useRef<HTMLDivElement | null>(null);
  const focusPaneResizeRef = useRef<{ pointerId: number; startX: number; startCodePaneWidth: number; maxCodePaneWidth: number } | null>(null);
  const [canvasLevel, setCanvasLevel] = useState<CanvasLevel>("token");
  const [queryPointMode, setQueryPointMode] = useState<QueryPointMode>("tokens");
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null);
  const [lineTokenScope, setLineTokenScope] = useState<number | null>(null);
  const [dragMode, setDragMode] = useState(false);
  const [dragMatchesByCandidate, setDragMatchesByCandidate] = useState<Record<string, DragMatchBundle>>({});
  const [dragPositionsByCandidate, setDragPositionsByCandidate] = useState<Record<string, Record<string, { x: number; y: number }>>>({});
  const [dragTrailsByCandidate, setDragTrailsByCandidate] = useState<Record<string, Record<string, TrailPoint[]>>>({});
  const [dragTargetsByCandidate, setDragTargetsByCandidate] = useState<Record<string, string[]>>({});
  const [dragTargetsConfirmedByCandidate, setDragTargetsConfirmedByCandidate] = useState<Record<string, boolean>>({});
  const [externalImpactsByCandidate, setExternalImpactsByCandidate] = useState<Record<string, ExternalImpact[]>>({});
  const [selectedDragMatchKey, setSelectedDragMatchKey] = useState<string | null>(null);
  const [selectedExternalImpactKey, setSelectedExternalImpactKey] = useState<string | null>(null);
  const [resetVersion, setResetVersion] = useState(0);

  useEffect(() => {
    document.title = appMode === "study"
      ? "Interactive RAG User Study"
      : appMode === "study_dev"
        ? "Interactive RAG Study Development"
      : appMode === "baseline"
        ? "Interactive RAG Baseline"
        : "Interactive RAG";
  }, [appMode]);
  const [neighborSignals, setNeighborSignals] = useState<Record<string, NeighborSignal>>({});
  const [exitedNeighbors, setExitedNeighbors] = useState<string[]>([]);
  const previousNeighborsRef = useRef<Record<string, { label: string; distance: number }> | null>(null);
  const previousFocusRef = useRef<string | null>(null);
  const [attribution, setAttribution] = useState<TokenPairAttribution | null>(null);
  const [attributionLoading, setAttributionLoading] = useState(false);
  const [attributionError, setAttributionError] = useState<string | null>(null);
  const [gradientAttribution, setGradientAttribution] = useState<GradientAttribution | null>(null);
  const [gradientLoading, setGradientLoading] = useState(false);
  const [gradientError, setGradientError] = useState<string | null>(null);
  const [crossSampleBatchLimit, setCrossSampleBatchLimit] = useState(3);
  const [loading, setLoading] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retrievalLocked, setRetrievalLocked] = useState(false);
  const [generationMode, setGenerationMode] = useState(false);
  const [generationConfirmation, setGenerationConfirmation] = useState<GenerationConfirmation | null>(null);
  const [generationOpeningCandidate, setGenerationOpeningCandidate] = useState<CandidateDetail | null>(null);
  const [generationResult, setGenerationResult] = useState<GenerationResult | null>(null);
  const [generationComparison, setGenerationComparison] = useState<GenerationComparison | null>(null);
  const [generationEvaluation, setGenerationEvaluation] = useState<GenerationEvaluation | null>(null);
  const [generationLoading, setGenerationLoading] = useState(false);
  const [referenceFinalized, setReferenceFinalized] = useState(false);
  const [confirmReferenceOpen, setConfirmReferenceOpen] = useState(false);
  const [postTaskMeasuresOpen, setPostTaskMeasuresOpen] = useState(false);
  const [studySession, setStudySession] = useState<StudySession | null>(null);
  const [caseAttemptId, setCaseAttemptId] = useState<string | null>(null);
  const [taskBrief, setTaskBrief] = useState<TaskBrief | null>(null);
  const [briefConfirmed, setBriefConfirmed] = useState(true);
  const [briefWorkspaceReady, setBriefWorkspaceReady] = useState(false);
  const [briefLoading, setBriefLoading] = useState(false);
  const [briefOpen, setBriefOpen] = useState(false);
  const [referenceHint, setReferenceHint] = useState<ReferenceHint | null>(null);
  const [hintLoading, setHintLoading] = useState(false);
  const hintOpenedAtRef = useRef<number | null>(null);
  const loadRequestRef = useRef(0);
  const dragRerankRequestRef = useRef(0);
  const generationPreviewRequestRef = useRef(0);

  useEffect(() => {
    if (isDemo) return;
    const raw = window.sessionStorage.getItem(`irag-study-session-${appMode}`);
    if (!raw) return;
    try {
      const restored = JSON.parse(raw) as StudySession;
      setStudySession(restored);
      setEventContext(restored);
    } catch { window.sessionStorage.removeItem(`irag-study-session-${appMode}`); }
  }, [appMode, isDemo]);

  useEffect(() => {
    getExperiments()
      .then((data) => {
        if (!isDemo) {
          const studyTests = appMode === "study_dev"
            ? ["csn_11772", "csn_8884", "csn_3846", "csn_42", "csn_9388"]
            : ["csn_11772", "csn_8884", "csn_3846", "csn_42", "csn_9388"];
          setTests(studyTests);
          setTestId((currentTestId) => studyTests.includes(currentTestId) ? currentTestId : studyTests[0]);
          return;
        }
        const focused = FOCUS_TEST_IDS.filter((id) => data.tests.includes(id));
        const visibleTests = focused.length ? focused : data.tests;
        setTests(visibleTests);
        if (testId && !visibleTests.includes(testId) && visibleTests.length) setTestId(visibleTests[0]);
      })
      .catch((err) => setError(String(err)));
  }, [appMode, isDemo]);

  function clearGradientTrace() {
    setGradientAttribution(null);
    setGradientError(null);
    setGradientLoading(false);
  }

  function resetPayloadCaches() {
    cacheGenerationRef.current += 1;
    candidateDetailCacheRef.current = {};
    graphCacheRef.current = {};
    candidateRequestCacheRef.current = {};
    graphRequestCacheRef.current = {};
  }

  function invalidateCandidatePayload(nextTestId: string, candidateId: string) {
    const detailKey = payloadCacheKey(nextTestId, candidateId);
    const graphKey = payloadCacheKey(nextTestId, candidateId, 4);
    delete candidateDetailCacheRef.current[detailKey];
    delete graphCacheRef.current[graphKey];
    delete candidateRequestCacheRef.current[detailKey];
    delete graphRequestCacheRef.current[graphKey];
  }

  function payloadCacheKey(nextTestId: string, candidateId: string, epoch = 4) {
    return `${modelId}:${nextTestId}:${candidateId}:${epoch}`;
  }

  function getCandidateDetail(nextTestId: string, candidateId: string) {
    const key = payloadCacheKey(nextTestId, candidateId);
    const cached = candidateDetailCacheRef.current[key];
    if (cached) return Promise.resolve(cached);
    const pending = candidateRequestCacheRef.current[key];
    if (pending) return pending;
    const generation = cacheGenerationRef.current;
    const request = loadCandidate(nextTestId, candidateId, modelId)
      .then((detail) => {
        if (generation === cacheGenerationRef.current) candidateDetailCacheRef.current[key] = detail;
        return detail;
      })
      .finally(() => { delete candidateRequestCacheRef.current[key]; });
    candidateRequestCacheRef.current[key] = request;
    return request;
  }

  function getVisualizationGraph(nextTestId: string, candidateId: string, epoch = 4) {
    const key = payloadCacheKey(nextTestId, candidateId, epoch);
    const cached = graphCacheRef.current[key];
    if (cached) return Promise.resolve(cached);
    const pending = graphRequestCacheRef.current[key];
    if (pending) return pending;
    const generation = cacheGenerationRef.current;
    const request = loadGraph(nextTestId, candidateId, epoch, modelId)
      .then((nextGraph) => {
        if (generation === cacheGenerationRef.current) graphCacheRef.current[key] = nextGraph;
        return nextGraph;
      })
      .finally(() => { delete graphRequestCacheRef.current[key]; });
    graphRequestCacheRef.current[key] = request;
    return request;
  }

  async function prefetchRankedCandidates(payload: SessionPayload, firstCandidateId: string, generation: number) {
    const ordered = [
      ...payload.candidates.filter((item) => item.isGroundTruth && item.id !== firstCandidateId),
      ...payload.candidates.filter((item) => !item.isGroundTruth && item.id !== firstCandidateId)
    ].filter((item) => {
      const detailKey = payloadCacheKey(payload.testId, item.id);
      return !candidateDetailCacheRef.current[detailKey] || !graphCacheRef.current[payloadCacheKey(payload.testId, item.id)];
    });
    let cursor = 0;
    const warmOne = async () => {
      while (generation === cacheGenerationRef.current && cursor < ordered.length) {
        const item = ordered[cursor++];
        try {
          await Promise.all([
            getCandidateDetail(payload.testId, item.id),
            getVisualizationGraph(payload.testId, item.id)
          ]);
        } catch {
          // A failed background preload must not block the interactive session.
        }
      }
    };
    await Promise.all(Array.from({ length: CANDIDATE_PREFETCH_CONCURRENCY }, warmOne));
  }

  async function loadAll(nextTestId = testId) {
    const requestId = ++loadRequestRef.current;
    ++dragRerankRequestRef.current;
    resetPayloadCaches();
    const cacheGeneration = cacheGenerationRef.current;
    setLoading(true);
    setGraphLoading(false);
    setLoadingStep(`Resetting ${nextTestId}`);
    setError(null);
    setCandidate(null);
    setActiveCandidateId(null);
    setGraph(null);
    setRetrievalLocked(false);
    setGenerationMode(false);
    setGenerationConfirmation(null);
    setGenerationResult(null);
    setGenerationComparison(null);
    setGenerationEvaluation(null);
    setReferenceFinalized(false);
    setConfirmReferenceOpen(false);
    setPostTaskMeasuresOpen(false);
    setReferenceHint(null);
    setTaskBrief(null);
    setBriefConfirmed(true);
    setBriefWorkspaceReady(false);
    setBriefLoading(true);
    setBriefOpen(false);
    const attemptId = createClientId("attempt");
    setCaseAttemptId(attemptId);
    setEventContext({ ...(studySession ?? {}), caseAttemptId: attemptId });
    try {
      setLoadingStep(`Loading task brief for ${nextTestId}`);
      const briefRequest = loadTaskBrief(nextTestId).catch(() => null);
      const bootstrapRequest = loadSessionBootstrap(nextTestId, modelId);
      const brief = await briefRequest;
      if (requestId !== loadRequestRef.current) return;
      setBriefLoading(false);
      if (brief) {
        setTaskBrief(brief);
        setBriefConfirmed(false);
        logEvent("scenario_open", { testId: nextTestId, scenarioVersion: brief.version });
      }
      setLoadingStep(`Preparing retrieval workspace for ${nextTestId}`);
      const bootstrap = await bootstrapRequest;
      if (requestId !== loadRequestRef.current) return;
      const loaded = bootstrap.session;
      setSession(loaded);
      setCandidates(loaded.candidates);
      const first = loaded.candidates[0];
      if (!first) throw new Error("No candidates available for this test.");
      candidateDetailCacheRef.current[payloadCacheKey(nextTestId, first.id)] = bootstrap.candidate;
      graphCacheRef.current[payloadCacheKey(nextTestId, first.id)] = bootstrap.graph;
      Object.entries(bootstrap.prefetched?.candidates ?? {}).forEach(([candidateId, detail]) => {
        candidateDetailCacheRef.current[payloadCacheKey(nextTestId, candidateId)] = detail;
      });
      Object.entries(bootstrap.prefetched?.graphs ?? {}).forEach(([candidateId, nextGraph]) => {
        graphCacheRef.current[payloadCacheKey(nextTestId, candidateId)] = nextGraph;
      });
      setCandidate(bootstrap.candidate);
      setActiveCandidateId(first.id);
      setGraph(bootstrap.graph);
      setBriefWorkspaceReady(true);
      setAdjudicationIds([]);
      setAdjudicationMode(false);
      setAdjudicationCandidates({});
      setSelectedConcepts([]);
      setSelectedLines([]);
      setSelectedTokenIds([]);
      setSelectedManualLinkId(null);
      setManualLinks([]);
      setManualLinkStore({});
      setDragMatchesByCandidate({});
      setDragPositionsByCandidate({});
      setDragTrailsByCandidate({});
      setDragTargetsByCandidate({});
      setExternalImpactsByCandidate({});
      setSelectedDragMatchKey(null);
      setLinkDraft(null);
      setPlaying(false);
      setCanvasLevel("token");
      setSelectedBlockId(null);
      setLineTokenScope(null);
      setDragMode(false);
      setNeighborSignals({});
      setExitedNeighbors([]);
      setAttribution(null);
      setAttributionError(null);
      setAttributionLoading(false);
      clearGradientTrace();
      previousNeighborsRef.current = null;
      setLoading(false);
      setLoadingStep(null);
      logEvent("task_start", { testId: nextTestId, candidateId: first.id, retrievalModel: modelId });
      window.setTimeout(() => void prefetchRankedCandidates(loaded, first.id, cacheGeneration), 250);
    } catch (err) {
      setError(isUnavailableTokenError(err) ? null : err instanceof Error ? err.message : String(err));
      if (requestId === loadRequestRef.current) setLoadingStep(null);
    } finally {
      if (requestId === loadRequestRef.current) {
        setLoading(false);
        setBriefLoading(false);
      }
    }
  }

  async function selectCandidate(next: CandidateSummary) {
    if (!session || retrievalLocked || !briefConfirmed) return;
    if (activeCandidateId === next.id) return;
    if (candidate) logEvent("candidate_close", { testId: session.testId, candidateId: candidate.id });
    const requestId = ++loadRequestRef.current;
    ++dragRerankRequestRef.current;
    const previousCandidateId = candidate?.id ?? null;
    const hasDragEdits = Object.values(dragPositionsByCandidate).some((positions) => Object.keys(positions).length > 0)
      || Object.values(dragTrailsByCandidate).some((trails) => Object.values(trails).some((trail) => trail.length > 1));
    flushSync(() => {
      setActiveCandidateId(next.id);
      setLoading(true);
      if (hasDragEdits) setCanvasLevel("token");
    });
    setGraphLoading(false);
    setError(null);
    try {
      const [detail, nextGraph] = await Promise.all([
        getCandidateDetail(session.testId, next.id),
        getVisualizationGraph(session.testId, next.id)
      ]);
      if (requestId !== loadRequestRef.current) return;
      setCandidate(detail);
      setGraph(nextGraph);
      setSelectedConcepts([]);
      setSelectedLines([]);
      setSelectedTokenIds([]);
      setSelectedManualLinkId(null);
      setSelectedDragMatchKey(null);
      setManualLinks(manualLinkStore[manualLinkKey(session.testId, next.id)] ?? []);
      setLinkDraft(null);
      setPlaying(false);
      setCanvasLevel("token");
      setSelectedBlockId(null);
      setLineTokenScope(null);
      setDragMode(false);
      setNeighborSignals({});
      setExitedNeighbors([]);
      setAttribution(null);
      setAttributionError(null);
      setAttributionLoading(false);
      clearGradientTrace();
      previousNeighborsRef.current = null;
      setLoading(false);
      setLoadingStep(null);
      logEvent("candidate_switch", { testId: session.testId, candidateId: next.id, initialRank: next.originalRank ?? next.rank, currentRank: next.rank, retrievalScore: next.similarity });
      logEvent("candidate_open", { testId: session.testId, candidateId: next.id, initialRank: next.originalRank ?? next.rank, currentRank: next.rank, retrievalScore: next.similarity });
    } catch (err) {
      if (requestId === loadRequestRef.current) setActiveCandidateId(previousCandidateId);
      setError(isUnavailableTokenError(err) ? null : err instanceof Error ? err.message : String(err));
      if (requestId === loadRequestRef.current) setLoadingStep(null);
    } finally {
      if (requestId === loadRequestRef.current) setLoading(false);
    }
  }

  function toggleAdjudicationCandidate(next: CandidateSummary) {
    if (!session) return;
    setAdjudicationIds((current) => {
      if (current.includes(next.id)) return current.filter((id) => id !== next.id);
      return [...current.slice(-1), next.id];
    });
  }

  function toggleAdjudicationMode() {
    setAdjudicationMode((current) => {
      const next = !current;
      if (next) {
        setDragMode(false);
        setLinkMode(false);
        setLinkDraft(null);
        setAdjudicationIds([]);
        setAdjudicationCandidates({});
      }
      return next;
    });
  }

  function boundedCodePaneWidth(nextWidth: number) {
    const workspaceWidth = focusPanesRef.current?.getBoundingClientRect().width ?? 0;
    const maxCodePaneWidth = workspaceWidth > 0 ? Math.max(280, Math.min(1100, workspaceWidth - 372)) : 1100;
    return clamp(nextWidth, 280, maxCodePaneWidth);
  }

  function beginFocusPaneResize(event: React.PointerEvent<HTMLDivElement>) {
    const workspaceWidth = focusPanesRef.current?.getBoundingClientRect().width ?? 0;
    if (workspaceWidth <= 0) return;
    const maxCodePaneWidth = Math.max(280, Math.min(1100, workspaceWidth - 372));
    focusPaneResizeRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startCodePaneWidth: codePaneWidth,
      maxCodePaneWidth,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setResizingFocusPanes(true);
  }

  function moveFocusPaneResize(event: React.PointerEvent<HTMLDivElement>) {
    const resize = focusPaneResizeRef.current;
    if (!resize || resize.pointerId !== event.pointerId) return;
    setCodePaneWidth(clamp(resize.startCodePaneWidth + event.clientX - resize.startX, 280, resize.maxCodePaneWidth));
  }

  function endFocusPaneResize(event: React.PointerEvent<HTMLDivElement>) {
    if (focusPaneResizeRef.current?.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    focusPaneResizeRef.current = null;
    setResizingFocusPanes(false);
  }

  function adjustCodePaneWidth(delta: number) {
    setCodePaneWidth((current) => boundedCodePaneWidth(current + delta));
  }

  useEffect(() => {
    const paneContainer = focusPanesRef.current;
    if (!paneContainer) return;
    const constrainPaneWidth = () => setCodePaneWidth((current) => boundedCodePaneWidth(current));
    constrainPaneWidth();
    const observer = new ResizeObserver(constrainPaneWidth);
    observer.observe(paneContainer);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!session || adjudicationIds.length !== 2) return;
    let cancelled = false;
    Promise.all(adjudicationIds.map(async (candidateId) => ({
      candidateId,
      detail: await getCandidateDetail(session.testId, candidateId)
    })))
      .then((items) => {
        if (cancelled) return;
        setAdjudicationCandidates(Object.fromEntries(items.map((item) => [item.candidateId, item.detail])));
        logEvent("candidate_adjudication_open", { testId: session.testId, candidateIds: adjudicationIds });
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => { cancelled = true; };
  }, [session, adjudicationIds]);

  function selectConcept(conceptId: number) {
    setSelectedManualLinkId(null);
    setSelectedDragMatchKey(null);
    setSelectedTokenIds([]);
    setSelectedLines([]);
    setSelectedConcepts((current) =>
      current.includes(conceptId) ? current.filter((id) => id !== conceptId) : [...current, conceptId]
    );
    logEvent("concept_select", { testId: session?.testId, candidateId: candidate?.id, conceptId });
  }

  function selectLine(lineNumber: number) {
    setSelectedManualLinkId(null);
    setSelectedDragMatchKey(null);
    setSelectedTokenIds([]);
    setSelectedConcepts([]);
    setSelectedLines((current) =>
      current.includes(lineNumber) ? current.filter((id) => id !== lineNumber) : [...current, lineNumber]
    );
    logEvent("code_line_select", { testId: session?.testId, candidateId: candidate?.id, lineNumber });
  }

  async function selectEpoch(epoch: number) {
    if (!session || !candidate) return;
    setLoading(true);
    setError(null);
    try {
      const nextGraph = await getVisualizationGraph(session.testId, candidate.id, epoch);
      setGraph(nextGraph);
      logEvent("epoch_select", { testId: session.testId, candidateId: candidate.id, epoch });
    } catch (err) {
      setError(isUnavailableTokenError(err) ? null : err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!playing || !graph || loading) return undefined;
    const timer = window.setTimeout(() => {
      const epochs = graph.availableEpochs?.length ? graph.availableEpochs : [1, 2, 3, 4];
      const current = Math.max(0, epochs.indexOf(graph.epoch));
      const next = epochs[(current + 1) % epochs.length];
      void selectEpoch(next);
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [playing, graph?.epoch, graph?.availableEpochs, loading]);

  useEffect(() => {
    const focusId = selectedTokenIds[0] ?? null;
    if (!graph || !focusId || !playing) {
      setNeighborSignals({});
      setExitedNeighbors([]);
      previousNeighborsRef.current = null;
      previousFocusRef.current = focusId;
      return;
    }
    if (previousFocusRef.current !== focusId) {
      previousNeighborsRef.current = null;
      previousFocusRef.current = focusId;
    }
    const current = nearestNeighbors(graph, focusId, 5);
    const currentMap: Record<string, { label: string; distance: number }> = {};
    current.forEach((item) => {
      currentMap[item.node.id] = { label: item.node.label, distance: item.distance };
    });
    const previous = previousNeighborsRef.current;
    if (!previous) {
      const initialSignals: Record<string, NeighborSignal> = {};
      current.forEach((item) => {
        initialSignals[item.node.id] = { status: "stable", distance: item.distance, delta: 0 };
      });
      setNeighborSignals(initialSignals);
      setExitedNeighbors([]);
    } else {
      const nextSignals: Record<string, NeighborSignal> = {};
      current.forEach((item) => {
        const old = previous[item.node.id];
        if (!old) {
          nextSignals[item.node.id] = { status: "new", distance: item.distance, delta: 0 };
          return;
        }
        const delta = item.distance - old.distance;
        nextSignals[item.node.id] = {
          status: Math.abs(delta) < 8 ? "stable" : delta < 0 ? "closer" : "farther",
          distance: item.distance,
          delta
        };
      });
      setNeighborSignals(nextSignals);
      setExitedNeighbors(Object.entries(previous).filter(([id]) => !currentMap[id]).map(([, item]) => item.label));
    }
    previousNeighborsRef.current = currentMap;
  }, [graph?.epoch, selectedTokenIds[0], playing]);

  useEffect(() => {
    if (!session || !candidate || !graph) {
      setAttribution(null);
      setAttributionError(null);
      setAttributionLoading(false);
      clearGradientTrace();
      return undefined;
    }
    const pair = selectedQueryCodePair(graph, selectedTokenIds);
    if (!pair) {
      setAttribution(null);
      setAttributionError(null);
      setAttributionLoading(false);
      clearGradientTrace();
      return undefined;
    }
    if (session.capabilities?.intervention === false) {
      setAttribution(null);
      setAttributionLoading(false);
      setAttributionError("Training and intervention traces are available for the XSearch adapter. CodeBERT remains a representation-level semantic diagnostic view.");
      clearGradientTrace();
      return undefined;
    }
    let cancelled = false;
    setAttributionLoading(true);
    setAttributionError(null);
    loadTokenPairAttribution({
      testId: session.testId,
      candidateId: candidate.id,
      queryTokenIndex: pair.queryNode.tokenIndex,
      codeTokenIndex: pair.codeNode.tokenIndex,
      epoch: graph.epoch,
      topK: 5
    })
      .then((result) => {
        if (!cancelled) setAttribution(result);
      })
      .catch((err) => {
        if (!cancelled) setAttributionError(isUnavailableTokenError(err) ? null : err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setAttributionLoading(false);
      });
    setGradientAttribution(null);
    setGradientError(null);
    return () => {
      cancelled = true;
    };
  }, [session?.testId, candidate?.id, graph?.epoch, selectedTokenIds.join("|")]);

  function toggleToken(node: GraphNode) {
    setSelectedManualLinkId(null);
    setSelectedDragMatchKey(null);
    setSelectedConcepts([]);
    setSelectedLines([]);
    setSelectedTokenIds((current) =>
      current.includes(node.id) ? current.filter((id) => id !== node.id) : [...current, node.id].slice(-6)
    );
    logEvent("token_select", { testId: session?.testId, candidateId: candidate?.id, nodeId: node.id, type: node.type });
  }

  function updateDragTargets(ids: string[]) {
    if (!candidate) return;
    setDragTargetsByCandidate((current) => ({ ...current, [candidate.id]: ids }));
    setDragTargetsConfirmedByCandidate((current) => ({ ...current, [candidate.id]: false }));
    setSelectedTokenIds([]);
    logEvent("drag_target_select", { testId: session?.testId, candidateId: candidate.id, targetIds: ids });
  }

  function toggleDragMode() {
    if (retrievalLocked || session?.capabilities?.intervention === false) return;
    setDragMode((current) => {
      const next = !current;
      if (next && candidate) {
        setError(null);
        setDragTargetsConfirmedByCandidate((confirmed) => ({ ...confirmed, [candidate.id]: false }));
      }
      return next;
    });
    setLinkMode(false);
    setLinkDraft(null);
  }

  function confirmDragTargets(confirmed: boolean) {
    if (!candidate) return;
    setDragTargetsConfirmedByCandidate((current) => ({ ...current, [candidate.id]: confirmed }));
    logEvent("drag_target_confirm", { testId: session?.testId, candidateId: candidate.id, confirmed });
  }

  function selectTokenId(tokenId: string) {
    setSelectedManualLinkId(null);
    setSelectedDragMatchKey(null);
    setSelectedConcepts([]);
    setSelectedLines([]);
    setSelectedTokenIds((current) =>
      current.includes(tokenId) ? current.filter((id) => id !== tokenId) : [...current, tokenId].slice(-6)
    );
    logEvent("token_select", { testId: session?.testId, candidateId: candidate?.id, nodeId: tokenId, source: "side_panel" });
  }

  async function handleGraphNode(node: GraphNode) {
    if (!linkMode) {
      toggleToken(node);
      return;
    }
    if (!linkDraft) {
      setLinkDraft(node);
      setSelectedTokenIds((current) => [...new Set([...current, node.id])].slice(-6));
      return;
    }
    if (linkDraft.type === node.type) {
      setLinkDraft(node);
      setSelectedTokenIds((current) => [...new Set([...current.filter((id) => id !== linkDraft.id), node.id])].slice(-6));
      return;
    }
    if (!session || !candidate || !graph) return;
    const queryNode = linkDraft.type === "query_token" ? linkDraft : node;
    const codeNode = linkDraft.type === "code_token" ? linkDraft : node;
    const distance = nodeDistance(queryNode, codeNode);
    setLoading(true);
    setError(null);
    try {
      const result = await createManualLink({
        testId: session.testId,
        candidateId: candidate.id,
        queryTokenIndex: queryNode.tokenIndex,
        codeTokenIndex: codeNode.tokenIndex,
        epoch: graph.epoch,
        distance,
        color: queryNode.color || codeNode.color
      });
      setManualLinks(result.links);
      setManualLinkStore((current) => {
        const nextStore = { ...current };
        if (result.linksByCandidate) {
          Object.entries(result.linksByCandidate).forEach(([candidateId, links]) => {
            nextStore[manualLinkKey(session.testId, candidateId)] = links;
          });
        } else {
          nextStore[manualLinkKey(session.testId, candidate.id)] = result.links;
        }
        return nextStore;
      });
      setCandidates(result.candidates);
      resetPayloadCaches();
      setSelectedTokenIds((current) => [...new Set([...current, queryNode.id, codeNode.id])].slice(-6));
      setSelectedManualLinkId(result.link?.id ?? null);
      setLinkDraft(null);
      logEvent("manual_link", { testId: session.testId, candidateId: candidate.id, queryTokenIndex: queryNode.tokenIndex, codeTokenIndex: codeNode.tokenIndex, distance });
    } catch (err) {
      setError(isUnavailableTokenError(err) ? null : err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleGraphDrop(payload: { node: GraphNode; updates: Array<{ nodeId: string; similarity: number }>; positions: Record<string, { x: number; y: number }> }) {
    if (retrievalLocked || session?.capabilities?.intervention === false) return;
    if (!session || !candidate) return;
    if (!graph) return;
    const targetIds = dragTargetsByCandidate[candidate.id] ?? [];
    if (!targetIds.length) {
      setError("Select at least one target token before dragging.");
      return;
    }
    const matches = computeDragMatches(graph, candidate, session, payload.positions, payload.node, targetIds);
    setDragMatchesByCandidate((current) => ({ ...current, [candidate.id]: matches }));
    setSelectedDragMatchKey(null);
    if (!matches.pairInterventions.length) return;
    const requestId = ++dragRerankRequestRef.current;
    const requestTestId = session.testId;
    const requestCandidateId = candidate.id;
    setLoading(true);
    setError(null);
    try {
      const result = await applyDragRerank({
        model: modelId,
        testId: session.testId,
        candidateId: candidate.id,
        draggedNode: { id: payload.node.id, type: payload.node.type, tokenIndex: payload.node.tokenIndex },
        pairInterventions: matches.pairInterventions
      });
      if (requestId !== dragRerankRequestRef.current) return;
      if (!result.candidates.length) {
        throw new Error("The rerank response did not include any candidates.");
      }
      setCandidates(result.candidates);
      resetPayloadCaches();
      let generalized: Record<string, DragMatchBundle> = {};
      if (session.capabilities?.external_effects !== false) {
        generalized = buildGeneralizedVisualMatches(result.generalizedMatchesByCandidate, session);
        const externalImpacts = buildExternalImpacts(result.generalizedMatchesByCandidate, candidate.id, targetIds, session);
        setExternalImpactsByCandidate(externalImpacts);
      } else {
        setExternalImpactsByCandidate({});
      }
      const backendLocal = result.localMatches;
      const localLineMatches = backendLocal?.lineMatches
        ? backendLocal.lineMatches as DragLineMatch[]
        : matches.lineMatches.map((match) => ({ ...match, source: "local_drag" as const })).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 3);
      setDragMatchesByCandidate((current) => ({
        ...current,
        ...generalized,
        [candidate.id]: {
          tokenMatches: backendLocal?.tokenMatches ? backendLocal.tokenMatches as DragTokenMatch[] : matches.tokenMatches,
          lineMatches: localLineMatches
        }
      }));
      if (result.activeCandidate) setCandidate(result.activeCandidate);
      if (result.activeGraph) setGraph(result.activeGraph);
      setActiveCandidateId(requestCandidateId);
      Object.entries(result.updatedCandidates ?? {}).forEach(([candidateId, detail]) => {
        candidateDetailCacheRef.current[payloadCacheKey(requestTestId, candidateId)] = detail;
      });
      Object.entries(result.updatedGraphs ?? {}).forEach(([candidateId, nextGraph]) => {
        graphCacheRef.current[payloadCacheKey(requestTestId, candidateId)] = nextGraph;
      });
      logEvent("projection_drag_drop", {
        testId: session.testId,
        candidateId: requestCandidateId,
        nodeId: payload.node.id,
        nodeType: payload.node.type,
        affectedNodes: payload.updates.length,
        pairInterventions: matches.pairInterventions.length,
        source: result.diagnostic.source
      });
    } catch (err) {
      if (requestId !== dragRerankRequestRef.current) return;
      setError(isUnavailableTokenError(err) ? null : err instanceof Error ? err.message : String(err));
    } finally {
      if (requestId === dragRerankRequestRef.current) setLoading(false);
    }
  }

  const neighborMode = playing && selectedTokenIds.length > 0;
  const currentDragMatches = candidate ? dragMatchesByCandidate[candidate.id] ?? { tokenMatches: [], lineMatches: [] } : { tokenMatches: [], lineMatches: [] };
  const currentCandidateSummary = candidate ? candidates.find((item) => item.id === candidate.id) : undefined;
  const currentDragTargets = candidate ? dragTargetsByCandidate[candidate.id] ?? [] : [];
  const currentDragTargetsConfirmed = candidate ? dragTargetsConfirmedByCandidate[candidate.id] ?? false : false;
  const currentExternalImpacts = candidate ? externalImpactsByCandidate[candidate.id] ?? [] : [];
  const supportsIntervention = session?.capabilities?.intervention !== false;

  async function confirmReferenceForGeneration(selectedCandidate?: CandidateDetail) {
    const referenceCandidate = selectedCandidate ?? candidate;
    if (!session || !referenceCandidate) return;
    const previousReferenceId = generationConfirmation?.selection.selectedCandidateId;
    const referenceChanged = previousReferenceId !== referenceCandidate.id;
    if (!referenceChanged && generationConfirmation) {
      setGenerationMode(true);
      setAdjudicationMode(false);
      return;
    }
    const requestId = ++generationPreviewRequestRef.current;
    flushSync(() => {
      setGenerationMode(true);
      setAdjudicationMode(false);
      setGenerationLoading(true);
      setGenerationOpeningCandidate(referenceCandidate);
      setGenerationConfirmation(null);
      setGenerationResult(null);
      setGenerationComparison(null);
      setGenerationEvaluation(null);
      setReferenceHint(null);
      setReferenceFinalized(false);
      setConfirmReferenceOpen(false);
    });
    setError(null);
    try {
      const summary = candidates.find((item) => item.id === referenceCandidate.id);
      const interactionUsed = Object.keys(dragPositionsByCandidate).some((candidateId) => Object.keys(dragPositionsByCandidate[candidateId]).length > 0)
        || Object.keys(dragMatchesByCandidate).length > 0;
      const confirmation = await confirmReference({
        caseId: session.testId,
        selectedCandidateId: referenceCandidate.id,
        selectedRank: summary?.rank ?? null,
        selectedScore: summary?.similarity ?? referenceCandidate.similarity,
        interactionUsed,
        model: modelId,
        appMode,
        sessionId: studySession?.sessionId,
        participantId: studySession?.participantId,
        caseAttemptId: caseAttemptId ?? undefined
      });
      if (requestId !== generationPreviewRequestRef.current) return;
      setGenerationConfirmation(confirmation);
      setGenerationOpeningCandidate(null);
      setDragMode(false);
      setLinkMode(false);
      setAdjudicationMode(false);
      logEvent("reference_preview", { testId: session.testId, candidateId: referenceCandidate.id, selectedRank: summary?.rank, interactionUsed, source: selectedCandidate ? "adjudication" : "toolbar" });
    } catch (err) {
      if (requestId !== generationPreviewRequestRef.current) return;
      setGenerationMode(false);
      setGenerationOpeningCandidate(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (requestId === generationPreviewRequestRef.current) setGenerationLoading(false);
    }
  }

  async function finalizeReferenceForTask() {
    if (!generationConfirmation || referenceFinalized) return;
    setGenerationLoading(true);
    setError(null);
    try {
      const confirmation = await finalizeReference(generationConfirmation.selectionId);
      setGenerationConfirmation(confirmation);
      setReferenceFinalized(true);
      setRetrievalLocked(true);
      setConfirmReferenceOpen(false);
      setGenerationResult(null);
      setGenerationComparison(null);
      setGenerationEvaluation(null);
      const selected = confirmation.selection;
      await logEvent("candidate_close", { testId: confirmation.task.caseId, candidateId: selected.selectedCandidateId, reason: "reference_finalized" });
      await logEvent("reference_confirm", { testId: confirmation.task.caseId, candidateId: selected.selectedCandidateId, selectedRank: selected.selectedRank, interactionUsed: selected.interactionUsed, selectionId: confirmation.selectionId });
      if (!isDemo) setPostTaskMeasuresOpen(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerationLoading(false);
    }
  }

  async function runGeneration(condition: "no_rag" | "interactive_rag") {
    if (!generationConfirmation) return;
    setGenerationLoading(true);
    setError(null);
    setGenerationComparison(null);
    setGenerationEvaluation(null);
    try {
      const result = await generateCode({
        caseId: generationConfirmation.task.caseId,
        selectionId: generationConfirmation.selectionId,
        condition
      });
      setGenerationResult(result);
      const evaluationRequest = evaluateGeneration(result.generationId);
      if (isDemo) {
        const [comparison, evaluation] = await Promise.all([loadGenerationComparison(result.generationId), evaluationRequest]);
        setGenerationComparison(comparison);
        setGenerationEvaluation(evaluation);
        logEvent("evaluation_complete", { caseId: result.caseId, generationId: result.generationId, status: evaluation.status, passRate: evaluation.passRate });
      } else {
        const evaluation = await evaluationRequest;
        setGenerationEvaluation(evaluation);
        logEvent("evaluation_complete", { caseId: result.caseId, generationId: result.generationId, status: evaluation.status, passRate: evaluation.passRate });
      }
      logEvent("generation_end", { caseId: result.caseId, condition: result.condition, contextCandidateId: result.contextCandidateId, generationId: result.generationId });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerationLoading(false);
    }
  }

  async function showReferenceHint() {
    if (!generationConfirmation || referenceHint) return;
    setHintLoading(true);
    try {
      const hint = await loadReferenceHint(generationConfirmation.selectionId);
      setReferenceHint(hint);
      hintOpenedAtRef.current = Date.now();
      logEvent("hint_opened", { caseId: generationConfirmation.task.caseId, selectionId: generationConfirmation.selectionId });
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setHintLoading(false); }
  }

  function confirmTaskBrief() {
    if (!taskBrief || !session) return;
    setBriefConfirmed(true);
    logEvent("scenario_confirm", { testId: session.testId, scenarioVersion: taskBrief.version });
  }

  async function startParticipant(participantId: string) {
    const created = await startStudySession(participantId, studyCondition);
    setStudySession(created);
    setEventContext(created);
    window.sessionStorage.setItem(`irag-study-session-${appMode}`, JSON.stringify(created));
    logEvent("study_session_start", { condition: created.condition });
  }

  function changeParticipant() {
    const confirmed = window.confirm("Change participant? The current participant session and task progress will be ended.");
    if (!confirmed) return;
    if (studySession) logEvent("session_end", { participantId: studySession.participantId });
    window.sessionStorage.removeItem(`irag-study-session-${appMode}`);
    setEventContext({});
    setStudySession(null);
    setSession(null);
    setCandidate(null);
    setActiveCandidateId(null);
    setActiveCandidateId(null);
    setGraph(null);
    setCandidates([]);
    setTaskBrief(null);
    setBriefConfirmed(true);
    setPostTaskMeasuresOpen(false);
  }

  async function submitPostTaskMeasures(confidence: number, difficulty: number, reason: string) {
    if (!generationConfirmation) return;
    const context = { caseId: generationConfirmation.task.caseId, selectionId: generationConfirmation.selectionId };
    await logEvent("confidence_submit", { ...context, confidence });
    await logEvent("difficulty_submit", { ...context, difficulty });
    if (hintOpenedAtRef.current) {
      await logEvent("hint_closed", { ...context, hintDwellTime: Math.round((Date.now() - hintOpenedAtRef.current) / 1000) });
      hintOpenedAtRef.current = null;
    }
    await logEvent("task_end", { ...context, selectionReason: reason });
    setPostTaskMeasuresOpen(false);
  }

  async function runGenerationEvaluation() {
    if (!generationResult) return;
    setGenerationLoading(true);
    setError(null);
    try {
      const result = await evaluateGeneration(generationResult.generationId);
      setGenerationEvaluation(result);
      logEvent("generation_evaluation", { generationId: generationResult.generationId, status: result.status, passRate: result.passRate });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerationLoading(false);
    }
  }

  function changeModel(nextModelId: string) {
    if (!isDemo) return;
    if (nextModelId === modelId) return;
    setModelId(nextModelId);
    resetPayloadCaches();
    setSession(null);
    setCandidate(null);
    setGraph(null);
    setCandidates([]);
    setDragMode(false);
    setLinkMode(false);
    setAdjudicationMode(false);
    setRetrievalLocked(false);
    setGenerationMode(false);
    setGenerationConfirmation(null);
    setGenerationResult(null);
    setGenerationComparison(null);
    setGenerationEvaluation(null);
    setReferenceFinalized(false);
    setPostTaskMeasuresOpen(false);
    setError(null);
    if (nextModelId === "codebert" && !testId.startsWith("csn_")) setTestId("csn_11087");
  }

  function handleHierarchyLine(lineNumber: number) {
    void lineNumber;
  }

  function handleHierarchyBlock(blockId: string) {
    void blockId;
  }

  function handleCanvasLevel(level: CanvasLevel) {
    void level;
    setCanvasLevel("token");
  }

  function handleCodeViewerLine(lineNumber: number) {
    selectLine(lineNumber);
  }

  function selectManualLink(link: ManualLink) {
    setSelectedManualLinkId(link.id);
    setSelectedDragMatchKey(null);
    setSelectedTokenIds([`q_tok_${link.queryTokenIndex}`, `c_tok_${link.codeTokenIndex}`]);
    setSelectedConcepts([]);
    setSelectedLines([]);
    setPlaying(false);
    setNeighborSignals({});
    setExitedNeighbors([]);
    setAttribution(null);
    setAttributionError(null);
    setAttributionLoading(false);
    clearGradientTrace();
    previousNeighborsRef.current = null;
    logEvent("manual_link_select", { testId: session?.testId, candidateId: candidate?.id, linkId: link.id });
  }

  async function resetView() {
    setSelectedLines([]);
    setSelectedTokenIds([]);
    setSelectedManualLinkId(null);
    setSelectedDragMatchKey(null);
    setSelectedConcepts([]);
    setLineTokenScope(null);
    setManualLinks([]);
    setManualLinkStore({});
      setDragMatchesByCandidate({});
      setDragPositionsByCandidate({});
      setDragTrailsByCandidate({});
      setDragTargetsByCandidate({});
      setDragTargetsConfirmedByCandidate({});
      setExternalImpactsByCandidate({});
    setLinkMode(false);
    setDragMode(false);
    setLinkDraft(null);
    setPlaying(false);
    setResetVersion((value) => value + 1);
    setNeighborSignals({});
    setExitedNeighbors([]);
    setAttribution(null);
    setAttributionError(null);
    setAttributionLoading(false);
    clearGradientTrace();
    previousNeighborsRef.current = null;
    if (session) {
      try {
        resetPayloadCaches();
        await resetInterventions(session.testId, modelId);
        const restored = await loadSession(session.testId, modelId);
        setSession(restored);
        setCandidates(restored.candidates);
        if (candidate) {
          const detail = await getCandidateDetail(session.testId, candidate.id);
          const nextGraph = await getVisualizationGraph(session.testId, candidate.id);
          setCandidate(detail);
          setGraph(nextGraph);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    }
  }

  function selectDragMatch(match: DragTokenMatch) {
    const key = dragMatchKey(match);
    const nextActive = selectedDragMatchKey !== key;
    setSelectedManualLinkId(null);
    setSelectedDragMatchKey(nextActive ? key : null);
    setSelectedExternalImpactKey(null);
    setSelectedTokenIds(nextActive ? [`q_tok_${match.queryTokenIndex}`, `c_tok_${match.codeTokenIndex}`] : []);
    setSelectedConcepts([]);
    setSelectedLines([]);
    setPlaying(false);
    setNeighborSignals({});
    setExitedNeighbors([]);
    setAttribution(null);
    setAttributionError(null);
    setAttributionLoading(false);
    clearGradientTrace();
    previousNeighborsRef.current = null;
    logEvent("drag_neighbor_select", {
      testId: session?.testId,
      candidateId: candidate?.id,
      queryTokenIndex: match.queryTokenIndex,
      codeTokenIndex: match.codeTokenIndex,
      delta: match.delta
    });
  }

  function selectExternalImpact(impact: ExternalImpact, conceptId?: number) {
    const key = externalImpactKey(impact);
    const active = selectedExternalImpactKey === key;
    const isConceptImpact = queryPointMode === "concept" && conceptId != null;
    setSelectedExternalImpactKey(active ? null : key);
    setSelectedDragMatchKey(null);
    setSelectedManualLinkId(null);
    setSelectedTokenIds(active ? [] : isConceptImpact ? [`c_tok_${impact.codeTokenIndex}`] : [`q_tok_${impact.queryTokenIndex}`, `c_tok_${impact.codeTokenIndex}`]);
    setSelectedConcepts([]);
    setSelectedLines([]);
    setPlaying(false);
    logEvent("external_impact_select", {
      testId: session?.testId,
      candidateId: candidate?.id,
      queryTokenIndex: impact.queryTokenIndex,
      codeTokenIndex: impact.codeTokenIndex,
      conceptId: conceptId ?? null,
      delta: impact.delta
    });
  }

  async function handleGradientTrace(
    pair: { queryNode: GraphNode; codeNode: GraphNode },
    lossScope: GradientAttribution["lossScope"] = "highlight_only"
  ) {
    if (!session || !candidate) return;
    setGradientLoading(true);
    setGradientError(null);
    const crossSample = lossScope === "highlight_plus_cross_sample_batch";
    try {
      const result = await runGradientAttribution({
        testId: session.testId,
        candidateId: candidate.id,
        queryTokenIndex: pair.queryNode.tokenIndex,
        codeTokenIndex: pair.codeNode.tokenIndex,
        mode: "pull",
        lossScope,
        crossSampleWeight: crossSample ? 2.0 : undefined,
        topK: 5,
        maxTrainSamples: crossSample ? Math.max(24, crossSampleBatchLimit * 8) : 24,
        maxBatches: crossSample ? crossSampleBatchLimit : undefined
      });
      setGradientAttribution(result);
      logEvent("gradient_trace", {
        testId: session.testId,
        candidateId: candidate.id,
        queryTokenIndex: pair.queryNode.tokenIndex,
        codeTokenIndex: pair.codeNode.tokenIndex,
        lossScope,
        maxBatches: crossSample ? crossSampleBatchLimit : undefined,
        status: result.status
      });
    } catch (err) {
      setGradientError(err instanceof Error ? err.message : String(err));
    } finally {
      setGradientLoading(false);
    }
  }

  if (!isDemo && !studySession) return <ParticipantGate condition={studyCondition} onStart={startParticipant} />;

  if (briefLoading || (taskBrief && !briefConfirmed)) return <div className={`app-shell app-mode-${appMode} task-brief-stage`}><header className="topbar"><h1>{appMode === "study" ? "Interactive RAG User Study" : appMode === "study_dev" ? "Interactive RAG Study Development" : appMode === "baseline" ? "Interactive RAG Baseline" : "Interactive RAG"}</h1>{!isDemo ? <button onClick={changeParticipant}>Change participant</button> : null}</header>{taskBrief ? <TaskBriefPanel brief={taskBrief} onConfirm={confirmTaskBrief} ready={briefWorkspaceReady} /> : <section className="panel task-brief-panel"><div className="panel-title">Task Brief</div><p>Loading task context...</p></section>}</div>;

  return (
    <div className={`app-shell app-mode-${appMode}`}>
      <header className="topbar">
        <div>
          <h1>{appMode === "study" ? "Interactive RAG User Study" : appMode === "study_dev" ? "Interactive RAG Study Development" : appMode === "baseline" ? "Interactive RAG Baseline" : "Interactive RAG"}</h1>
        </div>
        <div className="toolbar">
          {isDemo ? <label className="model-selector">
            <span>Model</span>
            <select value={modelId} onChange={(event) => changeModel(event.target.value)} disabled={loading}>
              <option value="xsearch">XSearch</option>
              <option value="codebert">CodeBERT</option>
            </select>
          </label> : null}
          {!isDemo ? <button onClick={changeParticipant}>Change participant</button> : null}
          {taskBrief ? <button onClick={() => setBriefOpen(true)}>Task Brief</button> : null}
          <select value={testId} onChange={(event) => setTestId(event.target.value)}>
            <option value="">Select an example</option>
            {tests.map((id) => (
              <option key={id} value={id}>
              {id === "48" ? "48 · allowed extension alignment" : id === "1556" ? "1556 · reset system state" : id === "1642" ? "1642 · compact rerank demo" : id === "2695" ? "2695 · EM iteration" : id === "2797" ? "2797 · command-line argument recovery" : id === "2836" ? "2836 · parent override logging recovery" : id === "3856" ? "3856 · comparable dictionary" : id === "954" ? "954 · API decorator specificity" : id === "csn_9848" ? "9848 · configuration return type" : id === "csn_11087" ? "11087 · right-click position" : id === "csn_11078" ? "11078 · error message display" : id === "csn_9406" ? "9406 · device buffer write" : id === "csn_400" ? "400 · parse options and commands" : id === "csn_13958" ? "13958 · line-pair diagnosis" : id === "csn_13527" ? "13527 · command-line logging" : id === "csn_8838" ? "8838 · interned keyword API bridge" : id === "csn_11772" ? "11772 · asset MIME-type extension bridge" : id === "csn_9388" ? "9388 · URL query removal" : id === "csn_8884" ? "8884 · Try AST dead-code cleanup" : id === "csn_3846" ? "3846 · decorator redefinition" : id === "csn_42" ? "42 · Cloud SQL delete completion" : id === "csn_12226" ? "12226 · alarm API result handling" : id === "csn_584" ? "584 · random perspective parameters" : id === "csn_2812" ? "2812 · qubit dimension log2 bridge" : id === "csn_7727" ? "7727 · Stokes calibration feed-type bridge" : id === "csn_4772" ? "4772 · KMIP DeviceCredential serialization bridge" : id === "csn_10023" ? "10023 · OSM replication state bridge" : id === "csn_2207" ? "2207 · window sum-square hop-length bridge" : id === "csn_5340" ? "5340 · GeoTiff VLR API bridge" : id === "csn_10164" ? "10164 · V4 meter request bridge" : id === "csn_13655" ? "13655 · application logging bridge" : id === "csn_14175" ? "14175 · notebook format bridge" : id === "csn_10643" ? "10643 · root logger bridge" : id === "csn_12075" ? "12075 · current tags API bridge" : `test ${id}`}
              </option>
            ))}
          </select>
          <button className="primary" onClick={() => loadAll()} disabled={loading || !testId}>
            {loading ? <Loader2 size={16} className="spin" /> : <CirclePlay size={16} />}
            Load
          </button>
          {!isBaseline ? <button onClick={resetView} disabled={!graph || loading}>
            <RefreshCw size={16} />
          </button> : null}
          {!isBaseline ? <button
            onClick={() => setFocusPaneOrder((value) => value === "graph-first" ? "code-first" : "graph-first")}
            title="交换 embedding space 和代码面板位置"
          >
            <ArrowLeftRight size={16} />
            Swap
          </button> : null}
          {!isBaseline ? <button
            className={dragMode ? "active-tool" : ""}
            onClick={toggleDragMode}
            disabled={!graph || !supportsIntervention || retrievalLocked}
            title={supportsIntervention ? "启用或关闭画布节点拖拽" : "Interactive representation editing is currently available for XSearch."}
          >
            <Move size={16} />
            Drag
          </button> : null}
          <button
            className={adjudicationMode ? "active-tool" : ""}
            onClick={toggleAdjudicationMode}
            disabled={!session || retrievalLocked}
            title="进入或退出两个候选的行级裁决对比"
          >
            <ArrowLeftRight size={16} />
            Adjudicate
          </button>
          <button
            className={generationMode ? "active-tool" : "reference-action"}
            onClick={generationMode ? () => setGenerationMode(false) : () => void confirmReferenceForGeneration()}
            disabled={loading || generationLoading || retrievalLocked || (!generationMode && (!session || !candidate || modelId !== "xsearch" || !session.referenceSelection?.enabled || !GENERATION_CASE_IDS.has(session.testId)))}
            title={generationMode ? "Return to retrieval and compare another reference" : modelId !== "xsearch" ? "Generation validation currently uses XSearch reference evidence." : "Use the current reference for generation; final submission happens in Generation."}
          >
            {generationMode ? <CirclePlay size={16} /> : <BookOpenCheck size={16} />}
            {generationMode ? "Retrieval" : "Use as Reference"}
          </button>
        </div>
      </header>

      {loadingStep && <div className="meta-line">{loadingStep}</div>}
      {error && <div className="error">{error}</div>}
      {briefOpen && taskBrief ? <div className="task-brief-overlay"><TaskBriefPanel brief={taskBrief} onConfirm={() => undefined} onClose={() => setBriefOpen(false)} /></div> : null}
      {confirmReferenceOpen && generationConfirmation ? <div className="reference-confirm-overlay" role="presentation">
        <section className="reference-confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="reference-confirm-title">
          <div className="panel-title" id="reference-confirm-title">Confirm Reference Selection</div>
          <p>You are about to submit this reference as your final choice for this task.</p>
          <div className="reference-confirm-summary">
            <strong>{generationConfirmation.selection.candidate.metadata.funcName || generationConfirmation.selection.selectedCandidateId}</strong>
            <span>Rank {generationConfirmation.selection.selectedRank ?? "-"} · score {generationConfirmation.selection.selectedScore?.toFixed(3) ?? "-"}</span>
          </div>
          <p className="reference-confirm-note">After confirmation, this task's reference selection is locked and cannot be changed.</p>
          <div className="reference-confirm-actions">
            <button onClick={() => setConfirmReferenceOpen(false)} disabled={generationLoading}>Cancel</button>
            <button className="primary" onClick={finalizeReferenceForTask} disabled={generationLoading}>{generationLoading ? "Submitting..." : "Confirm and Submit"}</button>
          </div>
        </section>
      </div> : null}
      {postTaskMeasuresOpen && generationConfirmation ? <div className="reference-confirm-overlay" role="presentation">
        <section className="reference-confirm-dialog post-task-measures-dialog" role="dialog" aria-modal="true" aria-labelledby="post-task-measures-title">
          <p id="post-task-measures-title">Your reference selection has been recorded. Please complete this short questionnaire to finish the task.</p>
          <PostTaskMeasures onSubmit={submitPostTaskMeasures} />
        </section>
      </div> : null}
      <div className={isBaseline ? "workspace baseline-workspace" : "workspace"}>
        {isBaseline ? <BaselineQueryPanel session={session} /> : <QueryPanel
          session={session}
          candidate={candidate}
          selectedConcepts={selectedConcepts}
          selectedTokenIds={selectedTokenIds}
          dragTokenMatches={currentDragMatches.tokenMatches}
          dragLineMatches={currentDragMatches.lineMatches}
          selectedDragMatchKey={selectedDragMatchKey}
          onConcept={selectConcept}
          onToken={selectTokenId}
          onDragMatch={selectDragMatch}
          externalImpacts={currentExternalImpacts}
          selectedExternalImpactKey={selectedExternalImpactKey}
          onExternalImpact={selectExternalImpact}
          queryPointMode={queryPointMode}
        />}
        <div className={`main-workspace${isBaseline ? " baseline-main-workspace" : ""}${adjudicationMode ? " adjudicating" : ""}`}>
          {generationMode && generationConfirmation ? (
            <GenerationPanel
              confirmation={generationConfirmation}
              result={generationResult}
              comparison={generationComparison}
              evaluation={generationEvaluation}
              loading={generationLoading}
              onGenerate={runGeneration}
              onEvaluate={runGenerationEvaluation}
              showInternal={isDemo}
              hint={referenceHint}
              hintLoading={hintLoading}
              onHint={showReferenceHint}
              showMeasures={false}
              onSubmitMeasures={submitPostTaskMeasures}
              onBack={() => setGenerationMode(false)}
              finalized={referenceFinalized}
              onFinalize={() => setConfirmReferenceOpen(true)}
              showFinalization={!isDemo}
            />
          ) : generationMode ? (
            <GenerationOpeningPanel candidate={generationOpeningCandidate} query={session?.query.rawText ?? ""} onBack={() => setGenerationMode(false)} />
          ) : (
          isBaseline ? (adjudicationMode && session ? (
            <div className="adjudication-workspace">
              <CandidatePanel
                candidates={candidates.length ? candidates : session.candidates}
                selectedId={activeCandidateId}
                onSelect={selectCandidate}
                modelId={modelId}
                showGroundTruth={false}
                adjudicationMode
                adjudicationIds={adjudicationIds}
                onAdjudicationToggle={toggleAdjudicationCandidate}
              />
              {adjudicationIds.length === 2 ? (
                <AdjudicationPanel
                  session={session}
                  candidates={adjudicationIds.map((id) => adjudicationCandidates[id]).filter((item): item is CandidateDetail => Boolean(item))}
                  summaries={Object.fromEntries((candidates.length ? candidates : session.candidates).map((item) => [item.id, item]))}
                  showGroundTruth={false}
                  showEvidence={false}
                  choosingReference={generationLoading}
                  onChooseReference={(selectedCandidate) => void confirmReferenceForGeneration(selectedCandidate)}
                />
              ) : <section className="panel adjudication-placeholder">Select two candidates from the ranked list to compare them before choosing one as a reference.</section>}
            </div>
          ) : (
            <>
              <BaselineCodeViewer candidate={candidate} displaySimilarity={currentCandidateSummary?.similarity} />
            </>
          )) : <>
          {adjudicationMode && session ? (
            <div className="adjudication-workspace">
              <CandidatePanel
                candidates={candidates.length ? candidates : session.candidates}
                selectedId={activeCandidateId}
                onSelect={selectCandidate}
                modelId={modelId}
                showGroundTruth={isDemo}
                adjudicationMode
                adjudicationIds={adjudicationIds}
                onAdjudicationToggle={toggleAdjudicationCandidate}
              />
              {adjudicationIds.length === 2 ? (
                <AdjudicationPanel
                  session={session}
                  candidates={adjudicationIds.map((id) => adjudicationCandidates[id]).filter((item): item is CandidateDetail => Boolean(item))}
                  summaries={Object.fromEntries((candidates.length ? candidates : session.candidates).map((item) => [item.id, item]))}
                  showGroundTruth={isDemo}
                  showEvidence
                  showUncoveredEvidence={isDemo}
                  choosingReference={generationLoading}
                  onChooseReference={(selectedCandidate) => void confirmReferenceForGeneration(selectedCandidate)}
                />
              ) : (
                <section className="panel adjudication-placeholder">Select two candidates from the ranked list to compare their line-level evidence.</section>
              )}
            </div>
          ) : <>
          <div
            ref={focusPanesRef}
            className={`focus-panes ${focusPaneOrder}${resizingFocusPanes ? " resizing" : ""}`}
            style={{ "--code-pane-width": `${codePaneWidth}px` } as React.CSSProperties}
          >
            {focusPaneOrder === "graph-first" ? (
              <>
                <VisualizationCanvas
                  candidate={candidate}
                  session={session}
                  canvasLevel={canvasLevel}
                  selectedBlockId={selectedBlockId}
                  lineTokenScope={lineTokenScope}
                  queryPointMode={queryPointMode}
                  onCanvasLevel={handleCanvasLevel}
                  onQueryPointModeChange={setQueryPointMode}
                  onBlock={handleHierarchyBlock}
                  onHierarchyLine={handleHierarchyLine}
                  onHierarchyConcept={selectConcept}
                  graph={graph}
                  selectedConcepts={selectedConcepts}
                  selectedLines={selectedLines}
                  selectedTokenIds={selectedTokenIds}
                  selectedManualLinkId={selectedManualLinkId}
                  manualLinks={manualLinks}
                  neighborMode={neighborMode}
                  neighborSignals={neighborSignals}
                  linkMode={linkMode}
                  linkDraft={linkDraft}
                  loading={loading || graphLoading}
                  onNode={handleGraphNode}
                  onDrop={handleGraphDrop}
                  dragMode={dragMode}
                  resetKey={resetVersion}
                  dragTokenMatches={currentDragMatches.tokenMatches}
                  selectedDragMatchKey={selectedDragMatchKey}
                  dragTargetIds={currentDragTargets}
                  onDragTargetChange={updateDragTargets}
                  dragTargetsConfirmed={currentDragTargetsConfirmed}
                  selectedExternalImpactKey={selectedExternalImpactKey}
                  onDragTargetsConfirm={confirmDragTargets}
                  persistedPositions={candidate ? dragPositionsByCandidate[candidate.id] ?? {} : {}}
                  onPositionsChange={(positions) => {
                    if (!candidate) return;
                    setDragPositionsByCandidate((current) => ({ ...current, [candidate.id]: positions }));
                  }}
                  persistedTrails={candidate ? dragTrailsByCandidate[candidate.id] ?? {} : {}}
                  onTrailsChange={(trails) => {
                    if (!candidate) return;
                    setDragTrailsByCandidate((current) => ({ ...current, [candidate.id]: trails }));
                  }}
                  externalImpacts={currentExternalImpacts}
                />
                <div
                  className="pane-resize-handle"
                  role="separator"
                  aria-label="Resize Code Viewer and Embedding Space"
                  aria-orientation="vertical"
                  aria-valuemin={280}
                  aria-valuenow={Math.round(codePaneWidth)}
                  tabIndex={0}
                  onPointerDown={beginFocusPaneResize}
                  onPointerMove={moveFocusPaneResize}
                  onPointerUp={endFocusPaneResize}
                  onPointerCancel={endFocusPaneResize}
                  onKeyDown={(event) => {
                    if (event.key === "ArrowLeft") { event.preventDefault(); adjustCodePaneWidth(-32); }
                    if (event.key === "ArrowRight") { event.preventDefault(); adjustCodePaneWidth(32); }
                  }}
                />
                <CodeViewer candidate={candidate} session={session} graph={graph} canvasLevel={canvasLevel} selectedBlockId={selectedBlockId} displaySimilarity={currentCandidateSummary?.similarity} selectedConcepts={selectedConcepts} selectedLines={selectedLines} selectedTokenIds={selectedTokenIds} manualLinks={manualLinks} dragTokenMatches={currentDragMatches.tokenMatches} dragLineMatches={currentDragMatches.lineMatches} allowPlainText={appMode === "study" || appMode === "study_dev"} onBlock={handleHierarchyBlock} onLine={handleCodeViewerLine} onToken={selectTokenId} />
              </>
            ) : (
              <>
                <CodeViewer candidate={candidate} session={session} graph={graph} canvasLevel={canvasLevel} selectedBlockId={selectedBlockId} displaySimilarity={currentCandidateSummary?.similarity} selectedConcepts={selectedConcepts} selectedLines={selectedLines} selectedTokenIds={selectedTokenIds} manualLinks={manualLinks} dragTokenMatches={currentDragMatches.tokenMatches} dragLineMatches={currentDragMatches.lineMatches} allowPlainText={appMode === "study" || appMode === "study_dev"} onBlock={handleHierarchyBlock} onLine={handleCodeViewerLine} onToken={selectTokenId} />
                <div
                  className="pane-resize-handle"
                  role="separator"
                  aria-label="Resize Code Viewer and Embedding Space"
                  aria-orientation="vertical"
                  aria-valuemin={280}
                  aria-valuenow={Math.round(codePaneWidth)}
                  tabIndex={0}
                  onPointerDown={beginFocusPaneResize}
                  onPointerMove={moveFocusPaneResize}
                  onPointerUp={endFocusPaneResize}
                  onPointerCancel={endFocusPaneResize}
                  onKeyDown={(event) => {
                    if (event.key === "ArrowLeft") { event.preventDefault(); adjustCodePaneWidth(-32); }
                    if (event.key === "ArrowRight") { event.preventDefault(); adjustCodePaneWidth(32); }
                  }}
                />
                <VisualizationCanvas
                  candidate={candidate}
                  session={session}
                  canvasLevel={canvasLevel}
                  selectedBlockId={selectedBlockId}
                  lineTokenScope={lineTokenScope}
                  queryPointMode={queryPointMode}
                  onCanvasLevel={handleCanvasLevel}
                  onQueryPointModeChange={setQueryPointMode}
                  onBlock={handleHierarchyBlock}
                  onHierarchyLine={handleHierarchyLine}
                  onHierarchyConcept={selectConcept}
                  graph={graph}
                  selectedConcepts={selectedConcepts}
                  selectedLines={selectedLines}
                  selectedTokenIds={selectedTokenIds}
                  selectedManualLinkId={selectedManualLinkId}
                  manualLinks={manualLinks}
                  neighborMode={neighborMode}
                  neighborSignals={neighborSignals}
                  linkMode={linkMode}
                  linkDraft={linkDraft}
                  loading={loading || graphLoading}
                  onNode={handleGraphNode}
                  onDrop={handleGraphDrop}
                  dragMode={dragMode}
                  resetKey={resetVersion}
                  dragTokenMatches={currentDragMatches.tokenMatches}
                  selectedDragMatchKey={selectedDragMatchKey}
                  dragTargetIds={currentDragTargets}
                  onDragTargetChange={updateDragTargets}
                  dragTargetsConfirmed={currentDragTargetsConfirmed}
                  selectedExternalImpactKey={selectedExternalImpactKey}
                  onDragTargetsConfirm={confirmDragTargets}
                  persistedPositions={candidate ? dragPositionsByCandidate[candidate.id] ?? {} : {}}
                  onPositionsChange={(positions) => {
                    if (!candidate) return;
                    setDragPositionsByCandidate((current) => ({ ...current, [candidate.id]: positions }));
                  }}
                  persistedTrails={candidate ? dragTrailsByCandidate[candidate.id] ?? {} : {}}
                  onTrailsChange={(trails) => {
                    if (!candidate) return;
                    setDragTrailsByCandidate((current) => ({ ...current, [candidate.id]: trails }));
                  }}
                  externalImpacts={currentExternalImpacts}
                />
              </>
            )}
          </div>
          </>}
          </>
          )}
        </div>
        {!generationMode && !adjudicationMode ? (
          <CandidatePanel
            candidates={candidates.length ? candidates : session?.candidates ?? []}
            selectedId={activeCandidateId}
            onSelect={selectCandidate}
            modelId={modelId}
            showGroundTruth={isDemo}
            adjudicationIds={adjudicationIds}
            onAdjudicationToggle={toggleAdjudicationCandidate}
          />
        ) : null}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
