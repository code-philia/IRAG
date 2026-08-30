export type Concept = {
  id: string;
  conceptId: number;
  tokenIndices: number[];
  text: string;
  color: string;
  weight?: number;
};

export type CandidateSummary = {
  id: string;
  codeIdx: number;
  rank: number;
  corpusRank?: number;
  originalRank?: number;
  rankDelta?: number;
  similarity: number;
  originalSimilarity?: number;
  similarityDelta?: number;
  manualBoost?: number;
  dragBoost?: number;
  dragSimilarity?: number | null;
  generalizedDelta?: number;
  adapterDelta?: number;
  contrastivePushDelta?: number;
  representationOriginal?: number;
  representationGeneralized?: number;
  generalizationSource?: string;
  isGroundTruth?: boolean;
  demoPreset?: boolean;
  metadata: Record<string, string>;
};

export type SessionPayload = {
  testId: string;
  model?: { id: string; name: string; type: string; description?: string };
  capabilities?: {
    concepts?: boolean;
    hierarchy?: boolean;
    token_similarity?: boolean;
    projection?: boolean;
    inspect?: boolean;
    intervention?: boolean;
    reranking_after_intervention?: boolean;
    external_effects?: boolean;
  };
  rankingSource?: string;
  conceptSource?: string;
  query: {
    rawText: string;
    tokens: string[];
    concepts: Concept[];
    metadata: Record<string, string>;
  };
  candidates: CandidateSummary[];
  referenceSelection?: {
    enabled: boolean;
    selectionLimit: 1;
  };
  groundTruth?: {
    codeIdx: number;
    rank: number;
    score: number;
    top1Correct: boolean;
  };
  generalizationActive?: boolean;
  generalizationDemo?: {
    enabled: boolean;
    label: string;
    presetSource: string;
    groundTruthCandidateId: string;
    groundTruthDisplayRank: number;
    originalStep7000Rank: number;
    interactionCandidateId: string;
    queryTokenIndices: number[];
    codeTokenIndex: number;
    groundTruthQueryTokenIndices?: number[];
    groundTruthCodeTokenIndex?: number | null;
    instruction: string;
  };
};

export type ConceptMatch = {
  id: string;
  conceptId: number;
  queryTokenIndices: number[];
  codeTokenIndices: number[];
  queryText: string;
  codeText: string;
  similarity: number;
  color: string;
};

export type CandidateDetail = {
  id: string;
  testId: string;
  codeIdx: number;
  queryTokens?: string[];
  rawCode: string;
  codeTokens: string[];
  codeLines: Array<{
    lineNumber: number;
    text: string;
    tokenIndices: number[];
  }>;
  similarity: number;
  conceptMatches: ConceptMatch[];
  metadata: Record<string, string>;
  rankingSource?: string;
  conceptSource?: string;
  model?: SessionPayload["model"];
  capabilities?: SessionPayload["capabilities"];
  generalizedRepresentationScore?: number;
  generalizationActive?: boolean;
  lineSimilarityTransitions?: Array<{
    conceptId: number;
    lineNumber: number;
    previousLineNumber?: number;
    similarity: number;
    baseline: number;
    delta: number;
    color: string;
    source?: "local_drag" | "generalized";
  }>;
};

export type GraphNode = {
  id: string;
  type: "query_token" | "code_token";
  label: string;
  conceptId: number | null;
  conceptIds: number[];
  color: string;
  colors?: string[];
  highlightScore?: number;
  tokenIndex: number;
  lineNumber?: number;
  x: number;
  y: number;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  type: string;
  conceptId: number;
  similarity: number;
  color: string;
};

export type SemanticLink = {
  source: string;
  target: string;
  similarity: number;
  sourceType: "query_token" | "code_token";
  targetType: "query_token" | "code_token";
};

export type ManualLink = {
  id: string;
  testId: string;
  candidateId: string;
  sourceCandidateId?: string;
  sourceCodeTokenIndex?: number;
  queryTokenIndex: number;
  queryToken: string;
  codeTokenIndex: number;
  codeToken: string;
  epoch: number;
  distance: number;
  similarity: number;
  modelCosine?: number;
  similaritySource?: string;
  color: string;
  inferred?: boolean;
};

export type ManualLinkResponse = {
  status: "ok";
  link: ManualLink | null;
  links: ManualLink[];
  linksByCandidate?: Record<string, ManualLink[]>;
  candidates: CandidateSummary[];
  diagnostic: Record<string, string | number>;
};

export type VisualizationGraph = {
  testId: string;
  candidateId: string;
  method: "DynaVis";
  epoch: number;
  availableEpochs: number[];
  representationSource: "packed_cache" | "fallback" | string;
  representationKind: string;
  fallbackReason?: string | null;
  hiddenDim?: number | null;
  codeTokenSlotOffset?: number | null;
  queryRepresentationSource?: string | null;
  contentPath: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  semanticLinks?: SemanticLink[];
  semanticLinkSource?: string;
  semanticLinkThreshold?: number;
  hierarchy?: {
    queryNodes: Array<{ id: string; type: string; label: string; conceptId: number | null; similarity: number; x: number; y: number; labelPlacement?: "left" | "right" }>;
    queryNodesByBlock?: Record<string, Array<{ id: string; type: string; label: string; conceptId: number | null; similarity: number; x: number; y: number; labelPlacement?: "left" | "right" }>>;
    recommendedTokenIndices?: number[];
    recommendedTokens?: Array<{ tokenIndex: number; conceptId: number; sourceTokenIndex: number; relation: "direct" | "token_family" }>;
    blocks: Array<{ id: string; kind: string; label: string; startLine: number; endLine: number; lineNumbers: number[]; tokenIndices: number[]; similarity: number; conceptId: number | null; conceptScores: Array<{ conceptId: number; similarity: number }>; displayConceptIds?: number[]; signals?: Array<{ kind: "latent_token" | "weak_block" | "uncovered_line"; conceptId?: number; tokenIndex?: number; similarity?: number; aggregateSimilarity?: number; source?: string }>; x: number; y: number }>;
    linesByBlock: Record<string, Array<{ lineNumber: number; text: string; tokenIndices: number[]; similarity: number; conceptId: number | null; conceptScores: Array<{ conceptId: number; similarity: number }>; displayConceptIds?: number[]; signals?: Array<{ kind: "latent_token" | "weak_block" | "uncovered_line"; conceptId?: number; tokenIndex?: number; similarity?: number; aggregateSimilarity?: number; source?: string }>; x: number; y: number }>>;
  };
};

export type TrainingEvidenceSample = {
  trainIndex: number;
  url: string;
  path: string;
  funcName: string;
  conceptId: string;
  conceptText: string;
  stepName: string;
  stepDesc: string;
  stepCode: string;
  docstring: string;
  code: string;
  querySimilarity: number;
  codeSimilarity: number;
  score: number;
  reason: string;
};

export type TokenPairAttribution = {
  status: "ok";
  source: string;
  selectedPair: {
    queryTokenIndex: number;
    codeTokenIndex: number;
    queryToken: string;
    codeToken: string;
    modelCosine: number;
    modelPositiveCosine: number;
    projectionDistance: number | null;
    queryHighlightScore: number | null;
    codeHighlightScore: number | null;
    queryHighlighted: boolean;
    codeHighlighted: boolean;
    queryConceptIds: number[];
    lineNumber: number | null;
  };
  candidateContext: {
    candidateId: string;
    codeIdx: number;
    rank: number | null;
    similarity: number | null;
    lineSimilarity: number | null;
    metadata: Record<string, string>;
    matchingConcepts: Array<{
      conceptId: number;
      queryText: string;
      codeText: string;
      lineNumber: number;
      similarity: number;
      containsBothSelectedTokens: boolean;
    }>;
  };
  trainingEvidence: {
    source: string;
    cachePath: string;
    sampleLimit: number;
    processedSamples: number;
    recordCount: number;
    method: string;
    supportingSamples: TrainingEvidenceSample[];
    conflictingSamples: TrainingEvidenceSample[];
  };
  diagnosis: {
    summary: string;
    evidenceCaveat: string;
  };
};

export type GradientInfluenceSample = {
  trainIndex: number;
  batchIndex: number;
  batchStart: number;
  batchEnd: number;
  influence: number;
  gradientDot: number;
  normalizedInfluence: number;
  trainGradientNorm: number;
  interpretation: "helpful" | "harmful";
  loss: number;
  nlHighlightLoss: number;
  codeHighlightLoss: number;
  url: string;
  path: string;
  funcName: string;
  docstring: string;
  code: string;
  validCommentSpans: Array<[string, number[]]>;
  validCodeSpans: Array<[string, number[]]>;
};

export type GradientInfluenceBatch = {
  batchIndex: number;
  batchStart: number;
  batchEnd: number;
  trainIndices: number[];
  influence: number;
  gradientDot: number;
  normalizedInfluence: number;
  trainGradientNorm: number;
  interpretation: "helpful" | "harmful";
  loss: number;
  highlightLoss: number;
  nlHighlightLoss: number;
  codeHighlightLoss: number;
  crossSampleLoss: number;
  crossSampleWeight: number;
  cacheHit?: boolean;
  cachePath?: string;
  selection?: {
    selectionScore: number;
    maxEvidenceScore: number;
    top3EvidenceScoreSum: number;
    hitCount: number;
    supportCount: number;
    conflictCount: number;
    exactTokenPairHits: number;
    sampleHits: Array<{
      trainIndex: number;
      source: string;
      score: number;
      querySimilarity: number;
      codeSimilarity: number;
      funcName: string;
      path: string;
      docstring: string;
      conceptText: string;
      stepDesc: string;
      stepCode: string;
    }>;
  };
  sampleSummaries: Array<{
    trainIndex: number;
    funcName: string;
    path: string;
    url: string;
    docstring: string;
  }>;
};

export type GradientAttribution = {
  status: "ok" | "cache_missing" | "no_candidates";
  source: string;
  lossScope?: "highlight_only" | "highlight_plus_cross_sample_batch";
  message?: string;
  queryLoss: null | {
    type: string;
    formula: string;
    value: number;
    modelCosine: number;
    gradientNorm: number;
  };
  scope?: {
    parameterScope: string[];
    lossScope: string;
    batchReconstruction: string;
    candidateSource: string;
    evaluatedTrainSamples: number;
    evaluatedBatches?: number;
    requestedBatches?: number;
    candidateTrainSamples?: number;
    candidateEvidenceLimit?: number;
    cacheHits?: number;
    cacheMisses?: number;
    crossSampleWeight?: number;
    batchSelector?: string;
    influenceMethod?: string;
  };
  helpfulSamples: GradientInfluenceSample[];
  harmfulSamples: GradientInfluenceSample[];
  helpfulBatches?: GradientInfluenceBatch[];
  harmfulBatches?: GradientInfluenceBatch[];
};

export type GenerationTask = {
  caseId: string;
  query: string;
  language: string;
  evaluationAvailable: boolean;
  functionSignature?: string;
  generationInstruction?: string;
};

export type RetrievalSelection = {
  id: string;
  caseId: string;
  selectedCandidateId: string;
  selectedRank: number | null;
  selectedScore: number | null;
  interactionUsed: boolean;
  candidate: {
    id: string;
    codeIdx: number;
    rawCode: string;
    metadata: Record<string, string>;
  };
};

export type GenerationConfirmation = {
  status: "ok";
  selectionId: string;
  task: GenerationTask;
  selection: RetrievalSelection;
};

export type TaskBrief = {
  caseId: string;
  version: string;
  role: string;
  systemContext: string;
  domainObjects: Array<{ name: string; description: string }>;
  essentialDomainKnowledge: string[];
  taskQuery: string;
  referenceSelectionInstruction: string;
  sections?: Array<{
    heading: string;
    content?: Array<{ type: "paragraph"; text: string } | { type: "code"; language?: string; content: string }>;
    paragraphs?: string[];
    codeBlocks?: Array<{ language?: string; content: string }>;
    bullets?: string[];
  }>;
};

export type StudySession = {
  sessionId: string;
  participantId: string;
  condition: "baseline" | "irag";
  experimentVersion: string;
  createdAt: number;
};

export type ReferenceHint = {
  status: "ok";
  selectionId: string;
  whatItDoes: string;
  usefulClue: string;
};

export type GenerationResult = {
  status: "ok";
  generationId: string;
  caseId: string;
  condition: "no_rag" | "automatic_rag" | "interactive_rag";
  contextCandidateId: string | null;
  generatedCode: string;
  model: string;
  promptVersion: string;
  generationTime: number;
  curatedGeneration?: boolean;
  curatedCondition?: string;
};

export type GenerationEvaluation = {
  status: "ok" | "evaluation_unavailable";
  generationId: string;
  testsPassed?: number;
  testsTotal?: number;
  passRate?: number;
  fullSuccess?: boolean;
  apiPrecision?: number;
  apiRecall?: number;
  apiF1?: number;
  matchedApis?: string[];
  generatedApis?: string[];
  groundTruthApis?: string[];
  testResults?: Array<{
    name: string;
    passed: boolean;
    error?: string;
  }>;
  message?: string;
};

export type GenerationComparison = {
  generationId: string;
  caseId: string;
  generatedCode: string;
  groundTruth: {
    candidateId: string;
    functionName: string;
    path: string;
    rawCode: string;
  };
  evaluationAvailable: boolean;
};
