# Interactive ConceptLens

Interactive ConceptLens is a React and Python workspace for inspecting and
editing query-code representation alignments in XSearch retrieval results.

## Repository Scope

This repository contains only the core application architecture:

- `frontend/`: React/Vite interface, hierarchical embedding views, code viewer,
  candidate comparison, and drag interaction state.
- `backend/`: Python API, candidate payload construction, DynaVis graph
  assembly, and session-scoped representation intervention.

Evaluation data, model checkpoints, packed representations, projections,
research scripts, and working documents are deployment assets and are not
versioned here.

## Development

Install the frontend dependencies and build the static interface:

```bash
cd frontend
npm install
npm run build
```

The backend requires approved local XSearch/CSN data assets and environment
paths before it can be started:

```bash
python backend/run_server.py
```

The production deployment serves `frontend/dist` through Nginx and runs the
Python API on `127.0.0.1:8765`.
