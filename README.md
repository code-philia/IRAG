# Interactive RAG

Interactive RAG is a React and Python workspace for inspecting, comparing, and
editing query-code representation alignments before selecting code evidence for
retrieval-augmented generation.

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
python -m pip install astroid
python backend/run_server.py
```

The `astroid` dependency is required by the `csn_3846` generation test suite.

The production deployment serves `frontend/dist` through Nginx and runs the
Python API on `127.0.0.1:8765`.

## Access Modes

The same deployment exposes three URL-based modes:

| Path | Purpose | Model and controls |
| --- | --- | --- |
| `/` | Interactive RAG demo | XSearch / CodeBERT selector and the full workspace. |
| `/study` | User-study workspace | XSearch only, with the full Interactive RAG workflow. |
| `/baseline` | Baseline workspace | XSearch only; plain Query and Code Viewer, Candidate List, and reference selection. |

`/study` and `/baseline` record their mode in interaction logs and reference
selection records so later evaluation can separate them from demo sessions.
