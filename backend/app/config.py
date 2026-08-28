from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_local_env() -> None:
    """Load local development secrets without adding a dotenv dependency."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()
ASSETS_DIR = Path(os.environ.get("XSEARCH_ASSETS_DIR", ROOT_DIR / "assets"))
XSEARCH_ROOT = Path(os.environ.get("XSEARCH_ROOT", ASSETS_DIR / "XSearch"))
_configured_user_study_dir = Path(os.environ.get("XSEARCH_USER_STUDY_DATA_DIR", ASSETS_DIR / "user-study-data"))
_local_user_study_dir = ROOT_DIR.parent / "XSearch-user-study" / "data"
USER_STUDY_DATA_DIR = _configured_user_study_dir if (_configured_user_study_dir / "python_filtered_cleaned_eval_processed.jsonl").exists() else _local_user_study_dir
COCOSODA_PATH = Path(os.environ.get("XSEARCH_COCOSODA_PATH", ASSETS_DIR / "CoCoSoDa"))
TTV_TOOL_PATH = Path(os.environ.get("XSEARCH_TTV_TOOL_PATH", ASSETS_DIR / "time-travelling-visualizer" / "tool"))

DEFAULT_EXPERIMENT_ID = "xsearch_user_study_python"
DEFAULT_DATASET_PATH = USER_STUDY_DATA_DIR / "python_filtered_cleaned_eval_processed.jsonl"
DEFAULT_MATCH_PATH = XSEARCH_ROOT / "checkpoints/python/analysis/topk_matches_xsearch.json"
DEFAULT_CHECKPOINT_PATH = XSEARCH_ROOT / "training_lab/checkpoints_lab/repro_python_coco_original_loss_1ep_csn/Epoch_1/subject_model_python.pth"
LATEST_STEP_CHECKPOINT_PATH = XSEARCH_ROOT / "training_lab/checkpoints_lab/repro_python_coco_original_loss_1ep_csn/Step_7000/subject_model_python.pth"
CODEBERT_BASE_PATH = Path(os.environ.get("CODEBERT_BASE_PATH", XSEARCH_ROOT.parent.parent / "codebert-base"))
CODEBERT_CHECKPOINT_PATH = Path(os.environ.get(
    "CODEBERT_CHECKPOINT_PATH",
    XSEARCH_ROOT / "checkpoints/python_codebert/Epoch_2/subject_model_python.pth",
))
CODEBERT_INDEX_PATH = Path(os.environ.get("CODEBERT_INDEX_PATH", ROOT_DIR / "data" / "codebert_generic_index.npz"))
LOCAL_COCOSODA_PATH = COCOSODA_PATH
TRAINING_EVAL_RESULTS_DIR = XSEARCH_ROOT / "training_lab/eval_results/repro_python_coco_original_loss_1ep_csn"
SMOKE_CODEBASE_PATH = XSEARCH_ROOT / "training_lab/eval_results/python_coco_smoke/codebase_first_200.jsonl"

RUNS_DIR = ROOT_DIR / "data" / "dynavis_runs"
LOG_DIR = ROOT_DIR / "data" / "logs"
STUDY_DIR = ROOT_DIR / "data" / "study"
STUDY_SESSION_DIR = STUDY_DIR / "sessions"
GENERATION_DIR = ROOT_DIR / "data" / "generation"
GENERATION_RECORDS_DIR = GENERATION_DIR / "records"
GENERATION_RESULTS_DIR = GENERATION_DIR / "results"
GENERATION_PROVIDER = os.environ.get("GENERATION_PROVIDER", "openai_compatible")
GENERATION_API_URL = os.environ.get("GENERATION_API_URL", "")
GENERATION_API_KEY = os.environ.get("GENERATION_API_KEY", "")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "")
GENERATION_TEMPERATURE = float(os.environ.get("GENERATION_TEMPERATURE", "0"))
GENERATION_MAX_TOKENS = int(os.environ.get("GENERATION_MAX_TOKENS", "1024"))
GENERATION_EXECUTION_ENABLED = os.environ.get("GENERATION_EXECUTION_ENABLED", "0").lower() in {"1", "true", "yes"}
ALIGNED_XSEARCH_DIR = ROOT_DIR / "data" / "aligned_xsearch"
ATTRIBUTION_DIR = ROOT_DIR / "data" / "attribution"
TRAINING_EVIDENCE_CACHE_PATH = ATTRIBUTION_DIR / "training_evidence_cache_full_v2.json"
TRAINING_EVIDENCE_INDEX_PATH = ATTRIBUTION_DIR / "training_evidence_index_full_v2.json"
GRADIENT_BATCH_CACHE_DIR = ATTRIBUTION_DIR / "gradient_batch_cache"
TRAIN_DATA_FILE = XSEARCH_ROOT / "preprocess_dataset/csn_data/python.jsonl"
CSN_PYTHON_TEST_PATH = XSEARCH_ROOT / "preprocess_dataset/csn_data/python_test.jsonl"
CSN_PYTHON_CODEBASE_PATH = XSEARCH_ROOT / "preprocess_dataset/csn_data/python_codebase.jsonl"
USER_STUDY_STEP7000_RANKING_PATH = ALIGNED_XSEARCH_DIR / "user_study_step7000_rankings.json"
FULL_EVAL_STEP7000_RANKING_PATH = ALIGNED_XSEARCH_DIR / "full_eval_step7000_rankings.json"
CSN_URL_MAPPED_API_DEMO_RANKINGS_PATH = ALIGNED_XSEARCH_DIR / "url_mapped_api_demo_rankings.json"
SINGLE_REFERENCE_DEMO_RANKINGS_PATH = ALIGNED_XSEARCH_DIR / "single_reference_demo_rankings.json"
API_BRIDGE_STEP7000_PACKED_PATH = ALIGNED_XSEARCH_DIR / "api_bridge_step7000_packed.pt"
CSN_API_BRIDGE_PREFIX_CACHE_PATH = ALIGNED_XSEARCH_DIR / "api_bridge_gt_prefix_cache.pt"
USER_STUDY_STEP7000_CODE_CACHE_PATH = ALIGNED_XSEARCH_DIR / "user_study_step7000_code_topk_cache.pt"
USER_STUDY_STEP7000_BLOCK_CACHE_PATH = ALIGNED_XSEARCH_DIR / "user_study_step7000_block_cache.pt"
CSN_FULL_STEP7000_CODE_CACHE_PATH = Path(
    os.environ.get(
        "XSEARCH_CSN_FULL_STEP7000_CODE_CACHE_PATH",
        TRAINING_EVAL_RESULTS_DIR / "python_full_step7000_fullcodebase_packed.pt",
    )
)
CSN_GT_PREFIX_CACHE_PATH = Path(
    os.environ.get(
        "XSEARCH_CSN_GT_PREFIX_CACHE_PATH",
        TRAINING_EVAL_RESULTS_DIR / "csn_gt_prefix_step7000_cache.pt",
    )
)
USER_STUDY_ROLES_PATH = XSEARCH_ROOT / "preprocess_dataset/role_tensor_python_40k_eval.npy"

CONCEPT_COLORS = [
    "#FFB6A1",
    "#A1FFB6",
    "#A1B6FF",
    "#FFA1D6",
    "#D6A1FF",
    "#A1FFF5",
    "#EEFF6B",
    "#FFC6A1",
    "#A1FFD6",
    "#C172FD",
]
