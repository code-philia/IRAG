from __future__ import annotations

import re
import time
import uuid
from typing import Any


SCENARIO_VERSION = "csn_11772-v8"
REFERENCE_SELECTION_INSTRUCTION = (
    "目标实现不包含在 Top20 候选中。请选择一段最有助于后续生成的 reference：它可以展示相关 API "
    "或属性的使用方式、extension/MIME type/compiler 之间的数据关系、已有 format-handling logic，"
    "或可迁移的控制逻辑与领域知识。Reference 不必是现成答案。"
)

TASK_BRIEFS: dict[str, dict[str, Any]] = {
    "csn_11772": {
        "caseId": "csn_11772",
        "version": SCENARIO_VERSION,
        "role": "你正在接手 Gears 资源构建系统中的一项相关维护工作。",
        "systemContext": "围绕同一个 LESS 资源，理解 source format、compiler output 与项目格式信息如何共存。",
        "domainObjects": [],
        "essentialDomainKnowledge": [],
        "sections": [],
        "taskQuery": "Implicit format extension on the asset by its compilers.",
        "referenceSelectionInstruction": REFERENCE_SELECTION_INSTRUCTION,
    },
    "csn_584": {
        "caseId": "csn_584",
        "version": "csn_584-v2",
        "role": "你正在维护图像变换库中随机透视变换的参数生成逻辑。",
        "systemContext": "同一变换既需要描述原图与变换后图的四个角点，也需要将这些角点交给底层透视变换实现。",
        "domainObjects": [
            {"name": "startpoints", "description": "原始图像四个角点的坐标列表。"},
            {"name": "endpoints", "description": "变换后四个角点的坐标列表。"},
            {"name": "distortion_scale", "description": "限制随机角点偏移范围的强度参数。"},
        ],
        "essentialDomainKnowledge": [
            "透视变换由原始角点与目标角点之间的对应关系定义。",
            "随机参数生成需要保持角点顺序，并让生成坐标落在图像的有效范围内。",
        ],
        "sections": [],
        "taskQuery": "Get parameters for ``perspective`` for a random perspective transform.",
        "referenceSelectionInstruction": "目标实现不包含在 Top20 候选中。请选择最能帮助后续实现随机透视变换参数生成的 reference；它应提供可迁移的角点、坐标或变换知识，而不是现成答案。",
    },
}


def get_task_brief(case_id: str) -> dict[str, Any]:
    brief = TASK_BRIEFS.get(str(case_id))
    if not brief:
        raise KeyError(f"No task brief is configured for {case_id}.")
    return dict(brief)


def create_study_session(payload: dict[str, Any]) -> dict[str, Any]:
    participant_id = re.sub(r"\s+", " ", str(payload.get("participantId") or "").strip())
    if not participant_id or len(participant_id) > 64:
        raise ValueError("participantId must contain 1-64 characters.")
    condition = str(payload.get("condition") or "")
    if condition not in {"baseline", "irag"}:
        raise ValueError("condition must be baseline or irag.")
    return {
        "sessionId": f"session_{uuid.uuid4().hex}",
        "participantId": participant_id,
        "condition": condition,
        "experimentVersion": "irag-study-v1",
        "createdAt": time.time(),
    }
