from __future__ import annotations

import re
import time
import uuid
from typing import Any


SCENARIO_VERSION = "csn_11772-v26"
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
        "version": "csn_584-v3",
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
    "csn_14238": {
        "caseId": "csn_14238",
        "version": "csn_14238-v1",
        "role": "你正在维护 Jupyter Qt Console 中处理 call-tip 回复的逻辑。",
        "systemContext": "kernel 的回复可能在用户继续输入、移动光标或发起后续请求之后才到达。",
        "domainObjects": [],
        "essentialDomainKnowledge": [],
        "sections": [
            {
                "heading": "Task",
                "paragraphs": ["处理 call-tip 的异步回复，并仅在该回复仍与当前请求和光标上下文相符时显示提示。"],
            },
            {
                "heading": "Available context",
                "bullets": [
                    "当前 call-tip 请求会记录请求标识和光标位置。",
                    "回复携带其父请求的标识；当前编辑器也可以提供光标位置。",
                ],
            },
            {
                "heading": "Expected behavior",
                "paragraphs": ["迟到或已经过期的回复不应更新当前 call-tip；仍然有效的回复应继续进入现有的提示显示流程。"],
            },
        ],
        "taskQuery": "Handle replies for call tips.",
        "referenceSelectionInstruction": "系统将提供 20 个候选代码片段，目标实现不在候选中。请选择 ONE Reference，作为后续代码生成的参考，而不是寻找现成答案。",
    },
    "csn_8884": {
        "caseId": "csn_8884",
        "version": "csn_8884-v5",
        "role": "你正在维护一个 Python AST optimizer，需要处理 try statement 对应的 AST node。",
        "systemContext": "当前任务是在保持 try / except 控制流结构和源码位置信息有效的前提下清理不可到达的代码；当 return、raise 等语句已经终止某个执行区域的控制流时，后续 statements 可能成为不可到达的 dead code。",
        "domainObjects": [],
        "essentialDomainKnowledge": [],
        "sections": [],
        "taskQuery": "Eliminate dead code from except try bodies.",
        "referenceSelectionInstruction": "系统将提供 20 个候选代码片段，目标实现不在候选中。请选择 ONE Reference，作为后续代码生成的参考。一个有价值的 Reference 应帮助你理解：项目如何遍历并重写包含多个 statement list 的 AST 节点；如何处理 control-flow node 中需要清理的 statement regions；AST 变换后如何保持节点结构和 source-location information 有效。",
    },
    "csn_3846": {
        "caseId": "csn_3846",
        "version": "csn_3846-v4",
        "role": "你正在维护 Pylint 中与方法 decorator 相关的判定逻辑。",
        "systemContext": "当前需要判断一个 dotted decorator 所引用的名称是否与当前方法名相同。Dotted decorator 是使用点号连接的 decorator 表达式，例如 @x.setter；其中 x 是点号左侧被引用的名称，setter 是在该名称上访问的 decorator 属性。",
        "domainObjects": [],
        "essentialDomainKnowledge": [],
        "sections": [],
        "taskQuery": "Return True if the object is a method redefined via decorator.",
        "referenceSelectionInstruction": "系统将提供 20 个候选代码片段，目标实现不在候选中。请选择 ONE Reference，作为后续代码生成的参考。一个有价值的 Reference 应帮助你理解：函数 decorator 如何被访问和逐项检查；dotted decorator 如何表示其所引用的名称；以及如何从 AST 信息中完成与当前方法名相关的判定。",
    },
    "csn_12226": {
        "caseId": "csn_12226",
        "version": "csn_12226-v1",
        "role": "你正在维护一个用于删除 alarm 的 CLI command。",
        "systemContext": "上游已经完成 API 请求；当前方法只根据 HTTP response 决定是否向终端输出内容。",
        "domainObjects": [],
        "essentialDomainKnowledge": [],
        "sections": [],
        "taskQuery": "Handle the results of the API call.",
        "referenceSelectionInstruction": "系统将提供 20 个候选代码片段，目标实现不在候选中。请选择 ONE Reference，作为后续代码生成的参考。一个有价值的 Reference 应帮助你理解：CLI 如何读取 API response 的状态和 body；成功与失败结果如何走向不同的终端输出行为；以及 JSON response 如何被处理为终端展示内容。",
    },
    "csn_42": {
        "caseId": "csn_42",
        "version": "csn_42-v2",
        "role": "你正在维护一个删除 Cloud SQL database 的操作。",
        "systemContext": "Cloud SQL 是云端托管的数据库服务。删除请求先返回本次异步任务对应的 Operation；函数必须等待该 Operation 成功完成后才能返回 None。如果 Operation 失败，应报告相应错误。",
        "domainObjects": [],
        "essentialDomainKnowledge": [],
        "sections": [],
        "taskQuery": "Deletes a database from a Cloud SQL instance.",
        "referenceSelectionInstruction": "系统将提供 20 个候选代码片段，目标实现不在候选中。请选择 ONE Reference，作为后续代码生成的参考。一个有价值的 Reference 应帮助你理解：类似云端数据库删除请求如何实际发送；请求返回后如何确认后台操作已经真正完成；以及如何区分请求已被接受与数据库已经删除。",
    },
    "csn_9388": {
        "caseId": "csn_9388",
        "version": "csn_9388-v2",
        "role": "你正在维护一个处理 URL 的小工具。",
        "systemContext": "URL 可以包含 scheme、host、path、query 和 fragment。当前任务只移除 query（查询参数），其他部分保持不变。",
        "domainObjects": [],
        "essentialDomainKnowledge": [],
        "sections": [],
        "taskQuery": "Return a URL with the query component removed.",
        "referenceSelectionInstruction": "系统将提供 20 个候选代码片段，目标实现不在候选中。请选择 ONE Reference，作为后续代码生成的参考。一个有价值的 Reference 应帮助你理解：代码如何读取和重新组合 URL 的不同部分；如何只改变 query component 而保留地址、路径和 fragment；以及同一 repository 中已有的 URL 处理约定。",
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
