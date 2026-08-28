from __future__ import annotations

import re
import time
import uuid
from typing import Any


SCENARIO_VERSION = "csn_11772-v3"
REFERENCE_SELECTION_INSTRUCTION = (
    "目标实现不包含在 Top20 候选中。请选择一段最有助于后续生成的 reference：它可以展示相关 API "
    "或属性的使用方式、extension/MIME type/compiler 之间的数据关系、已有 format-handling logic，"
    "或可迁移的控制逻辑与领域知识。Reference 不必是现成答案。"
)

TASK_BRIEFS: dict[str, dict[str, Any]] = {
    "csn_11772": {
        "caseId": "csn_11772",
        "version": SCENARIO_VERSION,
        "role": "你正在维护一个 Web 资源构建系统中的 AssetAttributes 组件。",
        "systemContext": (
            "该系统处理 JavaScript、CSS、CoffeeScript、模板等资源，并在编译、打包和发布前维护它们的格式信息。\n\n"
            "AssetAttributes 是单个资源的属性说明器：它保存资源已知的信息，也会结合当前构建环境和 compiler 的信息，"
            "推导 asset 的格式属性。一个资源可能包含多个 extension，例如 js/models.js.coffee 同时带有 .js 与 .coffee。\n\n"
            "并非每个 asset 都具备完整格式信息。这个组件的职责是利用当前可获得的对象和项目级约定，为后续构建流程提供一致的格式属性。"
        ),
        "domainObjects": [
            {"name": "Asset", "description": "需要构建或发布的资源；可能已有 extension，也可能只保留部分格式信息。"},
            {"name": "Extension / format", "description": "文件格式标识，例如 .js、.css、.coffee；可用于编译规则与输出命名。"},
            {"name": "MIME type", "description": "内容类型标识，例如 application/javascript 或 text/coffeescript。"},
            {"name": "Environment registry", "description": "environment.mimetypes 保存项目认可的 extension 与 MIME type 对应关系。"},
            {"name": "Compiler", "description": "处理资源的组件；compiler_mimetype 可能提供其输出内容的 MIME type，也可能为 None。"},
        ],
        "essentialDomainKnowledge": [
            "extension 与 MIME type 是描述同一类格式知识的两种方式：前者更接近文件名和编译规则，后者更接近内容类型与 compiler 能力。",
            "environment.mimetypes 是项目级格式 registry，而不是普通字符串集合；它提供 extension 与 MIME type 之间的约定关系。",
            "compiler 可以处理或产生特定格式，并可能通过 compiler_mimetype 暴露相应的 MIME type。",
            "当 asset 没有足够的直接格式信息时，compiler 提供的内容类型可以成为推断格式的上下文，但需要结合 environment 中的约定理解。",
            "验证一个已有 extension 是否有效，与根据 compiler 信息推断一个隐式 extension，是不同的职责。",
        ],
        "sections": [
            {
                "heading": "场景",
                "paragraphs": [
                    "你正在维护一个 Web 资源构建系统。这个系统会处理 JavaScript、CSS、CoffeeScript、模板等不同类型的资源，并在编译、打包和发布之前维护这些资源的格式信息。",
                    "你现在接手的是其中的 AssetAttributes 类。它可以理解为一个资源的“属性说明器”：它保存资源本身已有的信息，也会结合当前构建环境和 compiler 的信息，进一步推导这个 asset 的格式属性。",
                    "一个资源可能同时包含多个 extension；extension 就是常见的文件后缀，例如 .js、.css、.coffee。",
                ],
                "codeBlocks": [
                    {"language": "text", "content": "js/models.js.coffee"},
                    {"language": "python", "content": "extensions = ['.js', '.coffee']"},
                ],
            },
            {
                "heading": "这个系统中的格式信息",
                "paragraphs": [
                    "除了 extension，这个项目还会使用 MIME type 来描述内容类型。extension 是文件格式的名称；MIME type 是系统描述这种内容类型的另一种方式。",
                    "项目的 environment 中维护格式注册关系。environment.mimetypes 保存的是项目认可的 extension 和 MIME type 之间的对应关系。",
                    "有些 asset 还需要经过 compiler 处理。除了 compiler 对象本身，系统有时还能从 compiler 获得它所产生内容的 MIME 信息；在 AssetAttributes 中，这可能表现为 compiler_mimetype，也可能暂时为 None。",
                    "这些信息不一定在每个 asset 上全部存在。AssetAttributes 的职责之一，就是利用当前能够获得的信息，为后续构建流程提供尽可能完整的格式属性。",
                ],
                "codeBlocks": [
                    {"language": "text", "content": ".js      → application/javascript\n.css     → text/css\n.coffee  → text/coffeescript"},
                    {"language": "python", "content": "environment.mimetypes\ncompiler_mimetype = 'application/javascript'"},
                ],
            },
            {
                "heading": "你现在遇到的问题",
                "paragraphs": [
                    "某些 asset 在当前状态下没有足够的信息直接得到需要的 format extension，但是它所使用的 compiler 已经提供了关于输出内容格式的信息，同时当前项目的 environment 保存着自己的格式约定。",
                    "后续构建流程希望 AssetAttributes 能进一步回答：根据当前 asset 所使用的 compiler，可以推断出什么 format extension？",
                    "你刚接手这部分代码，并不完全清楚项目已有代码如何组织 extension、MIME type、compiler 和 environment format information。因此，在直接实现新功能之前，你决定先在现有代码库中寻找参考：哪些函数能够帮助你理解这些信息在项目内部如何连接？",
                ],
            },
            {
                "heading": "你的任务",
                "paragraphs": [
                    "检索系统会返回 20 个可能相关的代码片段。真正需要实现的目标函数不包含在这些候选中，因此你不是在 Top20 中寻找一个现成答案。",
                    "你需要选择一个最值得用于后续代码生成的 Reference。一个好的 Reference 不要求和目标函数完全相同。完成比较后，系统会使用当前 Query 与你选择的 Reference 来生成需要实现的函数。",
                ],
                "bullets": [
                    "展示项目中相关 API 或属性是如何使用的。",
                    "暴露 extension、MIME type、compiler 等信息之间的重要关系。",
                    "展示项目已有的 format-handling logic。",
                    "提供能够迁移到当前任务中的控制逻辑或领域知识。",
                ],
            },
            {
                "heading": "本次检索 Query",
                "codeBlocks": [{"language": "text", "content": "Implicit format extension on the asset by its compilers."}],
            },
        ],
        "taskQuery": "Implicit format extension on the asset by its compilers.",
        "referenceSelectionInstruction": REFERENCE_SELECTION_INSTRUCTION,
    }
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
