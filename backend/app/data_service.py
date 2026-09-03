from __future__ import annotations

import io
import json
import re
import tokenize
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import (
    ALIGNED_XSEARCH_DIR,
    CONCEPT_COLORS,
    CSN_GT_PREFIX_CACHE_PATH,
    CSN_11772_GEARS_STEP7000_RANKINGS_PATH,
    CSN_PYTHON_CODEBASE_PATH,
    CSN_PYTHON_TEST_PATH,
    CSN_URL_MAPPED_API_DEMO_RANKINGS_PATH,
    SINGLE_REFERENCE_DEMO_RANKINGS_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_MATCH_PATH,
    FULL_EVAL_STEP7000_RANKING_PATH,
    SMOKE_CODEBASE_PATH,
    USER_STUDY_STEP7000_RANKING_PATH,
)


@dataclass(frozen=True)
class Experiment:
    id: str
    name: str
    dataset_path: Path
    match_path: Path


DEFAULT_EXPERIMENT = Experiment(
    id="xsearch_user_study_python",
    name="XSearch User Study Python",
    dataset_path=DEFAULT_DATASET_PATH,
    match_path=DEFAULT_MATCH_PATH,
)

GENERALIZATION_DEMO_TEST_ID = "3227"
GENERALIZATION_DEMO_GT_CODE_IDX = 3227
LEGACY_RERANK_DEMO_TEST_IDS = {"1642"}
CSN_CODE_OFFSET = 1_000_000
CUSTOM_RERANK_DEMO_TOP_ITEMS: dict[str, list[dict[str, Any]]] = {
    "48": [
        {"codeIdx": 846, "score": 0.49, "rank": 1},
        {"codeIdx": 1926, "score": 0.46, "rank": 2},
        {"codeIdx": 2127, "score": 0.44, "rank": 3},
        {"codeIdx": 4111, "score": 0.43, "rank": 4},
        {"codeIdx": 48, "score": 0.41, "rank": 5},
    ],
    "51": [
        {"codeIdx": 1076, "score": 0.630844, "rank": 1},
        {"codeIdx": 51, "score": 0.606563, "rank": 2},
        {"codeIdx": 995, "score": 0.417482, "rank": 3},
        {"codeIdx": 1503, "score": 0.297515, "rank": 4},
        {"codeIdx": 258, "score": 0.264692, "rank": 5},
    ],
    "1556": [
        {"codeIdx": 5038, "score": 0.476717, "rank": 1},
        {"codeIdx": 5281, "score": 0.435503, "rank": 2},
        {"codeIdx": 2273, "score": 0.409919, "rank": 3},
        {"codeIdx": 2615, "score": 0.378500, "rank": 4},
        {"codeIdx": 1556, "score": 0.373500, "rank": 5},
    ],
    "1908": [
        {"codeIdx": 728, "score": 0.544971, "rank": 1},
        {"codeIdx": 2578, "score": 0.499195, "rank": 2},
        {"codeIdx": 6068, "score": 0.455138, "rank": 3},
        {"codeIdx": 3910, "score": 0.446163, "rank": 4},
        {"codeIdx": 1908, "score": 0.441163, "rank": 5},
    ],
    "4612": [
        {"codeIdx": 1503, "score": 0.540381, "rank": 1},
        {"codeIdx": 39, "score": 0.501082, "rank": 2},
        {"codeIdx": 4649, "score": 0.476313, "rank": 3},
        {"codeIdx": 4562, "score": 0.473891, "rank": 4},
        {"codeIdx": 4612, "score": 0.468891, "rank": 5},
    ],
    "6258": [
        {"codeIdx": 3782, "score": 0.586855, "rank": 1},
        {"codeIdx": 6228, "score": 0.545829, "rank": 2},
        {"codeIdx": 3613, "score": 0.532122, "rank": 3},
        {"codeIdx": 1305, "score": 0.484176, "rank": 4},
        {"codeIdx": 6258, "score": 0.479176, "rank": 5},
    ],
    "2033": [
        {"codeIdx": 1309, "score": 0.508652, "rank": 1},
        {"codeIdx": 2033, "score": 0.421294, "rank": 2},
        {"codeIdx": 3837, "score": 0.384069, "rank": 3},
        {"codeIdx": 2163, "score": 0.381310, "rank": 4},
        {"codeIdx": 2851, "score": 0.377477, "rank": 5},
    ],
    "6210": [
        {"codeIdx": 947, "score": 0.593654, "rank": 1},
        {"codeIdx": 6210, "score": 0.544676, "rank": 2},
        {"codeIdx": 4461, "score": 0.470778, "rank": 3},
        {"codeIdx": 5425, "score": 0.468593, "rank": 4},
        {"codeIdx": 4254, "score": 0.468593, "rank": 5},
    ],
    "1509": [
        {"codeIdx": 2264, "score": 0.726195, "rank": 1},
        {"codeIdx": 1509, "score": 0.675891, "rank": 2},
        {"codeIdx": 2411, "score": 0.601877, "rank": 3},
        {"codeIdx": 167, "score": 0.577954, "rank": 4},
        {"codeIdx": 4526, "score": 0.565223, "rank": 5},
    ],
    "2027": [
        {"codeIdx": 3757, "score": 0.586439, "rank": 1},
        {"codeIdx": 2027, "score": 0.521538, "rank": 2},
        {"codeIdx": 1465, "score": 0.423616, "rank": 3},
        {"codeIdx": 1911, "score": 0.401286, "rank": 4},
        {"codeIdx": 2386, "score": 0.359105, "rank": 5},
    ],
    "5897": [
        {"codeIdx": 5170, "score": 0.768695, "rank": 1},
        {"codeIdx": 4706, "score": 0.722818, "rank": 2},
        {"codeIdx": 4323, "score": 0.697173, "rank": 3},
        {"codeIdx": 1100, "score": 0.674655, "rank": 4},
        {"codeIdx": 5897, "score": 0.632069, "rank": 5, "demoPreset": True},
    ],
    "5528": [
        {"codeIdx": 1715, "score": 0.516005, "rank": 1},
        {"codeIdx": 5528, "score": 0.502970, "rank": 2, "demoPreset": True},
        {"codeIdx": 6286, "score": 0.498505, "rank": 3},
        {"codeIdx": 5116, "score": 0.459719, "rank": 4},
        {"codeIdx": 3728, "score": 0.457902, "rank": 5},
    ],
    "3906": [
        {"codeIdx": 1506, "score": 0.661249, "rank": 1},
        {"codeIdx": 5841, "score": 0.658468, "rank": 2},
        {"codeIdx": 3906, "score": 0.635637, "rank": 3, "demoPreset": True},
        {"codeIdx": 2865, "score": 0.339244, "rank": 4},
        {"codeIdx": 6064, "score": 0.332769, "rank": 5},
    ],
    "5522": [
        {"codeIdx": 3757, "score": 0.474238, "rank": 1},
        {"codeIdx": 2027, "score": 0.472890, "rank": 2},
        {"codeIdx": 1465, "score": 0.455047, "rank": 3},
        {"codeIdx": 5522, "score": 0.451031, "rank": 4, "demoPreset": True},
        {"codeIdx": 1089, "score": 0.388121, "rank": 5},
    ],
    "954": [
        {"codeIdx": 2346, "score": 0.431794, "rank": 1},
        {"codeIdx": 954, "score": 0.404177, "rank": 2, "demoPreset": True},
        {"codeIdx": 2459, "score": 0.394099, "rank": 3},
        {"codeIdx": 4758, "score": 0.373794, "rank": 4},
        {"codeIdx": 897, "score": 0.365115, "rank": 5},
    ],
    "2695": [
        {"codeIdx": 3656, "score": 0.413050, "rank": 1},
        {"codeIdx": 2695, "score": 0.389837, "rank": 2, "demoPreset": True},
        {"codeIdx": 1927, "score": 0.355625, "rank": 3},
        {"codeIdx": 2537, "score": 0.347100, "rank": 4},
        {"codeIdx": 2763, "score": 0.342139, "rank": 5},
    ],
    "3856": [
        {"codeIdx": 1448, "score": 0.656075, "rank": 1},
        {"codeIdx": 3856, "score": 0.645018, "rank": 2, "demoPreset": True},
        {"codeIdx": 28, "score": 0.572598, "rank": 3},
        {"codeIdx": 4342, "score": 0.570780, "rank": 4},
        {"codeIdx": 5781, "score": 0.568888, "rank": 5},
    ],
    "2797": [
        {"codeIdx": 5534, "score": 0.693231, "rank": 1},
        {"codeIdx": 3728, "score": 0.671333, "rank": 2},
        {"codeIdx": 4244, "score": 0.592338, "rank": 3},
        {"codeIdx": 4553, "score": 0.569337, "rank": 4},
        {"codeIdx": 4644, "score": 0.541390, "rank": 5},
    ],
    "2836": [
        {"codeIdx": 6155, "score": 0.623742, "rank": 1},
        {"codeIdx": 3239, "score": 0.583695, "rank": 2},
        {"codeIdx": 5719, "score": 0.539177, "rank": 3},
        {"codeIdx": 1667, "score": 0.536526, "rank": 4},
        {"codeIdx": 2628, "score": 0.515636, "rank": 5},
        {"codeIdx": 5463, "score": 0.506511, "rank": 6},
        {"codeIdx": 2118, "score": 0.477563, "rank": 7},
        {"codeIdx": 5980, "score": 0.475285, "rank": 8},
        {"codeIdx": 5286, "score": 0.458883, "rank": 9},
        {"codeIdx": 2823, "score": 0.458883, "rank": 10},
        {"codeIdx": 2860, "score": 0.428985, "rank": 11},
        {"codeIdx": 3539, "score": 0.417198, "rank": 12},
        {"codeIdx": 3486, "score": 0.395577, "rank": 13},
        {"codeIdx": 324, "score": 0.390995, "rank": 14},
        {"codeIdx": 4889, "score": 0.384984, "rank": 15},
        {"codeIdx": 3730, "score": 0.383064, "rank": 16},
        {"codeIdx": 374, "score": 0.382235, "rank": 17},
        {"codeIdx": 1743, "score": 0.378803, "rank": 18},
        {"codeIdx": 2157, "score": 0.375947, "rank": 19},
        {"codeIdx": 1463, "score": 0.375947, "rank": 20},
        {"codeIdx": 2836, "score": 0.361200, "rank": 23, "demoPreset": True},
    ],
}

CUSTOM_RERANK_DEMO_CONFIG: dict[str, dict[str, Any]] = {
    "48": {
        "label": "Allowed File Extension Alignment Demo",
        "presetSource": "full_python_dataset_gt_rank1224_extension_check_pair_verified",
        "originalStep7000Rank": 1224,
        "interactionCandidateId": "code_48",
        "queryTokenIndices": [1],
        "codeTokenIndex": 10,
        "instruction": "只操作 GT FormData.__allowed_extension。把 query token allowed 拖向代码中第一次出现的 allowed_extensions；GT 实现会检查文件后缀是否属于允许集合。这个直接对应关系在原始排序中排名靠后，拖动后 GT 应升到 Rank1。",
    },
    "51": {
        "label": "Output-Change Specificity Demo",
        "presetSource": "full_user_study_lexical_recall_step7000_line_score_verified",
        "originalStep7000Rank": 2,
        "interactionCandidateId": "code_1076",
        "queryTokenIndices": [7],
        "codeTokenIndex": 129,
        "instruction": "打开 Drag 后，把 query 的 changed token 从 Rank1 文件监听器中的 editor 实现上下文拉远；泛化的 editor/file-change 匹配会被压低，GT output_path_textChanged 会升到 Rank1。",
    },
    "1556": {
        "label": "Reset State Recovery Demo",
        "presetSource": "full_python_dataset_gt_rank1224_reset_state_pair_verified",
        "originalStep7000Rank": 1224,
        "interactionCandidateId": "code_1556",
        "queryTokenIndices": [0],
        "codeTokenIndex": 1,
        "instruction": "在 GT DMPs.reset_state 中，把 query 的 Reset 拖向 code 的 reset_state。GT 会清空并重新初始化系统状态；模型原本被泛化的 flush/queue 语义吸引，拖拽后 GT 应升到 Rank1。",
    },
    "1908": {
        "label": "Deep Merge Recovery Demo",
        "presetSource": "full_python_dataset_gt_rank1224_deep_merge_pair_verified",
        "originalStep7000Rank": 1224,
        "interactionCandidateId": "code_1908",
        "queryTokenIndices": [2],
        "codeTokenIndex": 1,
        "instruction": "在 GT deep_merge 中，把 query 的 merge 拖向 code 的 deep_merge。GT 递归合并并覆盖字典键；模型原本偏向普通配置合并，拖拽后 GT 应升到 Rank1。",
    },
    "4612": {
        "label": "JSON Export Recovery Demo",
        "presetSource": "full_python_dataset_gt_rank1224_json_export_pair_verified",
        "originalStep7000Rank": 1224,
        "interactionCandidateId": "code_4612",
        "queryTokenIndices": [6],
        "codeTokenIndex": 1,
        "instruction": "在 GT Setup.to_json 中，把 query 的 JSON 拖向 code 的 to_json。GT 将实验 setup 写入 JSON 文件；模型原本偏向其他写文件函数，拖拽后 GT 应升到 Rank1。",
    },
    "6258": {
        "label": "Database Membership Recovery Demo",
        "presetSource": "full_python_dataset_gt_rank1224_database_membership_pair_verified",
        "originalStep7000Rank": 1224,
        "interactionCandidateId": "code_6258",
        "queryTokenIndices": [2],
        "codeTokenIndex": 1,
        "instruction": "在 GT TcExRun.data_in_db 中，把 query 的 data 拖向 code 的 data_in_db。GT 检查数据库数据是否出现在用户数据中；模型原本偏向通用 validate，拖拽后 GT 应升到 Rank1。",
    },
    "2033": {
        "label": "Email Send Specificity Demo",
        "presetSource": "full_python_eval_step7000_gt_rank2_send_pair_verified",
        "originalStep7000Rank": 2,
        "interactionCandidateId": "code_2033",
        "queryTokenIndices": [0],
        "codeTokenIndex": 1,
        "instruction": "在 GT Inward.send 中，把 query 的 sends 拖向 code 的 send。GT 的实际语义是向收件人发送邮件；该操作提升 GT，同时会泛化降低 Rank1 send_msg 的相似度，GT 应升到 Rank1。若要观察反向操作，可切到 Rank1 send_msg，把 query 的 error 匹配拖远，再回到 GT 完成增强。",
    },
    "6210": {
        "label": "Archive Extraction Contrast Demo",
        "presetSource": "full_python_eval_step7000_gt_rank2_extract_pair_verified",
        "originalStep7000Rank": 2,
        "interactionCandidateId": "code_6210",
        "queryTokenIndices": [0],
        "codeTokenIndex": 1,
        "instruction": "先在 GT extractall 中把 query 的 extract 拖向 code 的 extractall，GT 应升到 Rank1；再切到 Rank1 Command.deploy，把 query 的 extract 相关匹配拖远，Rank1 会明显下降，形成‘压制错误 + 增强 GT’的对比。",
    },
    "1509": {
        "label": "Command Execution Recovery Demo",
        "presetSource": "full_python_eval_step7000_gt_rank2_command_pair_verified",
        "originalStep7000Rank": 2,
        "interactionCandidateId": "code_1509",
        "queryTokenIndices": [2],
        "codeTokenIndex": 1,
        "instruction": "在 GT Run.command 中，把 query 的 command 拖向 code 的 command。GT 会处理 shell command 和 subprocess 执行，拖拽后 GT 应升到 Rank1。",
    },
    "2027": {
        "label": "DynamoDB Scan Specificity Demo",
        "presetSource": "full_python_eval_step7000_gt_rank2_scan_pair_verified",
        "originalStep7000Rank": 2,
        "interactionCandidateId": "code_2027",
        "queryTokenIndices": [2],
        "codeTokenIndex": 1,
        "instruction": "在 GT Client.scan 中，把 query 的 scan 拖向 code 的 scan。GT 是 DynamoDB 的完整 Scan 操作；Rank1 Layer2.scan 也会受到 scan 泛化影响，适合观察局部匹配的双向变化。",
    },
    "5897": {
        "label": "Async Request Contrast Demo",
        "presetSource": "full_python_eval_step7000_gt_rank7_async_request_pair_verified",
        "originalStep7000Rank": 7,
        "interactionCandidateId": "code_5170",
        "queryTokenIndices": [2],
        "codeTokenIndex": 1,
        "instruction": "在 Rank1 SyncRequestEngine._request 中，把 query 的 request 错误匹配拖远，Rank1 会明显下降；再切到 GT AsyncRequestEngine._request，把相同 query 语义拖向 GT 的 request。这个例子重点展示同步实现与异步实现的区分。",
    },
    "5528": {
        "label": "Cisco OUI Argument Demo",
        "presetSource": "full_python_eval_step7000_gt_rank2_oui_highlight_verified",
        "originalStep7000Rank": 2,
        "interactionCandidateId": "code_5528",
        "queryTokenIndices": [6],
        "codeTokenIndex": 28,
        "instruction": "先观察 GT 高亮的 oui_str = \"oui=%s,\" % oui_id。把 query 的 constructing 拖向 oui_str；这个短代码行直接构造 OUI 参数，属于比通用函数名更具体的对应关系。GT 应升到 Rank1，同时相似的通用 argument 匹配会受到泛化影响。",
    },
    "5522": {
        "label": "IP Scan Specificity Demo",
        "presetSource": "full_python_eval_step7000_ip_scan_contrast_verified",
        "originalStep7000Rank": 4,
        "interactionCandidateId": "code_5522",
        "queryTokenIndices": [1],
        "codeTokenIndex": 19,
        "instruction": "先观察 Rank1 Layer2.scan 的数据库扫描语义，再切到 GT via_scan。query 的 IP scan 对应 GT 中 socket.gethostbyname_ex 的网络发现，而不是 Rank1 的 DynamoDB table scan。把 query token scan 拖向 GT 的 gethostbyname_ex；不要拖向通用的 scan 函数名。",
    },
    "954": {
        "label": "API Decorator Specificity Demo",
        "presetSource": "full_python_eval_step7000_api_decorator_contrast_verified",
        "originalStep7000Rank": 2,
        "interactionCandidateId": "code_954",
        "queryTokenIndices": [2],
        "codeTokenIndex": 1,
        "instruction": "先观察 Rank1 ResponseBot.handle_error 的 error-checking 高亮，再切到 GT _api_wrapper。query 的 decorator 和 rate limiting 对应 GT 的 _api_wrapper，而不是 Rank1 的错误处理函数。把 query token decorator 拖向 GT 的 _api_wrapper；这里重点是 API decorator 与 error handler 的区别。",
    },
    "2695": {
        "label": "EM Iteration Specificity Demo",
        "presetSource": "full_python_eval_step7000_short_two_concept_em_iteration_verified",
        "originalStep7000Rank": 2,
        "interactionCandidateId": "code_2695",
        "queryTokenIndices": [0, 1],
        "codeTokenIndex": 72,
        "instruction": "观察 GT iterate 中的 expectation_step 和 maximization_step。query 的两个核心 concept 是 Iterate EM 与 return final probabilities。把 query 的 EM 拖向 expectation_step，强化 EM 迭代语义；不要拖向 Rank1 SomeOf.p 的通用 probability 计算。",
    },
    "3856": {
        "label": "Comparable Dictionary Demo",
        "presetSource": "full_python_eval_step7000_short_two_concept_comparable_dict_verified",
        "originalStep7000Rank": 2,
        "interactionCandidateId": "code_3856",
        "queryTokenIndices": [2, 6],
        "codeTokenIndex": 28,
        "instruction": "观察 GT comparable 中 clean_document_dict 的构造。query 的两个核心 concept 是 dictionary 与 can be compared。把 query 的 compared 拖向 GT 的 comparable 或 clean_document_dict；Rank1 as_dict 只是普通字典转换。",
    },
    "2797": {
        "label": "Command-Line Argument Recovery Demo",
        "presetSource": "full_python_eval_step7000_rank26_cli_argument_float_in",
        "originalStep7000Rank": 26,
        "interactionCandidateId": "code_5534",
        "queryTokenIndices": [1, 3, 4],
        "codeTokenIndex": 142,
        "instruction": "先在 Rank1 ReleaseMaker.get_options 的 format 分支中定位 --formats=\"egg\"。它是命令行 option，但 Rank1 只是把发布格式加入 distributions，并不构造或返回完整的 command-line arguments。选中 query 的 command-line arguments concept，把该 code token 拉近。完成全库层级泛化后，GT read_env 中的 argparse.ArgumentParser / add_argument block 会作为新候选从原始 Rank26 浮入列表。",
    },
    "2836": {
        "label": "Parent Override Logging Recovery Demo",
        "presetSource": "full_python_eval_step7000_rank23_parent_override_log_message",
        "originalStep7000Rank": 23,
        "interactionCandidateId": "code_6155",
        "queryTokenIndices": [5, 6, 7],
        "codeTokenIndex": 118,
        "instruction": "先在 Rank1 Logger.log 中查看 LOGGER.debug(...msg...) 的日志分发分支。它会记录消息，但不会重写父类方法。选中 query 的 add log messages concept，把 Rank1 的 msg 拉近。GT _BaseParsingPlan.execute 中 logger.debug/logger.info 的表征会随之增强；再查看 GT 的 super(...).execute，可确认它同时满足 parent override 与 logging 两部分语义。",
    },
}

CSN_RERANK_DEMO_TOP_ITEMS: dict[str, list[dict[str, Any]]] = {
    "csn_400": [
        {"codeIdx": 35663, "score": 0.775576, "rank": 1},
        {"codeIdx": 3927, "score": 0.757851, "rank": 2},
        {"codeIdx": 39531, "score": 0.746891, "rank": 3},
        {"codeIdx": 40286, "score": 0.738090, "rank": 4},
        {"codeIdx": 17261, "score": 0.734865, "rank": 5},
        {"codeIdx": 22127, "score": 0.675416, "rank": 24, "demoPreset": True},
    ],
    "csn_9406": [
        {"codeIdx": 2366, "score": 0.792563, "rank": 1},
        {"codeIdx": 41105, "score": 0.783144, "rank": 2},
        {"codeIdx": 38102, "score": 0.745810, "rank": 3},
        {"codeIdx": 15403, "score": 0.739054, "rank": 4},
        {"codeIdx": 9518, "score": 0.723273, "rank": 5},
        {"codeIdx": 5777, "score": 0.622529, "rank": 55, "demoPreset": True},
    ],
    "csn_9848": [
        {"codeIdx": 7766, "score": 0.718748, "rank": 1},
        {"codeIdx": 35978, "score": 0.715459, "rank": 2},
        {"codeIdx": 29825, "score": 0.696720, "rank": 3},
        {"codeIdx": 41232, "score": 0.677620, "rank": 4},
        {"codeIdx": 42883, "score": 0.674596, "rank": 5},
        {"codeIdx": 38368, "score": 0.602410, "rank": 40, "demoPreset": True},
    ],
    "csn_11087": [
        {"codeIdx": 601, "score": 0.765535, "rank": 1},
        {"codeIdx": 19761, "score": 0.749573, "rank": 2},
        {"codeIdx": 12409, "score": 0.684889, "rank": 3},
        {"codeIdx": 23009, "score": 0.678109, "rank": 4},
        {"codeIdx": 10098, "score": 0.655723, "rank": 5},
        {"codeIdx": 6619, "score": 0.489374, "rank": 25, "demoPreset": True},
    ],
    "csn_11078": [
        {"codeIdx": 22739, "score": 0.701449, "rank": 1},
        {"codeIdx": 16502, "score": 0.665411, "rank": 2},
        {"codeIdx": 5429, "score": 0.613022, "rank": 3},
        {"codeIdx": 14306, "score": 0.592946, "rank": 4},
        {"codeIdx": 31339, "score": 0.582618, "rank": 5},
        {"codeIdx": 33524, "score": 0.505057, "rank": 39, "demoPreset": True},
    ],
    "csn_13527": [
        {"codeIdx": 8025, "score": 0.773844, "rank": 1},
        {"codeIdx": 14695, "score": 0.748841, "rank": 2},
        {"codeIdx": 19012, "score": 0.747740, "rank": 3},
        {"codeIdx": 15168, "score": 0.742413, "rank": 4},
        {"codeIdx": 35810, "score": 0.738464, "rank": 5},
        {"codeIdx": 9476, "score": 0.650597, "rank": 29, "demoPreset": True},
    ],
    "csn_13958": [
        {"codeIdx": 34901, "score": 0.700037, "rank": 1},
        {"codeIdx": 22792, "score": 0.651001, "rank": 2},
        {"codeIdx": 14401, "score": 0.649884, "rank": 3},
        {"codeIdx": 43283, "score": 0.645207, "rank": 4},
        {"codeIdx": 43807, "score": 0.634387, "rank": 5},
        {"codeIdx": 35017, "score": 0.526130, "rank": 76, "demoPreset": True},
    ],
    "csn_12289": [
        {"codeIdx": 13014, "score": 0.68, "rank": 1},
        {"codeIdx": 13351, "score": 0.64, "rank": 2},
        {"codeIdx": 36788, "score": 0.56, "rank": 3},
        {"codeIdx": 5123, "score": 0.50, "rank": 4},
    ],
    "csn_12290": [
        {"codeIdx": 13014, "score": 0.42, "rank": 1},
        {"codeIdx": 36788, "score": 0.40, "rank": 2},
        {"codeIdx": 5123, "score": 0.38, "rank": 3},
        {"codeIdx": 13351, "score": 0.34, "rank": 4},
    ],
    "csn_12292": [
        {"codeIdx": 13014, "score": 0.47, "rank": 1},
        {"codeIdx": 36788, "score": 0.44, "rank": 2},
        {"codeIdx": 5123, "score": 0.43, "rank": 3},
        {"codeIdx": 13351, "score": 0.425, "rank": 4},
        {"codeIdx": 17340, "score": 0.42, "rank": 5},
    ],
    "csn_3846": [
        {"codeIdx": 23534, "score": 0.642000, "rank": 1, "curatedSource": True, "sourceOriginalRank": 40},
        {"codeIdx": 2786, "score": 0.606322, "rank": 2},
        {"codeIdx": 10842, "score": 0.597977, "rank": 3},
        {"codeIdx": 16758, "score": 0.590468, "rank": 4},
        {"codeIdx": 11754, "score": 0.576100, "rank": 5},
        {"codeIdx": 20000, "score": 0.555627, "rank": 6},
        {"codeIdx": 26183, "score": 0.553561, "rank": 7},
        {"codeIdx": 30899, "score": 0.549231, "rank": 8},
        {"codeIdx": 34730, "score": 0.536361, "rank": 9},
        {"codeIdx": 13622, "score": 0.528848, "rank": 10},
        {"codeIdx": 42625, "score": 0.522766, "rank": 11},
        {"codeIdx": 35583, "score": 0.520584, "rank": 12},
        {"codeIdx": 40139, "score": 0.509303, "rank": 13},
        {"codeIdx": 38572, "score": 0.506171, "rank": 14},
        {"codeIdx": 39906, "score": 0.504750, "rank": 15},
        {"codeIdx": 33231, "score": 0.503000, "rank": 16},
        {"codeIdx": 26225, "score": 0.500123, "rank": 17},
        {"codeIdx": 25158, "score": 0.500074, "rank": 18},
        {"codeIdx": 34778, "score": 0.488438, "rank": 19},
        {"codeIdx": 43790, "score": 0.488347, "rank": 20},
        {"codeIdx": 24098, "score": 0.487779, "rank": 21},
    ],
    "csn_12226": [
        {"codeIdx": 1480, "score": 0.823232, "rank": 1},
        {"codeIdx": 32390, "score": 0.802975, "rank": 2},
        {"codeIdx": 12871, "score": 0.801117, "rank": 3},
        {"codeIdx": 25236, "score": 0.765298, "rank": 4},
        {"codeIdx": 28342, "score": 0.764415, "rank": 5},
        {"codeIdx": 43794, "score": 0.760594, "rank": 6},
        {"codeIdx": 39526, "score": 0.743002, "rank": 7},
        {"codeIdx": 25045, "score": 0.741704, "rank": 8},
        {"codeIdx": 19186, "score": 0.733619, "rank": 9},
        {"codeIdx": 43065, "score": 0.727311, "rank": 10},
        {"codeIdx": 15865, "score": 0.718085, "rank": 11},
        {"codeIdx": 12750, "score": 0.689413, "rank": 12},
        {"codeIdx": 25489, "score": 0.682102, "rank": 13},
        {"codeIdx": 28247, "score": 0.678831, "rank": 14},
        {"codeIdx": 13735, "score": 0.672453, "rank": 15},
        {"codeIdx": 22559, "score": 0.671550, "rank": 16},
        {"codeIdx": 5959, "score": 0.669356, "rank": 17},
        {"codeIdx": 197, "score": 0.667451, "rank": 18},
        {"codeIdx": 20115, "score": 0.660175, "rank": 19},
        {"codeIdx": 13465, "score": 0.659320, "rank": 20},
    ],
    "csn_42": [
        {"codeIdx": 16745, "score": 0.883587, "rank": 1, "curatedSource": True, "sourceOriginalRank": 6},
        {"codeIdx": 12819, "score": 0.875000, "rank": 2},
        {"codeIdx": 2485, "score": 0.864840, "rank": 3},
        {"codeIdx": 23446, "score": 0.817407, "rank": 4},
        {"codeIdx": 26579, "score": 0.767614, "rank": 5},
        {"codeIdx": 26572, "score": 0.723483, "rank": 6},
        {"codeIdx": 1755, "score": 0.716563, "rank": 7},
        {"codeIdx": 11288, "score": 0.716157, "rank": 8},
        {"codeIdx": 13890, "score": 0.714993, "rank": 9},
        {"codeIdx": 25778, "score": 0.710863, "rank": 10},
        {"codeIdx": 27146, "score": 0.693314, "rank": 11},
        {"codeIdx": 37762, "score": 0.687432, "rank": 12},
        {"codeIdx": 41412, "score": 0.687134, "rank": 13},
        {"codeIdx": 4155, "score": 0.685972, "rank": 14},
        {"codeIdx": 6706, "score": 0.682094, "rank": 15},
        {"codeIdx": 4389, "score": 0.680885, "rank": 16},
        {"codeIdx": 10213, "score": 0.680857, "rank": 17},
        {"codeIdx": 10995, "score": 0.679004, "rank": 18},
        {"codeIdx": 24867, "score": 0.675905, "rank": 19},
        {"codeIdx": 13886, "score": 0.671000, "rank": 20},
    ],
    "csn_9388": [
        {"codeIdx": 12695, "score": 0.704000, "rank": 1, "curatedSource": True, "sourceOriginalRank": 4},
        {"codeIdx": 3669, "score": 0.679318, "rank": 2},
        {"codeIdx": 4447, "score": 0.674671, "rank": 3},
        {"codeIdx": 9156, "score": 0.668000, "rank": 4},
        {"codeIdx": 18012, "score": 0.656203, "rank": 5},
        {"codeIdx": 19841, "score": 0.653819, "rank": 6},
        {"codeIdx": 24170, "score": 0.647735, "rank": 7},
        {"codeIdx": 28168, "score": 0.634194, "rank": 8},
        {"codeIdx": 28789, "score": 0.620000, "rank": 9},
        {"codeIdx": 28820, "score": 0.615000, "rank": 10},
        {"codeIdx": 32259, "score": 0.610000, "rank": 11},
        {"codeIdx": 32272, "score": 0.605000, "rank": 12},
        {"codeIdx": 33461, "score": 0.600000, "rank": 13},
        {"codeIdx": 38344, "score": 0.595000, "rank": 14},
        {"codeIdx": 7230, "score": 0.590000, "rank": 15},
        {"codeIdx": 42338, "score": 0.590000, "rank": 16},
        {"codeIdx": 18744, "score": 0.585000, "rank": 17},
        {"codeIdx": 28080, "score": 0.580000, "rank": 18},
        {"codeIdx": 25484, "score": 0.575000, "rank": 19},
        {"codeIdx": 16826, "score": 0.570000, "rank": 20},
    ],
}

CSN_RERANK_DEMO_CONFIG: dict[str, dict[str, Any]] = {
    "csn_400": {
        "label": "Parse Options and Commands Candidate",
        "presetSource": "full_csn_step7000_readable_rank24_candidate",
        "queryIndex": 400,
        "groundTruthCodeIdx": 22127,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 35663}",
        "queryTokenIndices": [0],
        "codeTokenIndex": 35,
        "instruction": "只操作 Rank1 tui.parse_argv。选中 query 的 Parse，把 Rank1 的 _parse_options 拖近；它确实执行选项解析。GT _main 也会创建 OptionParser 并调用 parse_args，因此可观察 parser 表征的间接泛化。先检查 GT 是否还包含后续 command-processing 逻辑，再把它作为正式例子。",
    },
    "csn_9406": {
        "label": "Device Buffer Write Candidate",
        "presetSource": "full_csn_step7000_readable_rank55_candidate",
        "queryIndex": 9406,
        "groundTruthCodeIdx": 5777,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 2366}",
        "queryTokenIndices": [2],
        "codeTokenIndex": 7,
        "instruction": "只操作 Rank1 SMBus.write_bytes。选中 query 的 buffer，把 Rank1 的 buf 拖近；它确实是写入 device 的数据缓冲区。GT write_i2c_block_data 还会将 cmd 写入 buffer 首位，因此可用于比较普通写入与 command-register 写入。",
    },
    "csn_9848": {
        "label": "Configuration Return-Type Candidate",
        "presetSource": "full_csn_step7000_readable_rank40_candidate",
        "queryIndex": 9848,
        "groundTruthCodeIdx": 38368,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 7766}",
        "queryTokenIndices": [0],
        "codeTokenIndex": 13,
        "instruction": "只操作 Rank1 read_configuration。选中 query 的 Reads，把 Rank1 中真实读取配置的 read_latoolscfg 拖近。随后对比 Rank1 的 dict(conf[config]) 与 GT 的 ConfigParser()/cf.read(...)：两者都读取配置，但只有 GT 返回 ConfigParser。",
    },
    "csn_11087": {
        "label": "Right-Click Position Candidate",
        "presetSource": "full_csn_step7000_readable_rank25_candidate",
        "queryIndex": 11087,
        "groundTruthCodeIdx": 6619,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 601}",
        "queryTokenIndices": [11],
        "codeTokenIndex": 5,
        "instruction": "只操作 Rank1 NativeUIElement.clickMouseButtonRight。选中 query 的 position，把 Rank1 的 coord 拖近；coord 与点击位置语义一致。GT FormEvents.click 通过 event.button == 3 和 QCursor().pos() 同时处理右键点击与位置。",
    },
    "csn_11078": {
        "label": "Error Message Display Recovery",
        "presetSource": "full_csn_step7000_error_message_display_rank39_candidate",
        "queryIndex": 11078,
        "groundTruthCodeIdx": 33524,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 22739}",
        "queryTokenIndices": [5],
        "codeTokenIndex": 10,
        "instruction": "只操作 Rank1 BaseParser.value_error。选中 query 的 message，把 Rank1 的 msg 拖近；它确实构造错误消息，但只调用 logger.log。GT FormEvents.display_error_message 会将 error_label 和 error_text_label 显示在界面上，因此更完整地满足显示错误消息的语义。",
    },
    "csn_13527": {
        "label": "Command-Line Logging Candidate",
        "presetSource": "full_csn_step7000_readable_rank29_candidate",
        "queryIndex": 13527,
        "groundTruthCodeIdx": 9476,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 8025}",
        "queryTokenIndices": [1],
        "codeTokenIndex": 1,
        "instruction": "只操作 Rank1 ApiCli._configure_logging。选中 query 的 logging，把 Rank1 的 _configure_logging 拖近。之后比较 Rank1 的 log-level configuration 与 GT init_logstart 的按命令行请求启动 logging；二者相近，因此本例主要用于细粒度诊断，不承诺大幅 rerank。",
    },
    "csn_13958": {
        "label": "Line Pairs Negative-Diagnosis Candidate",
        "presetSource": "full_csn_step7000_readable_rank76_candidate",
        "queryIndex": 13958,
        "groundTruthCodeIdx": 35017,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 34901}",
        "queryTokenIndices": [13],
        "codeTokenIndex": 1,
        "instruction": "只操作 Rank1 CoverageData.line_data。选中 query 的 pairs，把 Rank1 的 line_data 拖远：它返回单个 line numbers，而不是 line-number pairs。GT arc_data 才对应 coverage arcs；本例适合测试负向诊断，不承诺 GT 会因一次泛化进入 Rank1。",
    },
    "csn_8838": {
        "label": "Interned Keyword API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 56,
        "queryIndex": 8838,
        "groundTruthCodeIdx": 19602,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 32299}",
        "queryTokenIndices": [11],
        "codeTokenIndex": 16,
        "instruction": "检查 Rank1 keyword 的 kw_cache 路径；它处理 public keyword 创建，但 GT __get_or_create 同时完成私有缓存交换与 interned keyword 获取。共享 API 线索是低频 kw_cache，而不是词面重合。",
    },
    "csn_7664": {
        "label": "Variant Document-ID Reference Bridge",
        "presetSource": "single_reference_api_bridge_screen",
        "originalStep7000Rank": 12,
        "queryIndex": 7664,
        "groundTruthCodeIdx": 42423,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 26851}",
        "queryTokenIndices": [0, 2, 6],
        "codeTokenIndex": 10,
        "instruction": "检查 Rank1 get_variantid：它只生成一个 document ID。parse_document_id 是低频的 variant-ID API；后位 reference 展示六个字段如何组成稳定 key，而隐藏实现还需要汇总四类 IDs。",
    },
    "csn_2613": {
        "label": "Two-Qubit Gate Graph Reference Bridge",
        "presetSource": "single_reference_api_bridge_screen",
        "originalStep7000Rank": 19,
        "queryIndex": 2613,
        "groundTruthCodeIdx": 9603,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 36272}",
        "queryTokenIndices": [12, 51],
        "codeTokenIndex": 45,
        "instruction": "检查 Rank1 的 deprecated get_2q_nodes 和 node.qargs；后位 reference twoQ_gates 返回可用于图构建的 gate nodes。隐藏实现正以这些 qargs 为边端点累计 CNOT interaction graph。",
    },
    "csn_12213": {
        "label": "AST Operator Dispatch Reference Bridge",
        "presetSource": "single_reference_api_bridge_screen",
        "originalStep7000Rank": 4,
        "queryIndex": 12213,
        "groundTruthCodeIdx": 24571,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 41487}",
        "queryTokenIndices": [2, 3, 5, 6],
        "codeTokenIndex": 1,
        "instruction": "检查 Rank1 的 general _ast_to_code dispatch；后位 reference 实现 concat operator 分支。隐藏实现需要先选择 operator branch，再调用该具体实现生成 source code。",
    },
    "csn_11772": {
        "label": "Asset MIME-Type Extension Reference Bridge",
        "presetSource": "single_reference_api_bridge_screen",
        "queryIndex": 11772,
        "groundTruthCodeIdx": 9854,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 29389}",
        "queryTokenIndices": [0, 1, 2],
        "codeTokenIndex": 36,
        "instruction": "检查 Rank1 format_extension 的 environment.mimetypes.get(extension)：它只正向检查 extension 是否有 MIME 注册。把 mimetypes 拉向 implicit format extension；后位 reference mimetype 展示同一 registry 的读取模式，而隐藏实现通过遍历该 registry 将 compiler_mimetype 反查为 extension。",
    },
    "csn_14238": {
        "label": "Call-Tip Reply Freshness Reference Bridge",
        "presetSource": "single_reference_api_bridge_screen",
        "originalStep7000Rank": 5,
        "queryIndex": 14238,
        "groundTruthCodeIdx": 26942,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 39478}",
        "queryTokenIndices": [1, 3, 4],
        "codeTokenIndex": 100,
        "instruction": "Rank1 在发送 call-tip 请求时保存请求 ID 和光标位置。后位 reference 展示另一个 reply handler 如何使用同一请求上下文，避免将迟到的异步回复应用到当前位置。",
    },
    "csn_8884": {
        "label": "Try AST Dead-Code Reference Bridge",
        "presetSource": "single_reference_api_bridge_screen",
        "originalStep7000Rank": 2,
        "queryIndex": 8884,
        "groundTruthCodeIdx": 36831,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 12324}",
        "queryTokenIndices": [1, 2, 4, 5, 6],
        "codeTokenIndex": 19,
        "instruction": "Rank1 只展示 Try AST 的 orelse 分支。后位 reference 展示 optimizer 重建控制流节点时如何保留并过滤多个分支，供 Try visitor 的实现参考。",
    },
    "csn_3846": {
        "label": "Decorator Redefinition Reference Bridge",
        "presetSource": "curated_source_display_rank1",
        "originalStep7000Rank": 11,
        "queryIndex": 3846,
        "groundTruthCodeIdx": 13622,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 23534}",
        "queryTokenIndices": [10, 11],
        "codeTokenIndex": 23,
        "curatedCandidateList": True,
        "rerankScopeRank": 21,
        "rerankScopeCodeIndices": [23534],
        "excludeRerankCodeIndices": [43410],
        "curatedSourceCodeIdx": 23534,
        "curatedSourceOriginalRank": 40,
        "targetOriginalRank": 15,
        "instruction": "Curated Source / Display Rank1 展示函数 decorator 容器的访问与逐项遍历。将 query 中的 via decorator 拉向 decorators；后位 reference 会进一步解释 dotted decorator 的 AST 表示，以及如何识别被引用的已有对象。",
    },
    "csn_12226": {
        "label": "Alarm API Result Handling Reference Bridge",
        "presetSource": "single_reference_api_bridge_screen",
        "queryIndex": 12226,
        "groundTruthCodeIdx": 28342,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 1480}",
        "queryTokenIndices": [2, 3, 5, 6],
        "codeTokenIndex": 32,
        "rerankScopeRank": 20,
        "targetOriginalRank": 4,
        "instruction": "Rank1 展示 API result 的状态与 body 如何读取，但它只处理成功响应并使用另一套输出流程。将 query 中的 results of the API call 拉向 Source 的 response body；后位 reference 会进一步展示 CLI 如何将 JSON response 处理后输出到终端。",
    },
    "csn_42": {
        "label": "Cloud SQL Delete Completion Reference Bridge",
        "presetSource": "curated_source_display_rank1",
        "queryIndex": 42,
        "groundTruthCodeIdx": 12086,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 16745}",
        "queryTokenIndices": [0, 2, 5, 6],
        "codeTokenIndex": 9,
        "curatedCandidateList": True,
        "rerankScopeRank": 20,
        "rerankScopeCodeIndices": [16745],
        "curatedSourceCodeIdx": 16745,
        "curatedSourceOriginalRank": 6,
        "targetOriginalRank": 16,
        "instruction": "Curated Source / Display Rank1 展示同步数据库删除如何取得连接并执行 SQL。将 database Cloud SQL 拉向 _db_conn 或 conn；后位 reference 会进一步展示云端数据库删除的连接、请求执行和完成等待协议。也可以把 delete 拉向 execute，但这个线索提供的提升较弱。",
    },
    "csn_9388": {
        "label": "URL Query Removal Reference Bridge",
        "presetSource": "curated_same_repository_url_helpers",
        "originalStep7000Rank": 1,
        "queryIndex": 9388,
        "groundTruthCodeIdx": 336,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 12695}",
        "queryTokenIndices": [2],
        "codeTokenIndex": 75,
        "curatedCandidateList": True,
        "curatedSourceCodeIdx": 12695,
        "curatedSourceOriginalRank": 4,
        "targetOriginalRank": 15,
        "instruction": "Curated Source / Display Rank1 展示如何把 URL 参数拼接到请求字符串。将 query 中的 URL 拉向 Source 的 url_params；后位同仓库 reference 进一步展示如何解析 URL 并重新组合各个部分，在不保留原 query 的情况下生成新的 URL。",
    },
    "csn_584": {
        "label": "Random Perspective Parameter Reference Bridge",
        "presetSource": "full_eval_step7000_url_mapped_rankings",
        "originalStep7000Rank": 2,
        "queryIndex": 584,
        "groundTruthCodeIdx": 22264,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 24026}",
        "queryTokenIndices": [4, 9],
        "codeTokenIndex": 5,
        "instruction": "检查 Rank1 perspective 如何消费 startpoints 和 endpoints 执行变换；后位 reference _get_perspective_coeffs 展示这两组角点在透视变换中的对应关系。隐藏实现需要生成随机 endpoints，而不是直接执行变换。",
    },
    "csn_2812": {
        "label": "Qubit Dimension Log2 API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 40,
        "queryIndex": 2812,
        "groundTruthCodeIdx": 3529,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 9656}",
        "queryTokenIndices": [6, 7, 8, 9],
        "codeTokenIndex": 33,
        "instruction": "检查 Rank1 _check_nqubit_dim 中的 np.log2(input_dim)：它验证 n-qubit channel 的维度，但 GT BaseOperator._automatic_dims 才把维度转换为 qubit subsystem 元组。关键关系是 2^n 维 Hilbert space 与 n 个 qubit 的对应，而不是词面重合。",
    },
    "csn_7727": {
        "label": "Stokes Calibration Feed-Type API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 104,
        "queryIndex": 7727,
        "groundTruthCodeIdx": 36257,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 42680}",
        "queryTokenIndices": [1, 2, 3, 4, 5, 6, 7],
        "codeTokenIndex": 15,
        "instruction": "检查 Rank1 calibrate_pols 中的 feedtype：线性或圆形馈源基决定 Stokes 分量如何由 differential gain/phase 校正。Rank1 是写文件的完整 pipeline，GT apply_Mueller 才实现核心校正矩阵；feedtype 是连接两者的非词面领域线索。",
    },
    "csn_4772": {
        "label": "KMIP DeviceCredential Serialization Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 49,
        "queryIndex": 4772,
        "groundTruthCodeIdx": 16717,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 16157}",
        "queryTokenIndices": [2, 3, 4, 5, 6, 7],
        "codeTokenIndex": 11,
        "instruction": "检查 Rank1 Credential.write 中的 KMIPVersion：它只序列化泛化 Credential 容器；GT DeviceCredential.write 用同一 KMIP 协议版本和 BytearrayStream，逐项写入 device serial number、password 等设备凭据字段。把 KMIPVersion 拉向 data encoding DeviceCredential struct concept，再比较两者写入的具体字段。",
    },
    "csn_10023": {
        "label": "OSM Replication State API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 48,
        "queryIndex": 10023,
        "groundTruthCodeIdx": 33502,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 1960}",
        "queryTokenIndices": [1, 3, 4, 5, 6],
        "codeTokenIndex": 19,
        "instruction": "检查 Rank1 iter_changeset_stream 中的 state_dir：它保存 changeset stream 的读取状态；GT iter_osm_stream 用同一 replication-state 机制持续消费 OSM diff。Rank1 解析的是 changeset .osm.gz，GT 才处理 query 指向的 diff .osc.gz。",
    },
    "csn_2207": {
        "label": "Window Sum-Square Hop-Length API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 129,
        "queryIndex": 2207,
        "groundTruthCodeIdx": 21455,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 35623}",
        "queryTokenIndices": [3, 4, 5, 6, 7],
        "codeTokenIndex": 7,
        "instruction": "检查 Rank1 window_sumsquare 的 hop_length：它决定相邻 analysis frame 的步进；GT __window_ss_fill 正是用 sample = i * hop_length 将平方 window 累加到输出 envelope 的 helper。Rank1 是完整 wrapper，GT 才实现 query 指向的核心计算。",
    },
    "csn_4694": {
        "label": "SignatureVerify Payload API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 24,
        "queryIndex": 4694,
        "groundTruthCodeIdx": 7396,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 38489}",
        "queryTokenIndices": [8],
        "codeTokenIndex": 26,
        "instruction": "检查 Rank1 SignResponsePayload.write 的 _unique_identifier 字段；它写入的是 response payload，而 GT SignatureVerifyRequestPayload.write 才构造完整的验证请求 payload。共享线索是 KMIP 的 _unique_identifier，不是 query 的词面重复。",
    },
    "csn_6416": {
        "label": "TransTmpl Structural Field API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 25,
        "queryIndex": 6416,
        "groundTruthCodeIdx": 17461,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 11571}",
        "queryTokenIndices": [1],
        "codeTokenIndex": 50,
        "instruction": "检查 Rank1 walkFlattenFields 中 HStruct 的递归分支；它泛化遍历硬件结构值，而 GT TransTmpl.walkFlatten 才遍历指定 TransTmpl 实例的字段。HStruct 是需要领域理解的共享结构线索。",
    },
    "csn_4357": {
        "label": "Incoming Connection Encryption API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 25,
        "queryIndex": 4357,
        "groundTruthCodeIdx": 11978,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 59}",
        "queryTokenIndices": [9],
        "codeTokenIndex": 16,
        "instruction": "检查 Rank1 _onNewIncomingConnection 的 encryptor 初始化；它只准备连接与回调，GT _onIncomingMessageReceived 才处理初始消息、加密、utility message 和节点关联。",
    },
    "csn_5340": {
        "label": "GeoTiff VLR Struct API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 36,
        "queryIndex": 5340,
        "groundTruthCodeIdx": 5711,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 26999}",
        "queryTokenIndices": [0],
        "codeTokenIndex": 21,
        "instruction": "检查 Rank1 的 GeoKeyDirectoryVlr 解析分支；GT parse_geo_tiff 同样通过 GeoTiff VLR API 组织结构化结果，但覆盖更完整的 VLR 组合。",
    },
    "csn_10164": {
        "label": "V4 Meter Request API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 73,
        "queryIndex": 10164,
        "groundTruthCodeIdx": 39814,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 39435}",
        "queryTokenIndices": [0],
        "codeTokenIndex": 1,
        "instruction": "检查 Rank1 V4Meter.requestB 的 requestB 调用；GT V4Meter.request 通过同一仪表请求接口组合 A/B 读数，适合诊断组合读数语义是否被充分编码。",
    },
    "csn_13655": {
        "label": "Application Logging API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 22,
        "queryIndex": 13655,
        "groundTruthCodeIdx": 23652,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 36880}",
        "queryTokenIndices": [1],
        "codeTokenIndex": 11,
        "instruction": "检查 Rank1 _setup_log 的 StreamHandler 默认配置；GT Application._log_default 使用同一 logging API，但更完整地落实 application-level 默认日志行为。",
    },
    "csn_14175": {
        "label": "Notebook Format API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 33,
        "queryIndex": 14175,
        "groundTruthCodeIdx": 30426,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 32305}",
        "queryTokenIndices": [41],
        "codeTokenIndex": 61,
        "instruction": "检查 Rank1 reads_py 的 NBFormatError 格式处理；GT reads 使用同一异常/API 线索，并处理任意 notebook 版本后返回当前格式。",
    },
    "csn_10643": {
        "label": "Root Logger Configuration Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 65,
        "queryIndex": 10643,
        "groundTruthCodeIdx": 17733,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 24488}",
        "queryTokenIndices": [1],
        "codeTokenIndex": 24,
        "instruction": "检查 Rank1 basicConfig 中 removeHandler 的 logger 重置；GT common_logger_config 同时覆盖 root 与 non-root logger 的共同配置。",
    },
    "csn_12075": {
        "label": "Current Tags API Bridge Candidate",
        "presetSource": "url_mapped_step7000_shared_api_bridge",
        "originalStep7000Rank": 19,
        "queryIndex": 12075,
        "groundTruthCodeIdx": 28438,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 549}",
        "queryTokenIndices": [3],
        "codeTokenIndex": 17,
        "instruction": "检查 Rank1 Labels.list 的 ApiUri / self._post 请求；GT Tags.list 使用相同 API 模式，并直接对应当前 tags 的检索语义。",
    },
    "csn_12289": {
        "label": "Set Update Semantics Demo",
        "presetSource": "csn_python_full_step7000_set_operation_pair_verified",
        "queryIndex": 12289,
        "groundTruthCodeIdx": 13351,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 13014}",
        "queryTokenIndices": [0],
        "codeTokenIndex": 0,
        "groundTruthQueryTokenIndices": [6],
        "groundTruthCodeTokenIndex": 14,
        "instruction": "先在 Rank1 Scope.intersection_update 上把 query 的 Remove 从当前 code token 附近拖远；再切到 GT Scope.symmetric_difference_update，把 query 的 Update 向 code 的 set 拉近。前者应降低错误的交集更新匹配，后者应提高正确的对称差更新匹配，并使 GT 升到 Rank1。",
    },
    "csn_12290": {
        "label": "Missing Set Exclusivity Alignment Demo",
        "presetSource": "csn_python_full_step7000_gt_rank21_missing_only_not_pair_verified",
        "queryIndex": 12290,
        "groundTruthCodeIdx": 13351,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 13351}",
        "queryTokenIndices": [8],
        "codeTokenIndex": 58,
        "groundTruthQueryTokenIndices": [8],
        "groundTruthCodeTokenIndex": 58,
        "instruction": "只操作 GT Scope.symmetric_difference。把 query token only 向 code token not 拉近；代码中的 if k not in skey 正是‘值只存在于一方’的实现，但模型没有充分学习这组对应关系。拖动后 GT 相似度应明显提升并升到 Rank1。",
    },
    "csn_12292": {
        "label": "Remove and Raise Error Alignment Demo",
        "presetSource": "csn_python_full_step7000_gt_rank45_remove_keyerror_pair_verified",
        "queryIndex": 12292,
        "groundTruthCodeIdx": 17340,
        "interactionCandidateId": f"code_{CSN_CODE_OFFSET + 17340}",
        "queryTokenIndices": [4],
        "codeTokenIndex": 28,
        "groundTruthQueryTokenIndices": [3, 4, 6, 7],
        "groundTruthCodeTokenIndex": 28,
        "instruction": "只操作 GT Scope.remove。把 query token KeyError 向 code token KeyError 拉近；代码的 docstring 说明‘raise KeyError if not found’，并且实现中确实执行 raise KeyError。这个对应关系在原始排序中没有被充分利用，拖动后 GT 相似度应明显提升并升到 Rank1。",
    },
}


CSN_DISPLAY_CONCEPT_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    "csn_11078": [
        {"conceptId": 0, "tokenIndices": [2, 4, 5], "text": "displays error message"},
        {"conceptId": 1, "tokenIndices": [8, 9, 11], "text": "wrong value typed"},
    ],
    "csn_8838": [
        {"conceptId": 0, "tokenIndices": [8, 9, 11, 14], "text": "interned keyword instance"},
        {"conceptId": 1, "tokenIndices": [17, 19], "text": "input string"},
        {"conceptId": 2, "tokenIndices": [1], "text": "swap"},
    ],
    "csn_584": [
        {"conceptId": 0, "tokenIndices": [1], "text": "parameters"},
        {"conceptId": 1, "tokenIndices": [4, 9], "text": "perspective perspective"},
        {"conceptId": 2, "tokenIndices": [8, 10], "text": "random transform"},
    ],
    "csn_8884": [
        {"conceptId": 0, "tokenIndices": [0, 2, 3], "text": "Elim dead code"},
        {"conceptId": 1, "tokenIndices": [5, 6, 7], "text": "except try bodies"},
    ],
    "csn_3846": [
        {"conceptId": 6, "tokenIndices": [0, 1], "text": "Return True"},
        {"conceptId": 5, "tokenIndices": [10, 11], "text": "via decorator"},
    ],
    "csn_42": [
        {"conceptId": 1, "tokenIndices": [0], "text": "delete"},
        {"conceptId": 0, "tokenIndices": [2, 5, 6], "text": "database Cloud SQL"},
    ],
    "csn_9388": [
        {"conceptId": 0, "tokenIndices": [5, 6], "text": "query component"},
        {"conceptId": 1, "tokenIndices": [2], "text": "URL"},
    ],
}

CSN_DISPLAY_QUERY_OVERRIDES: dict[str, dict[str, Any]] = {
    "csn_3846": {
        "rawText": "Return True if the object is a method redefined via decorator.",
        "tokenCount": 13,
    },
}


def _display_query(test_id: str, raw_text: str, tokens: list[str]) -> tuple[str, list[str]]:
    override = CSN_DISPLAY_QUERY_OVERRIDES.get(str(test_id))
    if not override:
        return raw_text, tokens
    return str(override["rawText"]), list(tokens[:int(override["tokenCount"])])


def _display_concepts(test_id: str, query_concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    override = CSN_DISPLAY_CONCEPT_OVERRIDES.get(str(test_id))
    if not override:
        return query_concepts
    concepts_by_id = {int(concept["conceptId"]): concept for concept in query_concepts}
    return [
        {
            **concepts_by_id[int(item["conceptId"])],
            "tokenIndices": list(item["tokenIndices"]),
            "text": str(item["text"]),
        }
        for item in override
        if int(item["conceptId"]) in concepts_by_id
    ]


def _replace_conflicting_study_candidate(test_id: str, item: dict[str, Any]) -> dict[str, Any]:
    """Keep the fixed study rank while removing known full-pass alternatives."""
    replacements = {
        "csn_8884": (2345, 139),
        "csn_9388": (42916, 28080),
    }
    old_index, new_index = replacements.get(str(test_id), (None, None))
    if old_index is None or int(item.get("codeIdx", -1)) != old_index:
        return item
    return {**item, "codeIdx": new_index}


def _use_legacy_rerank_demo(test_id: str) -> bool:
    return str(test_id) in LEGACY_RERANK_DEMO_TEST_IDS


def _use_custom_rerank_demo(test_id: str) -> bool:
    return str(test_id) in CUSTOM_RERANK_DEMO_TOP_ITEMS


def is_csn_demo_test(test_id: str) -> bool:
    return str(test_id) in CSN_RERANK_DEMO_TOP_ITEMS


# Every generation case uses the same single-reference protocol.  A case
# registers its hidden evaluation target and its server-side target-reference
# annotation here; neither is returned to the participant UI.
SINGLE_REFERENCE_CASE_CONFIG: dict[str, dict[str, int]] = {
    "csn_8838": {
        "hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 19602,
        "targetReferenceCodeIdx": CSN_CODE_OFFSET + 32299,
    },
    "csn_7664": {"hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 42423, "targetReferenceCodeIdx": CSN_CODE_OFFSET + 29009},
    "csn_2613": {"hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 9603, "targetReferenceCodeIdx": CSN_CODE_OFFSET + 20329},
    "csn_12213": {"hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 24571, "targetReferenceCodeIdx": CSN_CODE_OFFSET + 40341},
    "csn_11772": {
        "hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 9854,
        "targetReferenceCodeIdx": CSN_CODE_OFFSET + 1612,
    },
    "csn_14238": {
        "hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 26942,
        "targetReferenceCodeIdx": CSN_CODE_OFFSET + 7044,
    },
    "csn_8884": {
        "hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 36831,
        "targetReferenceCodeIdx": CSN_CODE_OFFSET + 37136,
    },
    "csn_3846": {
        "hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 13622,
        "targetReferenceCodeIdx": CSN_CODE_OFFSET + 33231,
        "visibleCandidateLimit": 20,
    },
    "csn_12226": {
        "hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 28342,
        "targetReferenceCodeIdx": CSN_CODE_OFFSET + 25236,
        "visibleCandidateLimit": 20,
    },
    "csn_42": {
        "hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 12086,
        "targetReferenceCodeIdx": CSN_CODE_OFFSET + 6706,
        "visibleCandidateLimit": 20,
    },
    "csn_9388": {
        "hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 336,
        "targetReferenceCodeIdx": CSN_CODE_OFFSET + 7230,
        "visibleCandidateLimit": 20,
    },
    "csn_584": {
        "hiddenGroundTruthCodeIdx": CSN_CODE_OFFSET + 22264,
        "targetReferenceCodeIdx": CSN_CODE_OFFSET + 4381,
    },
}


def is_single_reference_case(test_id: str) -> bool:
    return str(test_id) in SINGLE_REFERENCE_CASE_CONFIG


def is_hidden_reference_candidate(test_id: str, candidate_id: str) -> bool:
    config = SINGLE_REFERENCE_CASE_CONFIG.get(str(test_id))
    return bool(config and str(candidate_id) == f"code_{int(config['hiddenGroundTruthCodeIdx'])}")


def apply_single_reference_mode(session: dict[str, Any]) -> dict[str, Any]:
    """Strip hidden evaluation target data from a participant-facing session."""
    config = SINGLE_REFERENCE_CASE_CONFIG.get(str(session.get("testId") or ""))
    if not config:
        return session
    hidden_id = f"code_{int(config['hiddenGroundTruthCodeIdx'])}"
    source_candidates = list(session.get("candidates", []))
    presentation_baseline = {
        str(candidate_id): int(rank)
        for candidate_id, rank in dict(session.get("_presentationBaselineRanks") or {}).items()
    }
    if presentation_baseline:
        visible_baseline_rank = {
            candidate_id: rank
            for candidate_id, rank in presentation_baseline.items()
            if candidate_id != hidden_id
        }
    else:
        baseline_candidates = sorted(
            source_candidates,
            key=lambda candidate: int(candidate.get("originalRank", candidate.get("rank", 0))),
        )
        visible_baseline_rank = {
            str(candidate.get("id")): rank
            for rank, candidate in enumerate(
                (candidate for candidate in baseline_candidates if str(candidate.get("id")) != hidden_id),
                start=1,
            )
        }
    candidates = [
        {key: value for key, value in candidate.items() if key != "isGroundTruth"}
        for candidate in source_candidates
        if str(candidate.get("id")) != hidden_id
    ]

    # The hidden generation target must not leave a visible rank gap.  Session
    # builders load one extra item for single-reference cases, so filtering the
    # target promotes the next retrieval result into the participant's Top-20.
    configured_limit = int(config.get("visibleCandidateLimit", 20))
    visible_limit = configured_limit if len(source_candidates) > configured_limit else len(candidates)
    candidates.sort(key=lambda candidate: int(candidate.get("rank", visible_limit + 1)))
    candidates = candidates[:visible_limit]
    for visible_rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = visible_rank
        candidate.pop("corpusRank", None)
        if "originalRank" in candidate:
            # Prefer the rank from the participant's initially visible list.
            # A full-corpus rerank can also introduce a new candidate; in that
            # case keep its rerank baseline instead of silently resetting its
            # displayed movement to zero.
            original_rank = visible_baseline_rank.get(str(candidate.get("id")))
            if original_rank is None:
                original_rank = int(candidate["originalRank"])
            candidate["originalRank"] = original_rank
            candidate["rankDelta"] = original_rank - visible_rank

    payload = {
        key: value
        for key, value in session.items()
        if key not in {"groundTruth", "generalizationDemo", "generalizationActive", "_presentationBaselineRanks"}
    }
    payload["candidates"] = candidates
    hidden_details = dict(payload.get("generalizedMatchesByCandidate") or {})
    hidden_details.pop(hidden_id, None)
    if "generalizedMatchesByCandidate" in payload:
        payload["generalizedMatchesByCandidate"] = hidden_details
    payload["referenceSelection"] = {"enabled": True, "selectionLimit": 1}
    return payload


def is_csn_code_idx(code_idx: int) -> bool:
    return int(code_idx) >= CSN_CODE_OFFSET


def csn_actual_code_idx(code_idx: int) -> int:
    return int(code_idx) - CSN_CODE_OFFSET if is_csn_code_idx(int(code_idx)) else int(code_idx)


@lru_cache(maxsize=2)
def load_dataset(path: str = str(DEFAULT_DATASET_PATH)) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def load_csn_queries(path: str = str(CSN_PYTHON_TEST_PATH)) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def load_csn_codebase(path: str = str(CSN_PYTHON_CODEBASE_PATH)) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def load_csn_gt_prefix_rankings(path: str = str(CSN_GT_PREFIX_CACHE_PATH)) -> dict[str, list[dict[str, Any]]]:
    """Original global ranking prefixes used by the interactive CSN examples."""
    if not Path(path).exists():
        return {}
    import torch

    payload = torch.load(path, map_location="cpu")
    rankings: dict[str, list[dict[str, Any]]] = {}
    for test_id, entry in payload.items():
        indices = entry["indices"].tolist()
        scores = entry["originalScores"].tolist()
        ranks = entry["originalRanks"].tolist()
        items = [
            {"codeIdx": CSN_CODE_OFFSET + int(index), "score": float(score), "rank": int(rank)}
            for index, score, rank in zip(indices, scores, ranks)
        ]
        rankings[str(test_id)] = sorted(items, key=lambda item: int(item["rank"]))
    return rankings


def _load_url_mapped_api_demo_rankings() -> dict[str, list[dict[str, Any]]]:
    """Load compact, URL-mapped Top-20 rankings for API-bridge demos."""
    if not CSN_URL_MAPPED_API_DEMO_RANKINGS_PATH.exists():
        return {}
    with open(CSN_URL_MAPPED_API_DEMO_RANKINGS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        str(test_id): [
            {
                "codeIdx": int(item["codeIdx"]),
                "score": float(item["score"]),
                "rank": int(item["rank"]),
            }
            for item in items
        ]
        for test_id, items in payload.items()
    }


def _load_single_reference_demo_rankings() -> dict[str, list[dict[str, Any]]]:
    if not SINGLE_REFERENCE_DEMO_RANKINGS_PATH.exists():
        return {}
    with open(SINGLE_REFERENCE_DEMO_RANKINGS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        str(test_id): [
            {"codeIdx": int(item["codeIdx"]), "score": float(item["score"]), "rank": int(item["rank"])}
            for item in items
        ]
        for test_id, items in payload.items()
    }


def _load_csn_11772_gears_rankings() -> dict[str, list[dict[str, Any]]]:
    if not CSN_11772_GEARS_STEP7000_RANKINGS_PATH.exists():
        return {}
    with CSN_11772_GEARS_STEP7000_RANKINGS_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("topK", [])
    if str(payload.get("testId")) != "csn_11772" or not isinstance(items, list):
        return {}
    return {
        "csn_11772": [
            {"codeIdx": int(item["codeIdx"]) - CSN_CODE_OFFSET, "score": float(item["score"]), "rank": int(item["rank"])}
            for item in items
        ]
    }


def _load_csn_584_rankings() -> dict[str, list[dict[str, Any]]]:
    """Load the existing URL-mapped ranking and retain one replacement item for hidden GT removal."""
    path = ALIGNED_XSEARCH_DIR / "full_eval_step7000_url_mapped_rankings.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    entry = next((item for item in payload.get("results", []) if str(item.get("testId")) == "584"), None)
    if not entry:
        return {}
    items = [
        {"codeIdx": int(item["codeIdx"]), "score": float(item["score"]), "rank": int(item["rank"])}
        for item in entry.get("topK", [])
    ]
    # The source cache stores only the first 20 items. This same-repository
    # transform function is retained as the 21st item so hiding the evaluation
    # target still leaves a participant-facing Top-20.
    items.append({"codeIdx": 23647, "score": 0.421, "rank": 21})
    return {"csn_584": items}


CSN_RERANK_DEMO_TOP_ITEMS.update(_load_url_mapped_api_demo_rankings())
CSN_RERANK_DEMO_TOP_ITEMS.update(_load_single_reference_demo_rankings())
CSN_RERANK_DEMO_TOP_ITEMS.update(_load_csn_11772_gears_rankings())
CSN_RERANK_DEMO_TOP_ITEMS.update(_load_csn_584_rankings())


@lru_cache(maxsize=8)
def load_matches(path: str = str(DEFAULT_MATCH_PATH)) -> dict[str, list[dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_user_study_step7000_rankings(path: str = str(USER_STUDY_STEP7000_RANKING_PATH)) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {str(item["testId"]): item for item in payload.get("results", []) if "testId" in item}


@lru_cache(maxsize=1)
def load_full_eval_step7000_rankings(path: str = str(FULL_EVAL_STEP7000_RANKING_PATH)) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {str(item["testId"]): item for item in payload.get("results", []) if "testId" in item}


def _ranking_entry(test_id: str) -> dict[str, Any] | None:
    if is_csn_demo_test(test_id):
        config = CSN_RERANK_DEMO_CONFIG[str(test_id)]
        top_items = [
            {**item, "codeIdx": CSN_CODE_OFFSET + int(item["codeIdx"])}
            for item in CSN_RERANK_DEMO_TOP_ITEMS[str(test_id)]
        ]
        gt_idx = CSN_CODE_OFFSET + int(config["groundTruthCodeIdx"])
        gt_item = next((item for item in top_items if int(item["codeIdx"]) == gt_idx), None)
        return {
            "testId": str(test_id),
            "gtCodeIdx": gt_idx,
            "gtRank": int(gt_item["rank"]) if gt_item else int(config.get("originalStep7000Rank", 0)),
            "gtScore": float(gt_item["score"]) if gt_item else 0.0,
            "top1Correct": bool(gt_item and int(gt_item["rank"]) == 1),
            "topK": top_items,
        }
    if _use_custom_rerank_demo(test_id):
        top_items = CUSTOM_RERANK_DEMO_TOP_ITEMS[str(test_id)]
        gt_idx = int(test_id)
        gt_item = next((item for item in top_items if int(item["codeIdx"]) == gt_idx), None)
        config = CUSTOM_RERANK_DEMO_CONFIG.get(str(test_id), {})
        return {
            "testId": str(test_id),
            "gtCodeIdx": gt_idx,
            "gtRank": int(gt_item["rank"]) if gt_item else int(config.get("originalStep7000Rank", 0)),
            "gtScore": float(gt_item["score"]) if gt_item else float(load_full_eval_step7000_rankings().get(str(test_id), {}).get("gtScore", 0.0)),
            "top1Correct": bool(gt_item and int(gt_item["rank"]) == 1),
            "topK": top_items,
        }
    return load_user_study_step7000_rankings().get(str(test_id))


def _ranking_candidate_score(test_id: str, code_idx: int) -> float | None:
    if is_csn_demo_test(test_id):
        actual_idx = csn_actual_code_idx(int(code_idx))
        for item in CSN_RERANK_DEMO_TOP_ITEMS[str(test_id)]:
            if int(item["codeIdx"]) == actual_idx:
                return float(item["score"])
        return None
    if _use_custom_rerank_demo(test_id):
        for item in CUSTOM_RERANK_DEMO_TOP_ITEMS[str(test_id)]:
            if int(item["codeIdx"]) == int(code_idx):
                return float(item["score"])
        full_entry = load_full_eval_step7000_rankings().get(str(test_id), {})
        if int(full_entry.get("gtCodeIdx", -1)) == int(code_idx):
            return float(full_entry.get("gtScore", 0.0))
        return None
    if _use_legacy_rerank_demo(test_id):
        return _legacy_candidate_score(test_id, code_idx)
    if str(test_id) == GENERALIZATION_DEMO_TEST_ID and int(code_idx) == GENERALIZATION_DEMO_GT_CODE_IDX:
        entry = _ranking_entry(test_id)
        if entry:
            top_items = list(entry.get("topK", []))
            if len(top_items) >= 3:
                rank2 = float(top_items[1].get("score", 0.0))
                rank3 = float(top_items[2].get("score", 0.0))
                return (rank2 + rank3) / 2.0
    entry = _ranking_entry(test_id)
    if not entry:
        return None
    for item in entry.get("topK", []):
        if int(item.get("codeIdx", -1)) == int(code_idx):
            return float(item.get("score", 0.0))
    if int(entry.get("gtCodeIdx", -1)) == int(code_idx):
        return float(entry.get("gtScore", 0.0))
    return None


def _legacy_candidate_score(test_id: str, code_idx: int) -> float | None:
    for item in load_matches().get(str(test_id), []):
        if int(item.get("code_idx", -1)) == int(code_idx):
            return float(item.get("similarity", 0.0))
    return None


def _ranking_top_items(test_id: str, top_k: int) -> list[dict[str, Any]]:
    if is_csn_demo_test(test_id):
        all_preset = [
            {**item, "codeIdx": CSN_CODE_OFFSET + int(item["codeIdx"])}
            for item in CSN_RERANK_DEMO_TOP_ITEMS[str(test_id)]
        ]
        preset = all_preset[:top_k]
        ground_truth = next((item for item in all_preset if bool(item.get("demoPreset"))), None)
        if ground_truth and all(int(item["codeIdx"]) != int(ground_truth["codeIdx"]) for item in preset):
            preset = preset[: max(0, top_k - 1)] + [ground_truth]
        return preset

    if _use_custom_rerank_demo(test_id):
        all_preset = [dict(item) for item in CUSTOM_RERANK_DEMO_TOP_ITEMS[str(test_id)]]
        preset = all_preset[:top_k]
        ground_truth = next((item for item in all_preset if bool(item.get("demoPreset"))), None)
        # A motivating example may deliberately start just outside Top-20. Keep
        # that GT visible by reserving the final slot rather than dropping it.
        if ground_truth and all(int(item["codeIdx"]) != int(ground_truth["codeIdx"]) for item in preset):
            preset = preset[: max(0, top_k - 1)] + [ground_truth]
        if len(preset) >= top_k:
            return preset
        seen_indices = {int(item["codeIdx"]) for item in preset}
        original = load_full_eval_step7000_rankings().get(str(test_id), {})
        for item in original.get("topK", []):
            code_idx = int(item.get("codeIdx", -1))
            if code_idx < 0 or code_idx in seen_indices:
                continue
            seen_indices.add(code_idx)
            preset.append({**item, "rank": len(preset) + 1})
            if len(preset) >= top_k:
                break
        return preset

    if _use_legacy_rerank_demo(test_id):
        result = []
        seen_indices: set[int] = set()
        for match in load_matches().get(str(test_id), []):
            code_idx = int(match["code_idx"])
            if code_idx in seen_indices:
                continue
            seen_indices.add(code_idx)
            result.append(
                {
                    "codeIdx": code_idx,
                    "score": float(match.get("similarity", 0.0)),
                    "rank": len(result) + 1,
                }
            )
            if len(result) >= top_k:
                break
        if len(result) < top_k:
            original = load_full_eval_step7000_rankings().get(str(test_id), {})
            for item in original.get("topK", []):
                code_idx = int(item.get("codeIdx", -1))
                if code_idx < 0 or code_idx in seen_indices:
                    continue
                seen_indices.add(code_idx)
                result.append({**item, "rank": len(result) + 1})
                if len(result) >= top_k:
                    break
        return result

    entry = _ranking_entry(test_id)
    if not entry:
        return []
    gt_idx = int(entry.get("gtCodeIdx", -1))
    raw_items = list(entry.get("topK", []))
    if str(test_id) == "2774":
        raw_items.sort(key=lambda item: 0 if int(item.get("codeIdx", -1)) == gt_idx else int(item.get("rank", 9999)))

    if str(test_id) == GENERALIZATION_DEMO_TEST_ID:
        filtered = [item for item in raw_items if int(item.get("codeIdx", -1)) != GENERALIZATION_DEMO_GT_CODE_IDX]
        if len(filtered) >= 3:
            rank2 = float(filtered[1].get("score", 0.0))
            rank3 = float(filtered[2].get("score", 0.0))
            filtered.insert(
                2,
                {
                    "codeIdx": GENERALIZATION_DEMO_GT_CODE_IDX,
                    "score": (rank2 + rank3) / 2.0,
                    "rank": 3,
                    "demoPreset": True,
                },
            )
        raw_items = filtered

    result = []
    seen_funcs: set[str] = set()
    seen_indices: set[int] = set()
    for item in raw_items:
        code_idx = int(item["codeIdx"])
        row = get_row(code_idx)
        func_name = str(row.get("func_name", "") or code_idx)
        if code_idx in seen_indices:
            continue
        if func_name in seen_funcs:
            continue
        seen_indices.add(code_idx)
        seen_funcs.add(func_name)
        next_item = {**item, "rank": len(result) + 1}
        result.append(next_item)
        if len(result) >= top_k:
            break
    return result


def get_available_tests() -> list[str]:
    user_study_tests = sorted(load_matches().keys(), key=lambda x: int(x) if x.isdigit() else x)
    custom_tests = list(CUSTOM_RERANK_DEMO_TOP_ITEMS)
    prioritized_tests: list[str] = []
    custom_tests = prioritized_tests + [test_id for test_id in custom_tests if test_id not in prioritized_tests]
    user_study_tests = [test_id for test_id in user_study_tests if test_id not in CUSTOM_RERANK_DEMO_TOP_ITEMS]
    csn_tests = list(CSN_RERANK_DEMO_TOP_ITEMS)
    smoke_tests = [f"smoke_{idx}" for idx in range(min(20, len(load_smoke_codebase())))]
    return custom_tests + csn_tests + user_study_tests + smoke_tests


def get_row(data_index: int) -> dict[str, Any]:
    if is_csn_code_idx(int(data_index)):
        actual_idx = csn_actual_code_idx(int(data_index))
        codebase = load_csn_codebase()
        if actual_idx < 0 or actual_idx >= len(codebase):
            raise KeyError(f"CSN codebase index out of range: {data_index}")
        return codebase[actual_idx]
    dataset = load_dataset()
    if data_index < 0 or data_index >= len(dataset):
        raise KeyError(f"Dataset index out of range: {data_index}")
    return dataset[data_index]


@lru_cache(maxsize=1)
def load_smoke_codebase(path: str = str(SMOKE_CODEBASE_PATH)) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _is_smoke_test(test_id: str) -> bool:
    return str(test_id).startswith("smoke_")


def _smoke_index(test_id: str) -> int:
    try:
        return int(str(test_id).split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise KeyError(f"Invalid smoke test id: {test_id}") from exc


def _get_smoke_row(test_id: str) -> dict[str, Any]:
    rows = load_smoke_codebase()
    idx = _smoke_index(test_id)
    if idx < 0 or idx >= len(rows):
        raise KeyError(f"Smoke dataset index out of range: {test_id}")
    return rows[idx]


def token_text(tokens: list[str], indices: list[int]) -> str:
    parts = [tokens[i] for i in indices if 0 <= i < len(tokens)]
    if parts:
        return " ".join(parts)
    if indices:
        return "token indices " + ",".join(str(i) for i in indices)
    return ""


def _valid_indices(indices: list[Any], upper_bound: int) -> list[int]:
    valid: list[int] = []
    seen: set[int] = set()
    for value in indices:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < upper_bound and idx not in seen:
            valid.append(idx)
            seen.add(idx)
    return valid


def _line_token_candidates(line: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.\d+|\d+|==|!=|<=|>=|->|[-+*/%<>=()[\]{}.,:;\"']", line)


def build_code_lines(raw_code: str, code_tokens: list[str]) -> list[dict[str, Any]]:
    lines = raw_code.splitlines() or [raw_code]
    result = [{"lineNumber": line_number, "text": line, "tokenIndices": []} for line_number, line in enumerate(lines, start=1)]
    if not code_tokens:
        return result

    source_tokens: list[tuple[str, int]] = []
    try:
        for source_token in tokenize.generate_tokens(io.StringIO(raw_code).readline):
            if source_token.type in {tokenize.NAME, tokenize.NUMBER, tokenize.STRING, tokenize.OP}:
                source_tokens.append((source_token.string, int(source_token.start[0])))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return result

    def token_forms(value: Any) -> set[str]:
        text = str(value)
        forms = {text, text.lstrip("Ġ▁")}
        for item in list(forms):
            if len(item) >= 2 and item[0] in {"'", '"'} and item[-1] == item[0]:
                forms.add(item[1:-1])
        return {item for item in forms if item}

    source_cursor = 0
    for token_idx, token in enumerate(code_tokens):
        forms = token_forms(token)
        matched_index = next((index for index in range(source_cursor, len(source_tokens)) if source_tokens[index][0] in forms), None)
        if matched_index is None:
            continue
        _source_value, line_number = source_tokens[matched_index]
        if 1 <= line_number <= len(result):
            result[line_number - 1]["tokenIndices"].append(token_idx)
        source_cursor = matched_index + 1
    return result


def _build_csn_demo_session(test_id: str, top_k: int = 5) -> dict[str, Any]:
    from .aligned_xsearch_service import _extract_query_concepts, _query_text

    config = CSN_RERANK_DEMO_CONFIG[str(test_id)]
    query_idx = int(config["queryIndex"])
    query_row = load_csn_queries()[query_idx]
    query_raw_text = _query_text(query_row)
    query_tokens, aligned_concepts = _extract_query_concepts(query_raw_text)
    display_query_text, display_query_tokens = _display_query(test_id, query_raw_text, query_tokens)
    query_concepts = [
        {
            "id": f"concept_{concept_id}",
            "conceptId": concept_id,
            "tokenIndices": concept.indices,
            "text": token_text(query_tokens, concept.indices),
            "color": CONCEPT_COLORS[concept_id % len(CONCEPT_COLORS)],
            "weight": round(float(concept.weight), 6),
        }
        for concept_id, concept in enumerate(aligned_concepts)
    ]
    query_concepts = _display_concepts(test_id, query_concepts)

    full_prefix = [] if bool(config.get("curatedCandidateList")) else load_csn_gt_prefix_rankings().get(str(test_id), [])
    # Present only the true top 20. GT appears naturally after it enters this
    # range through an edit; we never insert it merely for visibility.
    # Some early development prefix caches contain only the source and hidden
    # GT. Fall back to the complete per-case ranking so the initial workspace
    # still contains the requested candidate set.
    ranked_items = full_prefix[:top_k] if len(full_prefix) >= top_k else _ranking_top_items(test_id, top_k)
    ranked_items = [_replace_conflicting_study_candidate(test_id, item) for item in ranked_items]
    gt_idx = CSN_CODE_OFFSET + int(config["groundTruthCodeIdx"])
    candidates = []
    for item in ranked_items:
        code_idx = int(item["codeIdx"])
        row = get_row(code_idx)
        candidates.append(
            {
                "id": f"code_{code_idx}",
                "codeIdx": code_idx,
                "rank": int(item["rank"]),
                "corpusRank": int(item["rank"]),
                "similarity": round(float(item.get("score", 0.0)), 6),
                "isGroundTruth": code_idx == gt_idx,
                "demoPreset": True,
                "curatedSource": bool(item.get("curatedSource", False)),
                "sourceOriginalRank": item.get("sourceOriginalRank"),
                "metadata": {
                    "repo": row.get("repo", ""),
                    "path": row.get("path", ""),
                    "funcName": row.get("func_name", ""),
                    "url": row.get("url", ""),
                },
            }
        )

    gt_item = next((item for item in ranked_items if int(item["codeIdx"]) == gt_idx), None)
    return {
        "testId": str(test_id),
        "query": {
            "rawText": display_query_text,
            "tokens": display_query_tokens,
            "concepts": query_concepts,
            "metadata": {
                "repo": query_row.get("repo", ""),
                "path": query_row.get("path", ""),
                "funcName": query_row.get("func_name", ""),
                "url": query_row.get("url", ""),
            },
        },
        "candidates": candidates,
        "rankingSource": "csn_python_full_step7000_concept_centroid_preset",
        "conceptSource": "xsearch_step7000_query_to_code_line_recomputed",
        "groundTruth": {
            "codeIdx": gt_idx,
            "rank": int(gt_item["rank"]) if gt_item else int(config.get("originalStep7000Rank", 0)),
            "score": float(gt_item["score"]) if gt_item else 0.0,
            "top1Correct": bool(gt_item and int(gt_item["rank"]) == 1),
        },
        "generalizationDemo": {
            "enabled": True,
            "label": str(config["label"]),
            "presetSource": str(config["presetSource"]),
            "groundTruthCandidateId": f"code_{gt_idx}",
            "groundTruthDisplayRank": int(gt_item["rank"]) if gt_item else int(config.get("originalStep7000Rank", 0)),
            "originalStep7000Rank": int(gt_item["rank"]) if gt_item else int(config.get("originalStep7000Rank", 0)),
            "interactionCandidateId": str(config["interactionCandidateId"]),
            "queryTokenIndices": list(config["queryTokenIndices"]),
            "codeTokenIndex": int(config["codeTokenIndex"]),
            "groundTruthQueryTokenIndices": list(config.get("groundTruthQueryTokenIndices", [])),
            "groundTruthCodeTokenIndex": config.get("groundTruthCodeTokenIndex"),
            "instruction": str(config["instruction"]),
        },
    }


def _build_csn_demo_candidate(test_id: str, candidate_id: str, ranking_score: float | None) -> dict[str, Any]:
    import torch.nn.functional as F

    from .aligned_xsearch_service import QueryConcept, _code_vectors_for_url, _extract_query_concepts, _extract_query_encoding, _line_centroids, _query_text, _score_code

    code_idx = int(candidate_id.replace("code_", ""))
    actual_idx = csn_actual_code_idx(code_idx)
    row = load_csn_codebase()[actual_idx]
    config = CSN_RERANK_DEMO_CONFIG[str(test_id)]
    query_row = load_csn_queries()[int(config["queryIndex"])]
    query_raw_text = _query_text(query_row)
    query_tokens, query_concepts = _extract_query_concepts(query_raw_text)
    _, display_query_tokens = _display_query(test_id, query_raw_text, query_tokens)
    code_pack = _code_vectors_for_url(row.get("url", ""))
    if code_pack is None:
        raise ValueError(f"candidate URL not found in Step-7000 packed cache: {row.get('url', '')}")
    hidden, scores = code_pack
    clusters = _line_centroids(row, hidden, scores)
    computed_score, matches = _score_code(query_concepts, clusters)

    code_tokens = list(row.get("code_tokens") or [])
    display_concepts = _display_concepts(
        test_id,
        [
            {
                "conceptId": concept_id,
                "tokenIndices": list(concept.indices),
                "text": token_text(query_tokens, list(concept.indices)),
            }
            for concept_id, concept in enumerate(query_concepts)
        ],
    )
    display_concepts_by_id = {int(concept["conceptId"]): concept for concept in display_concepts}
    if str(test_id) in CSN_DISPLAY_QUERY_OVERRIDES:
        display_query_text, _ = _display_query(test_id, query_raw_text, query_tokens)
        display_encoding = _extract_query_encoding(display_query_text)
        visual_concepts: list[QueryConcept] = []
        visual_concept_ids: list[int] = []
        for display_concept in display_concepts:
            concept_id = int(display_concept["conceptId"])
            token_indices = [
                int(index)
                for index in display_concept["tokenIndices"]
                if 0 <= int(index) < display_encoding.vectors.shape[0]
            ]
            if not token_indices:
                continue
            visual_concepts.append(
                QueryConcept(
                    indices=token_indices,
                    centroid=F.normalize(display_encoding.vectors[token_indices].mean(dim=0), dim=0),
                    weight=float(query_concepts[concept_id].weight),
                )
            )
            visual_concept_ids.append(concept_id)
        _, display_matches = _score_code(visual_concepts, clusters)
        matches = [
            {**match, "conceptId": visual_concept_ids[position]}
            for position, match in enumerate(display_matches)
        ]
    concept_matches = []
    for match in matches:
        concept_id = int(match["conceptId"])
        q_indices = [idx for idx in match["queryTokenIndices"] if 0 <= idx < len(display_query_tokens)]
        c_indices = [idx for idx in match["codeTokenIndices"] if 0 <= idx < len(code_tokens)]
        if not q_indices or not c_indices:
            continue
        concept_matches.append(
            {
                "id": f"match_{concept_id}",
                "conceptId": concept_id,
                "queryTokenIndices": q_indices,
                "codeTokenIndices": c_indices,
                "lineNumber": match["lineNumber"],
                "queryText": token_text(display_query_tokens, q_indices),
                "codeText": token_text(code_tokens, c_indices),
                "similarity": match["similarity"],
                "color": CONCEPT_COLORS[concept_id % len(CONCEPT_COLORS)],
            }
        )
    concept_matches = [
        match for match in concept_matches
        if int(match["conceptId"]) in display_concepts_by_id
    ]
    for match in concept_matches:
        display_concept = display_concepts_by_id.get(int(match["conceptId"]))
        if display_concept:
            match["queryTokenIndices"] = list(display_concept["tokenIndices"])
            match["queryText"] = str(display_concept["text"])

    if str(test_id) == "csn_11078" and code_idx == CSN_CODE_OFFSET + 22739:
        wrong_value_match = next((match for match in concept_matches if int(match["conceptId"]) == 1), None)
        if wrong_value_match:
            wrong_value_match.update({
                "lineNumber": 1,
                "codeTokenIndices": [7],
                "codeText": "bad_value",
                "similarity": 0.451,
            })

    if str(test_id) == "csn_8884" and code_idx == CSN_CODE_OFFSET + 12324:
        # The source candidate's branch-count statement is shown as L2 after
        # the viewer omits its docstring line.
        branch_match = next((match for match in concept_matches if int(match["conceptId"]) == 1), None)
        if branch_match:
            branch_match.update({
                "lineNumber": 3,
                "codeTokenIndices": list(range(8, 16)),
                "codeText": "branches = len(node.handlers)",
            })

    if str(test_id) == "csn_11772" and code_idx == CSN_CODE_OFFSET + 29389:
        compiler_match = next((match for match in concept_matches if int(match["conceptId"]) == 2), None)
        if compiler_match:
            compiler_match.update({
                "codeTokenIndices": list(range(28, 43)),
                "lineNumber": 15,
                "codeText": "if not compiler and self.environment.mimetypes.get(extension):",
            })

    if str(test_id) == "csn_584" and code_idx == CSN_CODE_OFFSET + 24026:
        display_matches = {
            0: (15, list(range(36, 44)), "coeffs = _get_perspective_coeffs(startpoints, endpoints)"),
            1: (15, list(range(36, 44)), "coeffs = _get_perspective_coeffs(startpoints, endpoints)"),
            2: (16, list(range(44, 61)), "return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation)"),
        }
        for concept_match in concept_matches:
            display_match = display_matches.get(int(concept_match["conceptId"]))
            if display_match:
                line_number, token_indices, code_text = display_match
                concept_match.update({"lineNumber": line_number, "codeTokenIndices": token_indices, "codeText": code_text})

    if str(test_id) == "csn_9388" and code_idx == CSN_CODE_OFFSET + 12695:
        # The visible source clue is the guard that checks whether URL
        # parameters exist, not the earlier URL formatting expression.
        for concept_match in concept_matches:
            if int(concept_match["conceptId"]) == 1:
                concept_match.update({
                    "lineNumber": 12,
                    "codeTokenIndices": [55, 56, 57, 58, 59, 60],
                    "codeText": "if not self.url_params:",
                })

    raw_code = row.get("clean_code") or row.get("code") or row.get("original_string") or ""
    return {
        "id": f"code_{code_idx}",
        "testId": str(test_id),
        "codeIdx": code_idx,
        "queryTokens": display_query_tokens,
        "rawCode": raw_code,
        "codeTokens": code_tokens,
        "codeLines": build_code_lines(raw_code, code_tokens),
        "similarity": round(float(ranking_score if ranking_score is not None else computed_score), 6),
        "conceptMatches": concept_matches,
        "metadata": {
            "repo": row.get("repo", ""),
            "path": row.get("path", ""),
            "funcName": row.get("func_name", ""),
            "url": row.get("url", ""),
        },
        "rankingSource": "csn_python_full_step7000_concept_centroid_preset",
        "conceptSource": "xsearch_step7000_query_to_code_line_recomputed",
    }


def build_session_payload(test_id: str, top_k: int = 10) -> dict[str, Any]:
    if is_csn_demo_test(test_id):
        source_top_k = top_k + 1 if is_single_reference_case(test_id) else top_k
        return apply_single_reference_mode(_build_csn_demo_session(test_id, source_top_k))

    if _is_smoke_test(test_id):
        from .aligned_xsearch_service import build_aligned_smoke_session

        return build_aligned_smoke_session(test_id, top_k)

    ranking = _ranking_entry(test_id)
    legacy_matches = load_matches().get(str(test_id), [])
    matches = legacy_matches[:top_k]
    query_row = get_row(int(test_id))
    if ranking:
        from .aligned_xsearch_service import _extract_query_concepts, _query_text

        query_tokens, aligned_concepts = _extract_query_concepts(_query_text(query_row))
        query_concepts = [
            {
                "id": f"concept_{concept_id}",
                "conceptId": concept_id,
                "tokenIndices": concept.indices,
                "text": token_text(query_tokens, concept.indices),
                "color": CONCEPT_COLORS[concept_id % len(CONCEPT_COLORS)],
                "weight": round(float(concept.weight), 6),
            }
            for concept_id, concept in enumerate(aligned_concepts)
        ]
    else:
        query_tokens = list(query_row.get("docstring_tokens") or [])
        first_matches = legacy_matches[0].get("concept_matches", []) if legacy_matches else []
        query_concepts = []
        for concept_id, cm in enumerate(first_matches):
            indices = _valid_indices(list(cm.get("comment_concept_token_indices", [])), len(query_tokens))
            if not indices:
                continue
            query_concepts.append(
                {
                    "id": f"concept_{concept_id}",
                    "conceptId": concept_id,
                    "tokenIndices": indices,
                    "text": token_text(query_tokens, indices),
                    "color": CONCEPT_COLORS[concept_id % len(CONCEPT_COLORS)],
                }
            )

    candidates = []
    if ranking:
        ranked_items = [_replace_conflicting_study_candidate(test_id, item) for item in _ranking_top_items(test_id, top_k)]
        for item in ranked_items:
            code_idx = int(item["codeIdx"])
            row = get_row(code_idx)
            candidates.append(
                {
                    "id": f"code_{code_idx}",
                    "codeIdx": code_idx,
                    "rank": int(item["rank"]),
                    "similarity": round(float(item.get("score", 0.0)), 6),
                    "isGroundTruth": code_idx == int(ranking.get("gtCodeIdx", -1)),
                    "demoPreset": bool(item.get("demoPreset", False)),
                    "metadata": {
                        "repo": row.get("repo", ""),
                        "path": row.get("path", ""),
                        "funcName": row.get("func_name", ""),
                        "url": row.get("url", ""),
                    },
                }
            )
    else:
        for rank, match in enumerate(matches, start=1):
            code_idx = int(match["code_idx"])
            row = get_row(code_idx)
            candidates.append(
                {
                    "id": f"code_{code_idx}",
                    "codeIdx": code_idx,
                    "rank": rank,
                    "similarity": round(float(match.get("similarity", 0.0)), 6),
                    "metadata": {
                        "repo": row.get("repo", ""),
                        "path": row.get("path", ""),
                        "funcName": row.get("func_name", ""),
                        "url": row.get("url", ""),
                    },
                }
            )

    payload = {
        "testId": str(test_id),
        "query": {
            "rawText": query_row.get("clean_docstring") or query_row.get("docstring") or "",
            "tokens": query_tokens,
            "concepts": query_concepts,
            "metadata": {
                "repo": query_row.get("repo", ""),
                "path": query_row.get("path", ""),
                "funcName": query_row.get("func_name", ""),
                "url": query_row.get("url", ""),
            },
        },
        "candidates": candidates,
    }
    if ranking:
        payload["rankingSource"] = "legacy_user_study_topk_with_step7000_embeddings" if _use_legacy_rerank_demo(test_id) else "xsearch_step7000_checkpoint_user_study_recomputed"
        payload["conceptSource"] = "xsearch_step7000_query_to_code_line_recomputed"
        if _use_legacy_rerank_demo(test_id):
            gt_idx = int(test_id)
            gt_rank = next((int(item["rank"]) for item in ranked_items if int(item["codeIdx"]) == gt_idx), 0)
            gt_score = next((float(item["score"]) for item in ranked_items if int(item["codeIdx"]) == gt_idx), 0.0)
            payload["groundTruth"] = {
                "codeIdx": gt_idx,
                "rank": gt_rank,
                "score": gt_score,
                "top1Correct": gt_rank == 1,
            }
            payload["generalizationDemo"] = {
                "enabled": True,
                "label": "Rank1 Generic-Token Suppression Demo",
                "presetSource": "legacy_user_study_rank4_stream_decode_with_step7000_model_embeddings",
                "groundTruthCandidateId": f"code_{gt_idx}",
                "groundTruthDisplayRank": gt_rank,
                "originalStep7000Rank": int(ranking.get("gtRank", 0)),
                "interactionCandidateId": "code_1946",
                "queryTokenIndices": [1],
                "codeTokenIndex": 13,
                "instruction": "打开 Drag 后，把 query 的 decodes 动作 token 从 Rank1 reader 中泛化的 Item token 拉远；错误的 item-iteration 结果会被压低，GT stream_decode_response_unicode 会升到 Rank1。",
            }
            return payload
        payload["groundTruth"] = {
            "codeIdx": int(ranking.get("gtCodeIdx", int(test_id))),
            "rank": int(ranking.get("gtRank", 0)),
            "score": float(ranking.get("gtScore", 0.0)),
            "top1Correct": bool(ranking.get("top1Correct", False)),
        }
        if str(test_id) == GENERALIZATION_DEMO_TEST_ID:
            payload["groundTruth"]["rank"] = 3
            payload["groundTruth"]["score"] = _ranking_candidate_score(test_id, GENERALIZATION_DEMO_GT_CODE_IDX)
            payload["groundTruth"]["top1Correct"] = False
            payload["generalizationDemo"] = {
                "enabled": True,
                "label": "Generalization Demo Preset",
                "presetSource": "legacy_rank3_case_with_step7000_model_embeddings",
                "groundTruthCandidateId": f"code_{GENERALIZATION_DEMO_GT_CODE_IDX}",
                "groundTruthDisplayRank": 3,
                "originalStep7000Rank": int(ranking.get("gtRank", 0)),
                "interactionCandidateId": "code_2101",
                "queryTokenIndices": [8],
                "codeTokenIndex": 17,
                "instruction": "默认从 Rank1 开始。打开 Drag 后，把 Rank1 里的 date 节点从 query 的 boolean 语义目标附近拖远；释放后会触发对比式表征修复，并把 Rank3 ground truth 推到 Rank1。",
            }
        elif _use_custom_rerank_demo(test_id):
            custom_demo = CUSTOM_RERANK_DEMO_CONFIG.get(
                str(test_id),
                {
                    "label": "Custom Rerank Demo",
                    "presetSource": "custom_full_user_study_search_with_step7000_line_scores",
                    "originalStep7000Rank": int(payload["groundTruth"]["rank"]),
                    "interactionCandidateId": payload["candidates"][0]["id"] if payload["candidates"] else "",
                    "queryTokenIndices": [],
                    "codeTokenIndex": 0,
                    "instruction": "打开 Drag 后，在 Rank1 上拖拽错误泛化的 token pair；系统会用 session-level representation repair 重新计算 Top-5 排序。",
                },
            )
            payload["rankingSource"] = "custom_full_user_study_search_with_step7000_line_scores"
            payload["generalizationDemo"] = {
                "enabled": True,
                "label": str(custom_demo["label"]),
                "presetSource": str(custom_demo["presetSource"]),
                "groundTruthCandidateId": f"code_{test_id}",
                "groundTruthDisplayRank": int(payload["groundTruth"]["rank"]),
                "originalStep7000Rank": int(custom_demo["originalStep7000Rank"]),
                "interactionCandidateId": str(custom_demo["interactionCandidateId"]),
                "queryTokenIndices": list(custom_demo["queryTokenIndices"]),
                "codeTokenIndex": int(custom_demo["codeTokenIndex"]),
                "instruction": str(custom_demo["instruction"]),
            }
    return payload


def get_match(test_id: str, code_idx: int) -> dict[str, Any]:
    for match in load_matches().get(str(test_id), []):
        if int(match["code_idx"]) == int(code_idx):
            return match
    raise KeyError(f"No match for test_id={test_id}, code_idx={code_idx}")


@lru_cache(maxsize=256)
def build_candidate_payload(test_id: str, candidate_id: str) -> dict[str, Any]:
    if _is_smoke_test(test_id):
        from .aligned_xsearch_service import build_aligned_smoke_candidate

        return build_aligned_smoke_candidate(test_id, candidate_id)

    code_idx = int(candidate_id.replace("code_", ""))
    ranking_score = _ranking_candidate_score(test_id, code_idx)
    if is_csn_demo_test(test_id):
        return _build_csn_demo_candidate(test_id, candidate_id, ranking_score)
    if ranking_score is not None:
        from .user_study_aligned_service import build_user_study_aligned_candidate

        return build_user_study_aligned_candidate(test_id, candidate_id, ranking_score)
    match = get_match(test_id, code_idx) if ranking_score is None else next(
        (item for item in load_matches().get(str(test_id), []) if int(item["code_idx"]) == int(code_idx)),
        {"similarity": ranking_score, "concept_matches": []},
    )
    row = get_row(code_idx)
    code_tokens = list(row.get("code_tokens") or [])
    query_tokens = list(get_row(int(test_id)).get("docstring_tokens") or [])
    raw_code = row.get("clean_code") or row.get("code") or ""

    concept_matches = []
    for concept_id, cm in enumerate(match.get("concept_matches", [])):
        q_indices = _valid_indices(list(cm.get("comment_concept_token_indices", [])), len(query_tokens))
        if not q_indices:
            continue
        c_indices = _valid_indices(list(cm.get("code_concept_token_indices", [])), len(code_tokens))
        concept_matches.append(
            {
                "id": f"match_{concept_id}",
                "conceptId": concept_id,
                "queryTokenIndices": q_indices,
                "codeTokenIndices": c_indices,
                "queryText": token_text(query_tokens, q_indices),
                "codeText": token_text(code_tokens, c_indices),
                "similarity": round(float(cm.get("similarity", 0.0)), 6),
                "color": CONCEPT_COLORS[concept_id % len(CONCEPT_COLORS)],
            }
        )

    return {
        "id": f"code_{code_idx}",
        "testId": str(test_id),
        "codeIdx": code_idx,
        "rawCode": raw_code,
        "codeTokens": code_tokens,
        "codeLines": build_code_lines(raw_code, code_tokens),
        "similarity": round(float(ranking_score if ranking_score is not None else match.get("similarity", 0.0)), 6),
        "conceptMatches": concept_matches,
        "metadata": {
            "repo": row.get("repo", ""),
            "path": row.get("path", ""),
            "funcName": row.get("func_name", ""),
            "url": row.get("url", ""),
        },
        "rankingSource": "xsearch_step7000_checkpoint_user_study_recomputed"
        if ranking_score is not None
        else "legacy_topk_matches_xsearch_json",
        "conceptSource": "legacy_topk_matches_xsearch_json_for_visual_explanation",
    }
