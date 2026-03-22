#!/usr/bin/env python3
"""用 LLM 整理 Markdown 标题结构（仅改层级与顺序，不改正文），结果写入新文件。"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from loguru import logger
from openai import OpenAI

# 默认输出后缀（xxx.md -> xxx.reorganized.md）
DEFAULT_OUT_SUFFIX = ""
# 默认递归目录
DEFAULT_DIR = "_all_md_renamed"
# 默认允许的最大输出 token
DEFAULT_MAX_TOKENS = 20480
MAX_LLM_RETRIES = 3
DEFAULT_JOBS = 10
# 超过该字符数（len(raw)）则跳过，不调用 LLM；若正文中含 CHUNK_SPLIT_MARKER 则可按该标识切分后分段调用 LLM
DEFAULT_MAX_CHARS = 15000
# 超长文档人工/预处理插入的切分标识（整段作为分隔符，不出现在输出中）
CHUNK_SPLIT_MARKER = "0000-----0000"
DEFAULT_EVENT_LOG = Path("reorganize_events.log")
# 一级标题若以「数字.数字」开头（如 2.1、3.2.1），明显应为子节，不得为 #
SUSPICIOUS_H1_TITLE_RE = re.compile(r"^\d+\.\d+")
# 标题行正则：以 # 开头且后跟空格
HEADING_LINE_RE = re.compile(r"^#+\s")
HEADING_PREFIX_RE = re.compile(r"^#+\s*")

ENV_PATH = Path(__file__).with_name(".env")
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


class ProcessOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    SKIP_EXISTING = "SKIP_EXISTING"
    SKIP_TOO_LONG = "SKIP_TOO_LONG"
    ERROR_READ = "ERROR_READ"
    ERROR_LLM = "ERROR_LLM"
    ERROR_WRITE = "ERROR_WRITE"


@dataclass
class ProcessResult:
    path: Path
    out_path: Path
    outcome: ProcessOutcome
    detail: str = ""
    char_count: int | None = None


# 事件日志：标签 + ANSI 颜色（便于 grep 与终端 cat）
_EVENT_OUTCOME_ANSI: dict[ProcessOutcome, str] = {
    ProcessOutcome.SKIP_EXISTING: "\x1b[36m",  # cyan
    ProcessOutcome.SKIP_TOO_LONG: "\x1b[33m",  # yellow
    ProcessOutcome.ERROR_READ: "\x1b[31m",  # red
    ProcessOutcome.ERROR_LLM: "\x1b[31m",
    ProcessOutcome.ERROR_WRITE: "\x1b[31m",
    ProcessOutcome.SUCCESS: "\x1b[32m",  # green（汇总用）
}
_ANSI_RESET = "\x1b[0m"


def _one_line(s: str) -> str:
    return " ".join(s.split())


def format_event_line(res: ProcessResult) -> str:
    """单行事件：带 [TAG] + 颜色码（便于在文件中 grep TAG）。"""
    color = _EVENT_OUTCOME_ANSI.get(res.outcome, "")
    tag = f"[{res.outcome.value}]"
    head = f"{color}{tag}{_ANSI_RESET}"
    parts = [f"src={res.path}", f"out={res.out_path}"]
    if res.char_count is not None:
        parts.append(f"chars={res.char_count}")
    if res.detail:
        parts.append(f"detail={_one_line(res.detail)}")
    return f"{head} " + " ".join(parts)


def write_event_log_file(log_path: Path, results: list[ProcessResult]) -> None:
    """写入跳过与错误事件；成功不落行，避免刷屏。末尾写一行汇总。"""
    events = [r for r in results if r.outcome != ProcessOutcome.SUCCESS]
    lines_to_write = [
        format_event_line(r) for r in sorted(events, key=lambda x: str(x.path))
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    counts: Counter[ProcessOutcome] = Counter(r.outcome for r in results)
    summary = (
        f"{_EVENT_OUTCOME_ANSI[ProcessOutcome.SUCCESS]}[SUMMARY]{_ANSI_RESET} "
        f"time={ts} total={len(results)} "
        f"SUCCESS={counts[ProcessOutcome.SUCCESS]} "
        f"SKIP_EXISTING={counts[ProcessOutcome.SKIP_EXISTING]} "
        f"SKIP_TOO_LONG={counts[ProcessOutcome.SKIP_TOO_LONG]} "
        f"ERROR_READ={counts[ProcessOutcome.ERROR_READ]} "
        f"ERROR_LLM={counts[ProcessOutcome.ERROR_LLM]} "
        f"ERROR_WRITE={counts[ProcessOutcome.ERROR_WRITE]}"
    )

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n--- batch {ts} ---\n")
        for line in lines_to_write:
            f.write(line + "\n")
        f.write(summary + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 LLM 整理 Markdown 标题结构，仅改层级与顺序，不改正文，输出到新文件。"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"递归处理目录下所有 .md（默认: {DEFAULT_DIR}）",
    )
    group.add_argument(
        "--file",
        type=Path,
        default=None,
        metavar="PATH",
        help="仅处理单个 .md 文件（不检测输出是否已存在、不检测源长度，直接调用 LLM 并写入）",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        metavar="N",
        help="当按目录批量处理时，仅处理按文件名（相对路径字符串）排序后的第 N 个及之后的文件（从 1 开始计数）",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        metavar="N",
        help="当按目录批量处理时，仅处理按文件名（相对路径字符串）排序后的第 N 个及之前的文件（从 1 开始计数）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help=f"LLM 允许的最大输出 token（默认: {DEFAULT_MAX_TOKENS}）",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help=f"输出文件名后缀，如 .reorganized（默认: {DEFAULT_OUT_SUFFIX}）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default="./reorganized",
        metavar="PATH",
        help="所有输出写入该目录；不指定则与源文件同目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将要处理的文件，不调用 LLM、不写文件",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标输出文件已存在则跳过，不再调用 LLM（默认关闭）",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        metavar="N",
        help=(
            f"源文件字符数 len(raw) 超过 N 则跳过（不调用 LLM），"
            f"除非正文中含切分标识 {CHUNK_SPLIT_MARKER!r} 且每段长度均 ≤N；"
            f"-1 表示不限制（默认: {DEFAULT_MAX_CHARS}）"
        ),
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        default=DEFAULT_EVENT_LOG,
        metavar="PATH",
        help=f"将跳过与错误事件追加写入该日志（默认: {DEFAULT_EVENT_LOG}）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_LLM_TIMEOUT,
        metavar="SEC",
        help=f"单文件 LLM 请求超时秒数（默认: {DEFAULT_LLM_TIMEOUT}）",
    )
    parser.add_argument(
        "--model",
        "--dashscope-model",
        "--ollama-model",
        dest="model",
        type=str,
        default=None,
        metavar="NAME",
        help="模型名（默认用环境变量 DASHSCOPE_MODEL/OPENAI_MODEL，兜底 qwen3.5-32b-instruct）",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=DEFAULT_JOBS,
        metavar="N",
        help=f"批量处理时的并行线程数（默认 {DEFAULT_JOBS}）；遇 API 限流可调低",
    )
    parser.add_argument(
        "--no-preprocess-h1",
        action="store_true",
        help="关闭对「# 2.1」类标题的硬性预降级为 ##（输出后仍会审核并可重试）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_LLM_RETRIES,
        metavar="N",
        help=f"单文件最多调用 LLM 次数（含首次，默认 {MAX_LLM_RETRIES}）",
    )
    parser.add_argument(
        "--strict-headings",
        action="store_true",
        help="启用标题 multiset 严格校验（输出中每条标题文字及出现次数须与输入一致；默认关闭）",
    )
    parser.add_argument(
        "--heading-plan",
        action="store_true",
        help=(
            "标题计划模式：仅将标题骨架（id + 文本 + 原层级）送 LLM，"
            "返回带 op 的 JSON 计划后本地重建全文；支持 block 与 merge。"
            "与全文整理互斥，且不受全文 max_chars 超长跳过影响。"
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers / -j 必须 >= 1")
    if args.max_retries < 1:
        parser.error("--max-retries 必须 >= 1")
    if args.max_chars < -1:
        parser.error("--max-chars 必须为 >= -1（-1 表示不限制）")
    # 未指定 --dir 且未指定 --file 时，使用默认 --dir output/
    if args.dir is None and args.file is None:
        args.dir = Path(DEFAULT_DIR)
    return args


def collect_md_files(args: argparse.Namespace) -> list[Path]:
    """根据 args 收集待处理的 .md 路径列表。"""
    if args.file is not None:
        p = args.file.resolve()
        if not p.exists():
            logger.error("文件不存在: {}", p)
            sys.exit(1)
        if p.suffix.lower() != ".md":
            logger.error("请指定 .md 文件: {}", p)
            sys.exit(1)
        return [p]

    dir_path = (args.dir or Path(DEFAULT_DIR)).resolve()
    if not dir_path.is_dir():
        logger.error("目录不存在: {}", dir_path)
        sys.exit(1)

    out_suffix = args.out_suffix or DEFAULT_OUT_SUFFIX
    # 仅当指定了输出后缀时排除「已整理」产物，避免 out_suffix 为空时 exclude_suffix 变成 ".md" 误伤全部文件
    exclude_suffix = (out_suffix + ".md") if out_suffix else ""

    files: list[Path] = []
    for f in dir_path.rglob("*.md"):
        if not f.is_file():
            continue
        name = f.name
        if out_suffix and (
            name.endswith(exclude_suffix)
            or (out_suffix + "." in name and name.endswith(".md"))
        ):
            continue
        files.append(f)

    def _rel_posix(p: Path) -> str:
        # 相对路径用于稳定跨机器/跨前缀排序
        return p.relative_to(dir_path).as_posix()

    # 按“名字”排序（相对路径字典序），并将《食品安全国家标准 食品添加剂...》优先。
    # 这样可避免数字开头目录（如“1 牛奶...”）抢在前面，满足你按文件夹观感的起始顺序。
    def _sort_key(p: Path) -> tuple[int, str]:
        rel = _rel_posix(p)
        priority = 0 if rel.startswith("《食品安全国家标准 食品添加剂") else 1
        return (priority, rel)

    files.sort(key=_sort_key)
    return files


def heading_texts(text: str) -> list[str]:
    """提取围栏外标题文本（去掉 # 前缀），用于校验标题是否被增删/改写。"""
    out: list[str] = []
    in_fence = False
    for ln in text.split("\n"):
        st = ln.strip()
        if st.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if HEADING_LINE_RE.match(ln):
            out.append(re.sub(r"^#+\s*", "", ln).strip())
    return out


def headings_unchanged(original: str, llm_output: str) -> bool:
    """校验标题文本集合是否一致（允许调层级和块顺序，不允许增删改标题文本）。"""
    return Counter(heading_texts(original)) == Counter(heading_texts(llm_output))


def headings_mismatch_report(original: str, llm_output: str) -> str:
    """标题 multiset 不一致时生成可读说明（缺失 / 多出 / 计数）。"""
    co = Counter(heading_texts(original))
    cm = Counter(heading_texts(llm_output))
    so, sm = sum(co.values()), sum(cm.values())
    lines: list[str] = [
        f"基准标题条数: {so}，模型输出标题条数: {sm}",
    ]
    missing: list[tuple[str, int]] = []
    extra: list[tuple[str, int]] = []
    for t, n in co.items():
        d = n - cm.get(t, 0)
        if d > 0:
            missing.append((t, d))
    for t, n in cm.items():
        d = n - co.get(t, 0)
        if d > 0:
            extra.append((t, d))
    if missing:
        lines.append("输出里缺失或少于基准的次数：")
        for t, d in sorted(missing, key=lambda x: x[0]):
            lines.append(f"  - ×{d}  {t!r}")
    if extra:
        lines.append("输出里多出或重复的次数：")
        for t, d in sorted(extra, key=lambda x: x[0]):
            lines.append(f"  - ×{d}  {t!r}")
    return "\n".join(lines)


def _heading_level(line: str) -> int | None:
    """若为 Markdown 标题行（# 后接空格），返回 # 个数，否则 None。"""
    if not HEADING_LINE_RE.match(line):
        return None
    return len(line) - len(line.lstrip("#"))


def count_h1_and_suspicious(text: str) -> tuple[int, int]:
    """返回 (一级标题总数, 其中「多级编号开头」的可疑一级标题数)。"""
    total_h1 = 0
    suspicious = 0
    for ln in text.split("\n"):
        level = _heading_level(ln)
        if level != 1:
            continue
        total_h1 += 1
        title = HEADING_PREFIX_RE.sub("", ln).strip()
        if SUSPICIOUS_H1_TITLE_RE.match(title):
            suspicious += 1
    return total_h1, suspicious


def suspicious_h1_audit_passes(text: str) -> bool:
    """
    硬性审核：若存在过多「不应为一级标题」的 # 行，返回 False（应重试）。
    规则：可疑数 >= 2，或 可疑数/一级标题总数 >= 0.3。
    """
    total_h1, suspicious = count_h1_and_suspicious(text)
    if suspicious == 0:
        return True
    if suspicious >= 2:
        return False
    if suspicious / max(total_h1, 1) >= 0.3:
        return False
    return True


def preprocess_demote_obvious_subsection_h1(text: str) -> str:
    """
    将「一级标题 + 标题文本以 数字.数字 开头」硬性降为 ##，不改标题文字与其余行。
    """
    lines = text.split("\n")
    out: list[str] = []
    for ln in lines:
        level = _heading_level(ln)
        if level == 1:
            title = HEADING_PREFIX_RE.sub("", ln).strip()
            if SUSPICIOUS_H1_TITLE_RE.match(title):
                out.append("## " + title)
                continue
        out.append(ln)
    return "\n".join(out)


def split_by_chunk_marker(working: str, max_chars: int) -> list[str] | None:
    """
    当全文超过 max_chars 时，按 CHUNK_SPLIT_MARKER 切成多段。
    返回非空片段列表；无法使用（无标识、仅一段有效内容、任一段仍超长）时返回 None。
    """
    if max_chars < 0:
        return None
    if CHUNK_SPLIT_MARKER not in working:
        return None
    parts = [p.strip("\n") for p in working.split(CHUNK_SPLIT_MARKER)]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    for p in parts:
        if len(p) > max_chars:
            return None
    return parts


def split_heading_blocks(text: str) -> tuple[list[str], list[dict[str, object]]]:
    """把文档切分为：前导非标题行 + 若干标题块（标题行 + 到下一个标题前的行）。"""
    lines = text.split("\n")
    heading_indices: list[int] = [
        i for i, ln in enumerate(lines) if HEADING_LINE_RE.match(ln)
    ]
    if not heading_indices:
        return lines, []

    prefix = lines[: heading_indices[0]]
    blocks: list[dict[str, object]] = []
    for idx, start in enumerate(heading_indices):
        end = heading_indices[idx + 1] if idx + 1 < len(heading_indices) else len(lines)
        block_lines = lines[start:end]
        heading_line = block_lines[0]
        heading_text = HEADING_PREFIX_RE.sub("", heading_line).strip()
        blocks.append(
            {
                "id": f"H{idx + 1:04d}",
                "heading_text": heading_text,
                "orig_level": len(heading_line) - len(heading_line.lstrip("#")),
                "body_lines": block_lines[1:],
            }
        )
    return prefix, blocks


def _extract_json_array(text: str) -> str | None:
    """从模型输出中提取 JSON 数组字符串（兼容前后有解释文字/代码围栏）。"""
    t = unwrap_fenced_code(text).strip()
    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    return t[start : end + 1]


def build_by_heading_plan(original: str, plan_output: str) -> str | None:
    """
    根据模型返回的标题计划 JSON 重建文档。
    计划格式：[{\"id\":\"H0001\", \"level\":1}, ...]
    """
    prefix_lines, blocks = split_heading_blocks(original)
    if not blocks:
        return original

    raw_json = _extract_json_array(plan_output)
    if raw_json is None:
        return None
    try:
        plan = json.loads(raw_json)
    except Exception:
        return None
    if not isinstance(plan, list):
        return None

    block_map = {str(b["id"]): b for b in blocks}
    expected_ids = set(block_map.keys())
    seen_ids: list[str] = []
    normalized_plan: list[tuple[str, int]] = []

    for item in plan:
        if not isinstance(item, dict):
            return None
        item_id = str(item.get("id", "")).strip()
        level = item.get("level")
        if item_id not in expected_ids:
            return None
        if not isinstance(level, int) or level < 1 or level > 6:
            return None
        seen_ids.append(item_id)
        normalized_plan.append((item_id, level))

    if len(seen_ids) != len(set(seen_ids)):
        return None

    # 允许模型只返回部分 id：将缺失项按原顺序补齐，层级沿用原文层级。
    plan_ids = set(seen_ids)
    if not plan_ids.issubset(expected_ids):
        return None
    if plan_ids != expected_ids:
        missing_count = len(expected_ids - plan_ids)
        logger.warning("计划缺失 {} 个标题块，已按原顺序自动补齐", missing_count)
        for b in blocks:
            bid = str(b["id"])
            if bid in plan_ids:
                continue
            orig_level = int(b["orig_level"])  # type: ignore[index]
            normalized_plan.append((bid, orig_level))

    out_lines = list(prefix_lines)
    for item_id, level in normalized_plan:
        b = block_map[item_id]
        heading_text = str(b["heading_text"])
        body_lines = list(b["body_lines"])
        out_lines.append(("#" * level) + " " + heading_text)
        out_lines.extend(body_lines)

    return "\n".join(out_lines)


def non_heading_line_counter(text: str) -> Counter[str]:
    """
    围栏内所有行 + 围栏外非标题行 的多重集，用于标题计划模式校验正文未被改写。
    """
    c: Counter[str] = Counter()
    in_fence = False
    for ln in text.split("\n"):
        st = ln.strip()
        if st.startswith("```"):
            in_fence = not in_fence
            c[ln] += 1
            continue
        if in_fence:
            c[ln] += 1
            continue
        if HEADING_LINE_RE.match(ln):
            continue
        c[ln] += 1
    return c


def _merge_heading_texts_from_blocks(
    blocks_by_id: dict[str, dict[str, object]], ids_in_order: list[str]
) -> str:
    """合并标题：各块 heading_text 按顺序直接衔接（与全文模式「相邻合并」一致）。"""
    parts: list[str] = []
    for bid in ids_in_order:
        parts.append(str(blocks_by_id[bid]["heading_text"]))
    return "".join(parts)


def _normalize_heading_plan_line(line: str) -> str:
    """去掉列表符等前缀，得到 block/merge 起始行。"""
    s = line.strip()
    for prefix in ("- ", "* ", "• "):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
            break
    return s


def build_heading_skeleton_markdown(blocks: list[dict[str, object]]) -> str:
    """将标题块列表格式化为 Markdown，供标题计划模式送入 LLM（非 JSON）。"""
    lines: list[str] = [
        "## 标题块列表（按原文顺序）",
        "",
        "下列仅含占位 id、原层级与标题文字；正文未包含，请勿编造。",
        "",
    ]
    for b in blocks:
        bid = str(b["id"])
        ol = int(b["orig_level"])  # type: ignore[arg-type]
        ht = str(b["heading_text"])
        lines.append(f"### `{bid}` · 原层级 {ol}")
        lines.append("")
        lines.append(f"**标题文字**：{ht}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_heading_plan_markdown(plan_output: str) -> list[dict[str, object]] | None:
    """
    解析 LLM 返回的标题计划（纯文本行，非 JSON）：
      block <id> <level>
      merge <id1> <id2> ... <idN> <level>   （最后一个 token 为层级）
    允许空行；以 # 开头的行视为注释并跳过。
    """
    t = prepare_model_markdown(plan_output)
    if not t.strip():
        return None
    ops: list[dict[str, object]] = []
    for raw_line in t.split("\n"):
        line = _normalize_heading_plan_line(raw_line)
        if not line:
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        cmd = parts[0].lower()
        if cmd == "block":
            if len(parts) != 3:
                return None
            try:
                level = int(parts[2])
            except ValueError:
                return None
            if not 1 <= level <= 6:
                return None
            ops.append({"op": "block", "id": parts[1], "level": level})
        elif cmd == "merge":
            if len(parts) < 4:
                return None
            try:
                level = int(parts[-1])
            except ValueError:
                return None
            if not 1 <= level <= 6:
                return None
            ids = parts[1:-1]
            if len(ids) < 2:
                return None
            ops.append({"op": "merge", "ids": ids, "level": level})
        else:
            return None
    return ops if ops else None


def build_by_heading_plan_ops(original: str, plan_output: str) -> str | None:
    """
    根据扩展计划（Markdown 纯文本行）重建文档。
    计划行格式：
      block H0001 2
      merge H0001 H0002 1
    """
    prefix_lines, blocks = split_heading_blocks(original)
    if not blocks:
        return original

    plan = parse_heading_plan_markdown(plan_output)
    if plan is None:
        return None

    block_map = {str(b["id"]): b for b in blocks}
    id_to_idx = {str(b["id"]): i for i, b in enumerate(blocks)}
    expected_ids = set(block_map.keys())
    covered: set[str] = set()

    out_lines = list(prefix_lines)

    for item in plan:
        if not isinstance(item, dict):
            return None
        op_raw = item.get("op")
        op = str(op_raw).strip().lower() if op_raw is not None else ""
        if not op and "id" in item and "level" in item:
            op = "block"
        if op == "block":
            item_id = str(item.get("id", "")).strip()
            level = item.get("level")
            if item_id not in expected_ids or item_id in covered:
                return None
            if not isinstance(level, int) or level < 1 or level > 6:
                return None
            covered.add(item_id)
            b = block_map[item_id]
            heading_text = str(b["heading_text"])
            body_lines = list(b["body_lines"])  # type: ignore[assignment]
            out_lines.append(("#" * level) + " " + heading_text)
            out_lines.extend(body_lines)
        elif op == "merge":
            ids = item.get("ids")
            level = item.get("level")
            if not isinstance(ids, list) or len(ids) < 2:
                return None
            if not isinstance(level, int) or level < 1 or level > 6:
                return None
            str_ids = [str(x).strip() for x in ids]
            if len(set(str_ids)) != len(str_ids):
                return None
            for x in str_ids:
                if x not in expected_ids or x in covered:
                    return None
            idxs = sorted(id_to_idx[x] for x in str_ids)
            if idxs != list(range(idxs[0], idxs[-1] + 1)):
                return None
            ids_sorted = sorted(str_ids, key=lambda z: id_to_idx[z])
            covered.update(str_ids)
            merged_title = _merge_heading_texts_from_blocks(block_map, ids_sorted)
            out_lines.append(("#" * level) + " " + merged_title)
            for bid in ids_sorted:
                out_lines.extend(list(block_map[bid]["body_lines"]))  # type: ignore[arg-type]
        else:
            return None

    if covered != expected_ids:
        return None

    return "\n".join(out_lines)


def unwrap_fenced_code(text: str) -> str:
    """若被 ``` 或 ```md 等代码块包裹，去掉首尾围栏行后返回；否则原样返回。"""
    t = text.strip()
    if not t.startswith("```"):
        return text
    lines = t.split("\n")
    # 去掉首行 ```[language]?
    start = 1
    if lines[-1].strip() == "```":
        end = len(lines) - 1
    else:
        end = len(lines)
    return "\n".join(lines[start:end])


def prepare_model_markdown(text: str) -> str:
    """去掉 BOM、剥离模型偶发包裹的输出代码围栏，再供校验与写盘。"""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    if t.startswith("\ufeff"):
        t = t[1:]
    t = t.strip()
    if t.startswith("```"):
        t = unwrap_fenced_code(text).strip()
    return t


# --- LLM 调用（DashScope 兼容接口）---

DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    os.environ.get(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)
DASHSCOPE_API_KEY = os.environ.get(
    "DASHSCOPE_API_KEY", os.environ.get("OPENAI_API_KEY", "")
)
DASHSCOPE_MODEL = os.environ.get(
    "DASHSCOPE_MODEL", os.environ.get("OPENAI_MODEL", "qwen3.5-32b-instruct")
)
DEFAULT_LLM_TIMEOUT = 1000


def _chat(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    timeout: int,
    max_tokens: int,
) -> str | None:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            stream=False,
            temperature=0,
            top_p=0.1,
            extra_body={"enable_thinking": False, "reasoning_split": True},
            timeout=timeout,
        )
        content = (
            (r.choices[0].message.content or "").strip() if r and r.choices else ""
        )
        if not content:
            return None
        return content
    except Exception as e:
        logger.debug("LLM 请求异常: {}", e)
        return None


def call_dashscope(
    system: str,
    user: str,
    timeout: int = DEFAULT_LLM_TIMEOUT,
    model_override: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str | None:
    """调用 DashScope（OpenAI 兼容接口），失败返回 None。"""
    base = (os.environ.get("DASHSCOPE_BASE_URL") or DASHSCOPE_BASE_URL).strip()
    key = (os.environ.get("DASHSCOPE_API_KEY") or DASHSCOPE_API_KEY).strip()
    model = model_override or os.environ.get("DASHSCOPE_MODEL", DASHSCOPE_MODEL)
    if not key:
        logger.error("未设置 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）")
        return None
    client = OpenAI(api_key=key, base_url=base)
    out = _chat(client, model, system, user, timeout, max_tokens=max_tokens)
    if out is None:
        logger.warning("DashScope 调用失败或超时 (base={}, model={})", base, model)
    return out


def call_llm(
    system: str,
    user: str,
    timeout: int = DEFAULT_LLM_TIMEOUT,
    model_name: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str | None:
    """仅使用 DashScope 调用，失败返回 None。调用前打印完整 system / user 便于调试。"""
    logger.info(
        "【LLM 请求】system 字符数={}, user 字符数={}",
        len(system),
        len(user),
    )
    logger.info("---------- LLM system ----------\n{}", system)
    logger.info("---------- LLM user ----------\n{}", user)
    return call_dashscope(
        system,
        user,
        timeout,
        model_override=model_name,
        max_tokens=max_tokens,
    )


# --- Prompt 与单文件处理 ---

SYSTEM_PROMPT = """你是 Markdown 结构整理助手。

【唯一允许的操作】
1. 你只可以对以下行做修改：
   - 以 `# `、`## `、`### `、`#### `、`##### `、`###### ` 开头的「标题行」。
2. 对标题行，你可以：
   - 调整 `#` 的个数（修改标题层级），
   - 在文档内移动整段「标题 + 以下到下一个标题之间的正文块」的位置。

【编号与层级（强约束，常见于国家标准等文档）】
1. 标题文本若以多级章节编号开头（如 2.1、2.2、3.2.1，即「数字.数字…」形式），一律不得使用一级标题「# 」；须使用「##」或更深，使其层级高于父节（如「2」「3」或「2 术语和定义」等单节编号节）。
2. 单节编号（如「1」「2 范围」仅一段数字后接空格/文字）可以是一级标题，但其下的 2.1、2.2 等必须比该节更深至少一级。
3. 禁止把同一文档中大量小节全部标成「#」；结构应反映从属关系。

【目录/目次识别规则（强约束）】
1. 只允许根据文档中“真实 Markdown 标题行”（即以 `#` 开头且后接空格）判断结构。
2. 不得把“目次/目录”段落中的普通文本行当作标题来源。
3. 不得根据目录项去推断、补全、改写、重命名任何标题。
4. 除下条「相邻标题合并」外，标题用词应与输入一致；只允许改变标题前缀 `#` 个数、块位置，以及按规则合并。

【相邻标题合并（允许）】
1. 可将**语义上同属一节、在文中相邻**的多条标题行**合并为一条**（例如：重复的「食品安全国家标准」与标准全称分两行；「附录 A」与「检验方法」分两行；图谱类「B.1 …」与较长说明分两行）。
2. 合并后的标题文字须由被合并各行的原文**按顺序直接衔接**而成（可适当保留空格），**不得编造**输入中不存在的词句。
3. 合并时须把原属各标题下的正文块一并归入合并后的该节，**不得丢弃**正文或表格。

【严格禁止的操作】
1. 所有**非标题行**（不以 `#` 开头的行），必须逐字保留：
   - 不得新增删除任何行，
   - 不得改动任何字符（包括标点、空格、公式、数字、中文等）。
2. 不允许为了“整理结构”而改写段落内容，只能移动「标题+段落块」的位置。
3. 除上述「相邻标题合并」外，不得无故增加标题行或删除章节；合并导致标题行数变少是允许的。
4. 不要添加你自己的说明、注释或解释性文字。

【输出要求】
- 直接输出整理后的完整 Markdown 文本，不要使用代码块包裹。
- 非标题行须与输入逐字一致；标题行可按上文调整层级、顺序与合并。"""

USER_PROMPT_PREFIX = (
    "请仅整理以下文档的标题层级与顺序，保持正文不变。"
    "可将相邻且同属一节的标题合并为一行（如附录编号与附录名、重复封面题名），合并文字须来自原文衔接。"
    "只能依据真实 Markdown 标题行（# 后跟空格）判断结构，"
    "不要根据“目次/目录”文本推断或补全标题。"
    "若以 2.1、3.2.1 这类多级编号开头，不得使用一级标题 #。\n\n"
)

USER_PROMPT_INPUT_HINT_SUSPICIOUS_H1 = (
    "【硬性提醒】输入中仍有以多级编号（如 2.1、3.2.1）开头却标为「# 」的标题；"
    "输出中它们必须改为 ## 或更深，标题文字与正文须与下文逐字一致。\n\n"
)

RETRY_FEEDBACK_HEADINGS = (
    "\n\n【上次输出未通过校验】标题行数量与各标题文字必须与上述文档完全一致，"
    "不得增删或改写任何标题文字；请仅调整 # 层级与块顺序后重新输出全文。"
)

RETRY_FEEDBACK_SUSPICIOUS_H1 = (
    "\n\n【上次输出未通过校验】仍将「2.1」「3.2.1」这类多级编号小节标为一级标题 #。"
    "它们至少应为 ## 或更深，且须与父节形成正确从属关系；标题文字与正文须与上述文档逐字一致，请重新输出全文。"
)

SYSTEM_PLAN_PROMPT = """你是 Markdown 标题重排规划器。

你只需返回一个 JSON 数组（不要任何额外文字、不要代码块）：
[
  {"id":"H0001","level":1},
  {"id":"H0002","level":2}
]

严格要求：
1. 只能使用输入给出的 id，不得新增/删除/重复。
2. level 必须是 1..6 的整数。
3. 不要输出除 JSON 数组外的任何字符。"""

SYSTEM_HEADING_PLAN_MERGE_PROMPT = """你是 Markdown 标题重排规划器（标题计划模式）。

你只根据输入的 Markdown「标题块列表」输出**标题计划**：使用纯文本行描述，**不要使用 JSON**。

每一行一条指令（可省略空行；不要用代码块包裹整段输出亦可）：

1) 单块（对应一个原标题块）：
   block <id> <level>
   示例：block H0001 2

2) 合并块（将**原文档顺序中相邻**的多个块合并为一行标题；合并后标题文字由程序按各块标题文字顺序直接拼接，你不得改写文字）：
   merge <id1> <id2> ... <idN> <level>
   最后一个数字为层级。示例：merge H0001 H0002 1

严格要求：
- 只能使用输入中出现的 id，每个 id 在整段计划中**恰好出现一次**（出现在唯一一行 block 或唯一一行 merge 的 id 列表里）。
- merge 中的块在原文顺序上必须**连续**（不允许跳过中间块合并）。
- level 必须是 1..6 的整数。
- 可调整输出行的顺序以反映正确的章节层级与从属关系；可将相邻块合并为 merge。
- 多级编号小节（如 2.1、3.2.1）不得给 level=1。
- 除上述 block/merge 行外不要输出其它解释文字（不要用 JSON）。"""

USER_HEADING_PLAN_PREFIX = (
    "以下为文档的标题块摘录（Markdown）。请根据系统说明输出 block/merge 计划行。\n\n"
)

RETRY_FEEDBACK_PLAN_LINES = (
    "\n\n【上次输出无效】请只输出文本行：block <id> <level> 或 merge <id>... <level>；"
    "不要使用 JSON；每个 id 恰好出现一次；merge 的 id 须在原文顺序中连续。"
)

RETRY_FEEDBACK_PLAN_BODY = (
    "\n\n【上次输出未通过校验】重建后围栏内外非标题正文行必须与输入逐行多重集一致；"
    "请只调整 op、顺序、level 与 merge，不要改变任何标题文字拼接结果以外的内容。"
)


def reorganize_heading_plan_with_retries(
    working: str,
    path: Path,
    *,
    timeout: int,
    model_name: str | None,
    max_tokens: int,
    max_llm_retries: int,
    chunk_tag: str = "",
) -> str | None:
    """标题计划模式：骨架送 LLM，按 block/merge 计划重建。"""
    _, blocks = split_heading_blocks(working)
    if not blocks:
        return working

    skeleton = build_heading_skeleton_markdown(blocks)
    _, susp_in = count_h1_and_suspicious(working)
    input_hint = USER_PROMPT_INPUT_HINT_SUSPICIOUS_H1 if susp_in > 0 else ""
    feedback = ""
    extra = f" ({chunk_tag})" if chunk_tag else ""

    for attempt in range(max_llm_retries):
        user_content = (
            USER_HEADING_PLAN_PREFIX + input_hint + skeleton + feedback
        )
        raw_out = call_llm(
            SYSTEM_HEADING_PLAN_MERGE_PROMPT,
            user_content,
            timeout,
            model_name=model_name,
            max_tokens=max_tokens,
        )
        if raw_out is None:
            logger.warning(
                "LLM 调用失败或空输出 {}{} (尝试 {}/{})",
                path,
                extra,
                attempt + 1,
                max_llm_retries,
            )
            continue

        prepared = prepare_model_markdown(raw_out)
        if not prepared:
            logger.warning(
                "模型输出经去围栏/BOM 后为空 {}{} (尝试 {}/{})",
                path,
                extra,
                attempt + 1,
                max_llm_retries,
            )
            continue

        built = build_by_heading_plan_ops(working, prepared)
        if built is None:
            logger.warning(
                "标题计划解析或校验失败 {}{} (尝试 {}/{})",
                path,
                extra,
                attempt + 1,
                max_llm_retries,
            )
            feedback = RETRY_FEEDBACK_PLAN_LINES
            continue

        if non_heading_line_counter(working) != non_heading_line_counter(built):
            logger.warning(
                "标题计划模式：正文多重集不一致 {}{} (尝试 {}/{})",
                path,
                extra,
                attempt + 1,
                max_llm_retries,
            )
            feedback = RETRY_FEEDBACK_PLAN_BODY
            continue

        if not suspicious_h1_audit_passes(built):
            total_h1, susp = count_h1_and_suspicious(built)
            logger.warning(
                "一级标题结构审核未通过 {}{}：可疑 H1 {} / 一级共 {} (尝试 {}/{})",
                path,
                extra,
                susp,
                total_h1,
                attempt + 1,
                max_llm_retries,
            )
            feedback = RETRY_FEEDBACK_SUSPICIOUS_H1
            continue

        return built

    logger.error("标题计划模式失败（已达最大 LLM 尝试次数）: {}{}", path, extra)
    return None


def reorganize_markdown_with_retries(
    working: str,
    path: Path,
    *,
    timeout: int,
    model_name: str | None,
    max_tokens: int,
    max_llm_retries: int,
    strict_headings: bool,
    chunk_tag: str = "",
) -> str | None:
    """
    对一段 Markdown 调用 LLM 整理标题，通过校验后返回结果；失败返回 None。
    chunk_tag 非空时写入日志，便于多段并行识别。
    """
    _, susp_in = count_h1_and_suspicious(working)
    input_hint = USER_PROMPT_INPUT_HINT_SUSPICIOUS_H1 if susp_in > 0 else ""
    feedback = ""
    out: str | None = None
    extra = f" ({chunk_tag})" if chunk_tag else ""
    for attempt in range(max_llm_retries):
        user_content = USER_PROMPT_PREFIX + input_hint + working + feedback
        out = call_llm(
            SYSTEM_PROMPT,
            user_content,
            timeout,
            model_name=model_name,
            max_tokens=max_tokens,
        )
        if out is None:
            logger.warning(
                "LLM 调用失败或空输出 {}{} (尝试 {}/{})",
                path,
                extra,
                attempt + 1,
                max_llm_retries,
            )
            continue

        out = prepare_model_markdown(out)
        if not out:
            logger.warning(
                "模型输出经去围栏/BOM 后为空 {}{} (尝试 {}/{})",
                path,
                extra,
                attempt + 1,
                max_llm_retries,
            )
            continue

        if strict_headings and not headings_unchanged(working, out):
            logger.warning(
                "标题集合校验失败 {}{} (尝试 {}/{})\n{}",
                path,
                extra,
                attempt + 1,
                max_llm_retries,
                headings_mismatch_report(working, out),
            )
            feedback = RETRY_FEEDBACK_HEADINGS
            continue

        if not suspicious_h1_audit_passes(out):
            total_h1, susp = count_h1_and_suspicious(out)
            logger.warning(
                "一级标题结构审核未通过 {}{}：可疑 H1 {} / 一级共 {} (尝试 {}/{})",
                path,
                extra,
                susp,
                total_h1,
                attempt + 1,
                max_llm_retries,
            )
            feedback = RETRY_FEEDBACK_SUSPICIOUS_H1
            continue

        return out

    logger.error("处理失败（已达最大 LLM 尝试次数）: {}{}", path, extra)
    return None


def process_one(
    path: Path,
    out_suffix: str,
    out_dir: Path | None,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: int = DEFAULT_LLM_TIMEOUT,
    model_name: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    preprocess_h1: bool = True,
    max_llm_retries: int = MAX_LLM_RETRIES,
    strict_headings: bool = False,
    skip_existing: bool = False,
    single_file_mode: bool = False,
    heading_plan: bool = False,
) -> ProcessResult:
    """处理单个 MD 文件：读入 → 可选预降级 H1 → 调 LLM（可重试）→ 校验（可选标题 multiset + 一级标题规则）→ 写新文件。

    single_file_mode 为 True（即命令行使用 --file）时：不检测目标是否已存在；
    源长度超过 max_chars 时若存在 CHUNK_SPLIT_MARKER 则按标识分段，否则仍整篇调用 LLM（可覆盖已有输出）。
    heading_plan 为 True 时仅送标题骨架，走 block/merge 计划重建，且不因全文长度触发超长跳过。
    """
    if out_dir is not None:
        out_path = out_dir / (path.stem + out_suffix + path.suffix)
    else:
        out_path = path.parent / (path.stem + out_suffix + path.suffix)

    if not single_file_mode and skip_existing and out_path.exists():
        logger.info("<cyan>跳过</cyan> {} -> {}", path, out_path)
        return ProcessResult(path, out_path, ProcessOutcome.SKIP_EXISTING)

    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("读取失败 {}: {}", path, e)
        return ProcessResult(path, out_path, ProcessOutcome.ERROR_READ, detail=str(e))

    n = len(raw)
    working = (
        preprocess_demote_obvious_subsection_h1(raw) if preprocess_h1 else raw
    )

    if heading_plan:
        logger.info("开始处理（标题计划模式）: {} -> {}", path, out_path)
        logger.info("将调用 LLM（标题骨架）: {} -> {}", path, out_path)
        out_hp = reorganize_heading_plan_with_retries(
            working,
            path,
            timeout=timeout,
            model_name=model_name,
            max_tokens=max_tokens,
            max_llm_retries=max_llm_retries,
        )
        if out_hp is None:
            return ProcessResult(
                path,
                out_path,
                ProcessOutcome.ERROR_LLM,
                detail="heading plan: max LLM retries exhausted",
            )
        try:
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(out_hp, encoding="utf-8")
            logger.info("已写入: {}", out_path)
            return ProcessResult(path, out_path, ProcessOutcome.SUCCESS, char_count=n)
        except Exception as e:
            logger.error("写入失败 {}: {}", out_path, e)
            return ProcessResult(path, out_path, ProcessOutcome.ERROR_WRITE, detail=str(e))

    marker_parts = (
        split_by_chunk_marker(working, max_chars)
        if max_chars >= 0 and n > max_chars
        else None
    )
    if max_chars >= 0 and n > max_chars:
        if marker_parts is not None:
            chunks = marker_parts
        elif single_file_mode:
            chunks = [working]
        else:
            logger.info(
                "<yellow>超长跳过</yellow> chars={}/{}（无 {} 或某段仍超限） {} -> {}",
                n,
                max_chars,
                CHUNK_SPLIT_MARKER,
                path,
                out_path,
            )
            return ProcessResult(
                path, out_path, ProcessOutcome.SKIP_TOO_LONG, char_count=n
            )
    else:
        chunks = [working]

    logger.info("开始处理: {} -> {}", path, out_path)
    if len(chunks) > 1:
        logger.info(
            "<cyan>超长分 {} 段</cyan> marker={!r} {}",
            len(chunks),
            CHUNK_SPLIT_MARKER,
            path,
        )

    # 未跳过时通常会长时间等待 LLM（默认 --timeout 很大），提前打日志避免误以为进程卡死
    logger.info("将调用 LLM: {} -> {}", path, out_path)

    merged_parts: list[str] = []
    for ci, chunk in enumerate(chunks):
        tag = f"段 {ci + 1}/{len(chunks)}" if len(chunks) > 1 else ""
        part = reorganize_markdown_with_retries(
            chunk,
            path,
            timeout=timeout,
            model_name=model_name,
            max_tokens=max_tokens,
            max_llm_retries=max_llm_retries,
            strict_headings=strict_headings,
            chunk_tag=tag,
        )
        if part is None:
            return ProcessResult(
                path,
                out_path,
                ProcessOutcome.ERROR_LLM,
                detail="max LLM retries exhausted",
            )
        merged_parts.append(part)

    out = "\n".join(merged_parts)

    if len(chunks) > 1:
        if strict_headings and not headings_unchanged(working, out):
            logger.error(
                "合并后全文标题集合与输入不一致 {}\n{}",
                path,
                headings_mismatch_report(working, out),
            )
            return ProcessResult(
                path,
                out_path,
                ProcessOutcome.ERROR_LLM,
                detail="merged output headings mismatch",
            )
        if not suspicious_h1_audit_passes(out):
            total_h1, susp = count_h1_and_suspicious(out)
            logger.error(
                "合并后全文一级标题审核未通过 {}：可疑 H1 {} / 一级共 {}",
                path,
                susp,
                total_h1,
            )
            return ProcessResult(
                path,
                out_path,
                ProcessOutcome.ERROR_LLM,
                detail="merged output suspicious H1 audit failed",
            )

    try:
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
        logger.info("已写入: {}", out_path)
        return ProcessResult(path, out_path, ProcessOutcome.SUCCESS, char_count=n)
    except Exception as e:
        logger.error("写入失败 {}: {}", out_path, e)
        return ProcessResult(path, out_path, ProcessOutcome.ERROR_WRITE, detail=str(e))


def main() -> None:
    args = parse_args()
    files = collect_md_files(args)

    # 仅在目录批量模式下支持按区间选择要处理的文件
    if args.file is None and (args.start is not None or args.end is not None):
        total = len(files)
        if total == 0:
            logger.info("没有找到待处理的 .md 文件")
            return

        start = args.start if args.start is not None else 1
        end = args.end if args.end is not None else total

        if start < 1 or end < 1 or start > total or end > total or start > end:
            logger.error(
                "无效的区间 --start/--end（有效范围: 1-{} 且 start <= end），当前为 start={}, end={}",
                total,
                start,
                end,
            )
            sys.exit(1)

        # Python 切片右侧为开区间，这里将用户给定的闭区间 [start, end] 转为切片
        files = files[start - 1 : end]
        logger.info(
            "按区间处理文件: 总数 {}, 实际处理区间 [{}-{}]，共 {} 个文件",
            total,
            start,
            end,
            len(files),
        )

    if not files:
        logger.info("没有找到待处理的 .md 文件")
        return

    if args.dry_run:
        logger.info("--dry-run: 共 {} 个文件", len(files))
        for p in files:
            print(p)
        return

    out_suffix = args.out_suffix or DEFAULT_OUT_SUFFIX
    out_dir = args.out_dir.resolve() if args.out_dir else None
    timeout = getattr(args, "timeout", DEFAULT_LLM_TIMEOUT)
    model_name = getattr(args, "model", None)
    max_tokens = getattr(args, "max_tokens", DEFAULT_MAX_TOKENS)
    workers = getattr(args, "workers", 1)
    preprocess_h1 = not args.no_preprocess_h1
    max_llm_retries = args.max_retries
    strict_headings = args.strict_headings
    skip_existing = args.skip_existing
    max_chars = args.max_chars
    event_log_path = args.event_log.resolve()
    single_file_mode = args.file is not None
    heading_plan = getattr(args, "heading_plan", False)

    def _job(path: Path) -> ProcessResult:
        return process_one(
            path,
            out_suffix,
            out_dir,
            max_chars=max_chars,
            timeout=timeout,
            model_name=model_name,
            max_tokens=max_tokens,
            preprocess_h1=preprocess_h1,
            max_llm_retries=max_llm_retries,
            strict_headings=strict_headings,
            skip_existing=skip_existing,
            single_file_mode=single_file_mode,
            heading_plan=heading_plan,
        )

    if workers <= 1 or len(files) == 1:
        results_list = [_job(p) for p in files]
    else:
        pool = min(workers, len(files))
        with ThreadPoolExecutor(max_workers=pool) as ex:
            results_list = list(ex.map(_job, files))

    write_event_log_file(event_log_path, results_list)

    ok = sum(1 for r in results_list if r.outcome == ProcessOutcome.SUCCESS)
    skip_existing_n = sum(
        1 for r in results_list if r.outcome == ProcessOutcome.SKIP_EXISTING
    )
    skip_long_n = sum(
        1 for r in results_list if r.outcome == ProcessOutcome.SKIP_TOO_LONG
    )
    failed_paths = [
        r.path
        for r in results_list
        if r.outcome
        in (
            ProcessOutcome.ERROR_READ,
            ProcessOutcome.ERROR_LLM,
            ProcessOutcome.ERROR_WRITE,
        )
    ]
    logger.info(
        "完成: 共 {} 个文件, 成功 {}, 跳过已存在 {}, 超长跳过 {}, 失败 {}",
        len(files),
        ok,
        skip_existing_n,
        skip_long_n,
        len(failed_paths),
    )
    for p in failed_paths:
        logger.warning("失败: {}", p)
    logger.info("事件日志已追加: {}", event_log_path)


if __name__ == "__main__":
    main()
