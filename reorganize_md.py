#!/usr/bin/env python3
"""用 LLM 整理 Markdown 标题结构（仅改层级与顺序，不改正文），结果写入新文件。"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from loguru import logger
from openai import OpenAI

# 默认输出后缀（xxx.md -> xxx.reorganized.md）
DEFAULT_OUT_SUFFIX = ".reorganized"
# 默认递归目录
DEFAULT_DIR = "_all_md"
# 默认允许的最大输出 token
DEFAULT_MAX_TOKENS = 20480
MAX_LLM_RETRIES = 3
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
        help="仅处理单个 .md 文件",
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
    args = parser.parse_args()
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
    # 排除已整理文件：名称包含 .reorganized. 或以 .reorganized.md 结尾
    exclude_suffix = out_suffix + ".md"

    files: list[Path] = []
    for f in dir_path.rglob("*.md"):
        if not f.is_file():
            continue
        name = f.name
        if name.endswith(exclude_suffix) or (
            out_suffix + "." in name and name.endswith(".md")
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


def _normalize_body(text: str) -> str:
    """统一换行、去掉首尾空白，用于正文对比。"""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def strip_heading_lines(text: str) -> str:
    """去掉所有 Markdown 标题行（^#+ 后跟空格），保留其余行，规范化后返回。"""
    lines = text.split("\n")
    kept = [ln for ln in lines if not HEADING_LINE_RE.match(ln)]
    return _normalize_body("\n".join(kept))


def body_unchanged(original: str, llm_output: str) -> bool:
    """对比原文与 LLM 输出的「仅正文」部分是否一致（标题行不计）。"""
    a = strip_heading_lines(original)
    b = strip_heading_lines(llm_output)
    return a == b


def heading_texts(text: str) -> list[str]:
    """提取标题文本（去掉 # 前缀），用于校验标题是否被增删/改写。"""
    out: list[str] = []
    for ln in text.split("\n"):
        if HEADING_LINE_RE.match(ln):
            out.append(re.sub(r"^#+\s*", "", ln).strip())
    return out


def headings_unchanged(original: str, llm_output: str) -> bool:
    """校验标题文本集合是否一致（允许调层级和块顺序，不允许增删改标题文本）。"""
    return Counter(heading_texts(original)) == Counter(heading_texts(llm_output))


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
                {
                    "role": "system",
                    "content": "整理当前文档，只需要调整文档的结构，不需要做任何其他改动。清除没有意义的换行和空格。不可以增加任何新的内容。直接输出整理后的文档，不要任何解释。",
                },
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            stream=False,
            temperature=0,
            top_p=0.1,
            extra_body={"enable_thinking": False},
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
    """仅使用 DashScope 调用，失败返回 None。"""
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

【目录/目次识别规则（强约束）】
1. 只允许根据文档中“真实 Markdown 标题行”（即以 `#` 开头且后接空格）判断结构。
2. 不得把“目次/目录”段落中的普通文本行当作标题来源。
3. 不得根据目录项去推断、补全、改写、重命名任何标题。
4. 标题文本必须与输入逐字一致；只允许改变标题前缀 `#` 数量和块位置。

【严格禁止的操作】
1. 所有**非标题行**（不以 `#` 开头的行），必须逐字保留：
   - 不得新增删除任何行，
   - 不得改动任何字符（包括标点、空格、公式、数字、中文等）。
2. 不允许为了“整理结构”而改写段落内容，只能移动「标题+段落块」的位置。
3. 不允许新增或删除任何标题行。
4. 不要添加你自己的说明、注释或解释性文字。

【输出要求】
- 直接输出整理后的完整 Markdown 文本，不要使用代码块包裹。
- 除了上述允许的标题行修改和段落块移动外，输出必须与输入逐字一致。"""

USER_PROMPT_PREFIX = (
    "请仅整理以下文档的标题层级与顺序，保持正文不变。"
    "只能依据真实 Markdown 标题行（# 后跟空格）判断结构，"
    "不要根据“目次/目录”文本推断或补全标题。\n\n"
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


def process_one(
    path: Path,
    out_suffix: str,
    out_dir: Path | None,
    timeout: int = DEFAULT_LLM_TIMEOUT,
    model_name: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> bool:
    """处理单个 MD 文件：读入 → 调 LLM → 正文校验 → 写新文件。成功返回 True。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("读取失败 {}: {}", path, e)
        return False

    base_user_content = USER_PROMPT_PREFIX + raw
    out = call_llm(
        SYSTEM_PROMPT,
        base_user_content,
        timeout,
        model_name=model_name,
        max_tokens=max_tokens,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (path.stem + out_suffix + path.suffix)
    else:
        out_path = path.parent / (path.stem + out_suffix + path.suffix)

    try:
        out_path.write_text(out, encoding="utf-8")
        logger.info("已写入: {}", out_path)
        return True
    except Exception as e:
        logger.error("写入失败 {}: {}", out_path, e)
        return False


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
    ok, failed_paths = 0, []
    for path in files:
        if process_one(
            path,
            out_suffix,
            out_dir,
            timeout=timeout,
            model_name=model_name,
            max_tokens=max_tokens,
        ):
            ok += 1
        else:
            failed_paths.append(path)
    logger.info(
        "完成: 共 {} 个文件, 成功 {}, 失败 {}", len(files), ok, len(failed_paths)
    )
    for p in failed_paths:
        logger.warning("失败: {}", p)


if __name__ == "__main__":
    main()
