#!/usr/bin/env python3
"""对仍无法解析 GB 编号的 Markdown，用 LLM 根据裁剪后的封面上下文推断标准号。

读取 skipped_no_gb.tsv（或任意 TSV 首列为 source_relative），为每个文件生成 snippet，
调用 DashScope/OpenAI 兼容接口，输出 audit.jsonl、mapping_from_llm.tsv、mapping_manual_review.tsv。

依赖与 reorganize_md.py 相同：.env 中 DASHSCOPE_API_KEY 或 OPENAI_API_KEY。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

ENV_PATH = Path(__file__).with_name(".env")
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    os.environ.get(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)
DEFAULT_MODEL = os.environ.get(
    "DASHSCOPE_MODEL", os.environ.get("OPENAI_MODEL", "qwen3.5-32b-instruct")
)

GB_CODE_VALID = re.compile(r"^\d[\d.]*-\d{4}$")
PUBLISH_LINE = re.compile(r"^\d{4}-\d{2}-\d{2}\s*(发布|实施)")
LOG_PROCESSING = re.compile(r"Processing\s+\[[^\]]+\]\s*(.+)$")


def load_skipped_paths(tsv: Path) -> list[str]:
    out: list[str] = []
    with tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or row[0].strip().lower() in ("source_relative", ""):
                continue
            if row[0].startswith("#"):
                continue
            out.append(row[0].strip())
    return out


def find_log_line(log_path: Path, basename: str) -> str | None:
    if not log_path.is_file():
        return None
    needle = basename.strip()
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if needle in line and "Processing" in line:
                m = LOG_PROCESSING.search(line)
                return (m.group(1).strip() if m else line.strip())[:400]
    return None


def build_snippet(
    rel: str,
    body: str,
    max_lines: int = 55,
    max_chars: int = 3500,
) -> str:
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")[:max_lines]
    kept: list[str] = [f"文件名: {rel}"]
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if "<table" in s.lower() or s.startswith("<"):
            continue
        if s.startswith("#"):
            kept.append(s)
            continue
        if PUBLISH_LINE.match(s):
            kept.append(s)
            continue
        if len(s) < 120 and "GB" not in s.upper():
            kept.append(s)
    text = "\n".join(kept)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(截断)"
    return text


def unwrap_json_block(text: str) -> str:
    t = text.strip()
    for sep in ("`</think>`", "</think>"):
        if sep in t:
            t = t.split(sep)[-1].strip()
            break
    if t.startswith("```"):
        lines = t.split("\n")
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if "{" in t:
        a, b = t.find("{"), t.rfind("}")
        if a != -1 and b > a:
            t = t[a : b + 1]
    return t


def call_llm(
    client: OpenAI,
    model: str,
    user_payload: str,
    timeout: int,
) -> str | None:
    schema_hint = (
        '{"gb_code":null,"proposed_title":null,"confidence":"low","rationale":""}'
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是中国食品安全国家标准编号助手。根据文件名与文档开头片段，"
                        "判断本文件对应的国家标准编号（仅数字段-四位年份，如 1886.1-2015）。\n"
                        "禁止输出思考过程、markdown、代码围栏。禁止输出 GB/T 作为本标准号。\n"
                        "无法确定时 gb_code 为 null，confidence 为 low。\n"
                        "必须且只能输出一个 JSON 对象，键：gb_code, proposed_title, confidence, rationale。"
                        f"示例空对象：{schema_hint}"
                    ),
                },
                {"role": "user", "content": user_payload},
            ],
            max_tokens=512,
            temperature=0,
            timeout=timeout,
            extra_body={"enable_thinking": False},
        )
        return (r.choices[0].message.content or "").strip() if r.choices else None
    except Exception as e:
        print(f"LLM 错误: {e}", file=sys.stderr)
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--skipped-tsv",
        type=Path,
        default=Path("_all_md_renamed/skipped_no_gb.tsv"),
        help="首列为 source_relative 的 TSV",
    )
    p.add_argument("--src", type=Path, default=Path("_all_md"), help="Markdown 根目录")
    p.add_argument(
        "--convert-log",
        type=Path,
        default=Path("convert_to_md.log"),
        help="可选：转换日志，附加 Processing 行摘要",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("_all_md_renamed"),
        help="输出目录",
    )
    p.add_argument("--model", type=str, default=None, help="模型名")
    p.add_argument("--timeout", type=int, default=120, help="单次请求超时秒数")
    p.add_argument("--start", type=int, default=1, help="从第几条开始（1-based）")
    p.add_argument("--limit", type=int, default=None, help="最多处理条数")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="不调用 LLM，只打印将处理的文件与 snippet 长度",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="请求间隔秒数，降低限频",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src_root = args.src.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = load_skipped_paths(args.skipped_tsv.resolve())
    model = args.model or DEFAULT_MODEL
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not args.dry_run and not key:
        print("未设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY", file=sys.stderr)
        return 1

    start = max(1, args.start)
    end = len(paths) + 1
    if args.limit is not None:
        end = min(end, start + args.limit)

    slice_paths = paths[start - 1 : end - 1]
    audit_path = out_dir / "enrich_gb_audit.jsonl"
    map_ok = out_dir / "mapping_from_llm.tsv"
    map_review = out_dir / "mapping_manual_review.tsv"

    client: OpenAI | None = None
    if not args.dry_run:
        client = OpenAI(
            api_key=key,
            base_url=(os.environ.get("DASHSCOPE_BASE_URL") or DASHSCOPE_BASE_URL).strip(),
        )

    ok_rows: list[tuple[str, str]] = []
    review_rows: list[tuple[str, str, str]] = []

    mode = "a" if audit_path.exists() else "w"
    with audit_path.open(mode, encoding="utf-8") as audit_f:
        for rel in slice_paths:
            path = src_root / rel
            if not path.is_file():
                rec = {
                    "source_relative": rel,
                    "error": "file_not_found",
                    "gb_code": None,
                    "confidence": "low",
                }
                audit_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                review_rows.append((rel, "", "file_not_found"))
                continue

            body = path.read_text(encoding="utf-8", errors="replace")
            snippet = build_snippet(rel, body)
            log_hint = find_log_line(args.convert_log.resolve(), Path(rel).name)
            user = "【文档片段】\n" + snippet
            if log_hint:
                user += "\n\n【转换日志摘要（仅供参考）】\n" + log_hint

            if args.dry_run:
                print(f"{rel}\tsnippet_len={len(snippet)}")
                continue

            assert client is not None
            raw = call_llm(client, model, user, args.timeout)
            rec: dict = {
                "source_relative": rel,
                "raw_response": raw,
            }
            gb_code = None
            conf = "low"
            rationale = ""
            if raw:
                try:
                    obj = json.loads(unwrap_json_block(raw))
                    rec["parsed"] = obj
                    gb_code = obj.get("gb_code")
                    conf = str(obj.get("confidence") or "low").lower()
                    rationale = str(obj.get("rationale") or "")
                except json.JSONDecodeError as e:
                    rec["parse_error"] = str(e)

            rec["gb_code"] = gb_code
            rec["confidence"] = conf
            audit_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            audit_f.flush()

            code_str = str(gb_code).strip() if gb_code else ""
            if code_str and GB_CODE_VALID.match(code_str.replace("—", "-")):
                code_str = code_str.replace("—", "-")
                if conf in ("high", "medium"):
                    ok_rows.append((rel, code_str))
                else:
                    review_rows.append((rel, code_str, rationale or conf))
            else:
                review_rows.append((rel, code_str, rationale or "no_valid_code"))

            time.sleep(args.sleep)

    if args.dry_run:
        print(f"dry-run: would process {len(slice_paths)} file(s)")
        return 0

    with map_ok.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source_relative", "gb_code"])
        w.writerows(ok_rows)

    with map_review.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source_relative", "gb_code_guess", "note"])
        w.writerows(review_rows)

    print(
        f"done: mapping_from_llm={len(ok_rows)} manual_review={len(review_rows)} "
        f"audit={audit_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
