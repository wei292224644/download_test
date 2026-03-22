#!/usr/bin/env python3
"""将食品安全国家标准 Markdown 复制到新目录，统一文件名为「GB <编号> <全称>.md」。

不修改源目录；无法解析标准号的文件跳过并写入 skipped_no_gb.tsv。
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

# 标准号：数字段 + 分隔符 + 四位年份
GB_CODE_CORE = r"\d[\d.]*\s*[-—]\s*\d{4}"
# 文件名中尾部噪声（去掉扩展名后匹配）
FILENAME_TRAIL_JUNK = re.compile(
    r"(?:\s+现行|\s*\.?\s*doc|\s+\.doc)\s*$",
    re.IGNORECASE,
)
# 前导序号：「1 牛奶中…」
FILENAME_LEADING_ENUM = re.compile(r"^\d+\s+")

HEADING_RE = re.compile(r"^#+\s*(.*)$")
# 正文独行标准号
BODY_GB_LINE = re.compile(
    rf"^GB\s*{GB_CODE_CORE}\s*$",
    re.IGNORECASE,
)
# 正文紧凑标准号（整行或行尾落款）
BODY_GB_COMPACT = re.compile(
    rf"GB\s*({GB_CODE_CORE})",
    re.IGNORECASE,
)
# 文件名内：（GB x.x-2022）
FNAME_GB_PAREN_CN = re.compile(
    rf"[（(]\s*GB\s*({GB_CODE_CORE})\s*[）)]",
    re.IGNORECASE,
)
# 文件名开头 GB 与编号（四位年份 + 连字符）
FNAME_GB_PREFIX = re.compile(
    rf"^GB[_\s\-]*({GB_CODE_CORE})",
    re.IGNORECASE,
)
# GB 29921 2013、GB29922 2013（空格分隔年份，无连字符）
FNAME_GB_SPACE_YEAR = re.compile(
    r"^GB\s*(\d[\d.]*)\s+(\d{4})\b",
    re.IGNORECASE,
)
# GB29923 2013（GB 与数字紧贴）
FNAME_GB_NO_SPACE_BEFORE_NUM = re.compile(
    r"^GB(\d[\d.]*)\s+(\d{4})\b",
    re.IGNORECASE,
)
# 文件名中错误的三位年份（如 4789.28-203），需结合正文「发布」日期修正
FNAME_TRAILING_3DIGIT_YEAR = re.compile(
    r"(GB[_\s\-]*\d[\d.]*)\s*[-—]\s*(\d{3})(?=[^\d\-]|$)",
    re.IGNORECASE,
)
# 修改单
AMENDMENT_RE = re.compile(r"第\s*(\d+)\s*号\s*修改单")

# 标题收集停止：明显进入正文结构
TITLE_STOP_HEADINGS = re.compile(
    r"^(?:目次|前言|\d+[\s\.]?\s*范围|"
    r"National\s+food\s+safety|"
    r"Determination\s+of|"
    r"中华人民共和国国家标准)\s*$",
    re.IGNORECASE,
)

INVALID_FILENAME_CHARS = '/\\:*?"<>|'

# 正文紧凑匹配时跳过（多为引用其他标准）
CITATION_LINE_HINTS = re.compile(
    r"应符合|按\s*GB|按GB|见\s*GB|见GB|引用|参照|依据\s*GB|执行\s*GB|按照\s*GB|"
    r"GB/T|GB\s*4789|GB\s*2760|GB\s*2761|GB\s*2762|采样后",
)


def normalize_gb_code(match_or_str: str) -> str:
    """统一为「数字.子版本-年份」形式，年份前为 ASCII 连字符。"""
    s = match_or_str.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("—", "-").replace("–", "-")
    return s


def sanitize_filename(name: str) -> str:
    """文件名安全字符：去除 macOS/跨平台问题字符。"""
    for ch in INVALID_FILENAME_CHARS:
        name = name.replace(ch, "·")
    name = re.sub(r"[\x00-\x1f]", "", name)
    name = re.sub(r" +", " ", name).strip()
    name = name.strip(". ")
    return name


def strip_book_title_wrappers(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[《\s]+", "", s)
    s = re.sub(r"[》\s]+$", "", s)
    s = s.replace("\u3000", " ").strip()
    s = re.sub(r" +", " ", s)
    return s


def infer_residue_gb_2013(stem: str, body: str) -> str | None:
    """前导序号 + 2013-09 发布批次：GB 29681-2013 … GB 29709-2013（29680+序号）。"""
    if publication_year_from_body(body) != "2013":
        return None
    s = FILENAME_TRAIL_JUNK.sub("", stem).strip()
    m = re.match(r"^(\d{1,2})\s+", s)
    if not m:
        return None
    n = int(m.group(1))
    if not 1 <= n <= 40:
        return None
    return f"{29680 + n}-2013"


def publication_year_from_body(body: str, max_lines: int = 40) -> str | None:
    """正文前几行中的「YYYY-MM-DD 发布」年份。"""
    head = "\n".join(
        body.replace("\r\n", "\n").replace("\r", "\n").split("\n")[:max_lines]
    )
    m = re.search(r"^(\d{4})-\d{2}-\d{2}\s*发布", head, re.MULTILINE)
    return m.group(1) if m else None


def normalize_stem_three_digit_year(stem: str, body: str) -> str:
    """将文件名中 hyphen 后三位错误年份改为正文发布年（如 4789.28-203 → 4789.28-2013）。"""
    pub_y = publication_year_from_body(body)
    if not pub_y:
        return stem

    def repl(mm: re.Match[str]) -> str:
        return f"{mm.group(1)}-{pub_y}"

    return FNAME_TRAILING_3DIGIT_YEAR.sub(repl, stem, count=1)


def extract_gb_from_filename(stem: str) -> tuple[str | None, str]:
    """从文件名解析标准号；返回 (code, 去掉编号后的残余标题线索)."""
    stem = FILENAME_TRAIL_JUNK.sub("", stem).strip()
    stem = FILENAME_LEADING_ENUM.sub("", stem).strip()

    m = FNAME_GB_PAREN_CN.search(stem)
    if m:
        code = normalize_gb_code(m.group(1))
        rest = stem[: m.start()] + stem[m.end() :]
        rest = re.sub(r"^[\s_\-]+|[\s_\-]+$", "", rest)
        return code, rest

    m = FNAME_GB_PREFIX.match(stem)
    if m:
        code = normalize_gb_code(m.group(1))
        rest = stem[m.end() :].lstrip(" _-")
        return code, rest

    m = FNAME_GB_SPACE_YEAR.match(stem)
    if m:
        code = normalize_gb_code(f"{m.group(1)}-{m.group(2)}")
        rest = stem[m.end() :].lstrip(" _-")
        return code, rest

    m = FNAME_GB_NO_SPACE_BEFORE_NUM.match(stem)
    if m:
        code = normalize_gb_code(f"{m.group(1)}-{m.group(2)}")
        rest = stem[m.end() :].lstrip(" _-")
        return code, rest

    return None, stem


def extract_gb_from_body(text: str, max_lines: int = 200) -> str | None:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    scan = lines[:max_lines]
    for line in scan:
        raw = line.strip()
        if BODY_GB_LINE.match(raw):
            m = re.search(rf"GB\s*({GB_CODE_CORE})", raw, re.IGNORECASE)
            if m:
                return normalize_gb_code(m.group(1))
    for line in scan:
        raw = line.strip()
        if "GB" not in raw.upper():
            continue
        if CITATION_LINE_HINTS.search(raw):
            continue
        if raw.count("GB") > 1 and "测定" not in raw and "标准" not in raw[:20]:
            continue
        for m in BODY_GB_COMPACT.finditer(raw):
            cand = normalize_gb_code(m.group(1))
            if re.match(r"^\d+-\d{4}$", cand) or re.match(r"^\d+\.\d+-\d{4}$", cand) or "." in cand:
                if len(raw) < 80 or re.search(
                    rf"GB\s*{re.escape(cand.split('-')[0])}", raw
                ):
                    return cand
            elif re.match(r"^\d{4,5}-\d{4}$", cand):
                if len(raw) < 60:
                    return cand
    # 封底/版权：末 40 行中仅含标准号的独行
    footer = lines[-40:] if len(lines) > 40 else lines
    footer_line = re.compile(
        rf"^GB\s+{GB_CODE_CORE}\s*$",
        re.IGNORECASE,
    )
    for line in reversed(footer):
        raw = line.strip()
        if footer_line.match(raw):
            m = re.search(rf"GB\s*({GB_CODE_CORE})", raw, re.IGNORECASE)
            if m:
                return normalize_gb_code(m.group(1))
    return None


def amendment_from_filename(filename_stem: str) -> tuple[bool, str | None]:
    """仅从文件名识别修改单，避免前言中「代替…及第1号修改单」误报。"""
    m = AMENDMENT_RE.search(filename_stem)
    if m:
        return True, m.group(1)
    return False, None


def title_from_filename_remainder(rest: str, gb_code: str) -> str | None:
    """从去掉 GB 编号后的文件名片段提取全称线索。"""
    if not rest:
        return None
    s = rest.replace(".md", "").strip()
    s = FILENAME_TRAIL_JUNK.sub("", s).strip()
    s = s.replace("\u3000", " ")
    s = re.sub(r" +", " ", s)
    s = strip_book_title_wrappers(s)
    # 《…》（GB…）拆开后，中间可能残留半个书名号
    s = s.replace("《", "").replace("》", "").strip()
    s = re.sub(r" +", " ", s)
    inner = re.match(r"^《([^》]+)》\s*(.*)$", s)
    if inner:
        s = inner.group(1).strip()
        tail = inner.group(2).strip()
        if tail:
            s = f"{s} {tail}".strip()
    s = re.sub(rf"^GB[_\s\-]*{re.escape(gb_code)}\s*", "", s, flags=re.I)
    s = re.sub(r"^[《]?食品安全国家标准[》\s]*", "", s).strip()
    if not s:
        return None
    if not s.startswith("食品安全国家标准"):
        s = "食品安全国家标准 " + s
    return s


def parse_title_from_headings(text: str) -> str | None:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    n = len(lines)
    i = 0
    while i < n:
        m = HEADING_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        content = m.group(1).strip()
        if content.startswith("食品安全国家标准"):
            if content != "食品安全国家标准":
                rest = content[len("食品安全国家标准") :].strip()
                if rest and not TITLE_STOP_HEADINGS.match(rest):
                    return (
                        sanitize_filename(content)
                        if "/" not in content
                        else content.replace("/", "·")
                    )
            parts: list[str] = []
            j = i + 1
            while j < n:
                raw_j = lines[j].strip()
                if not raw_j:
                    j += 1
                    continue
                m2 = HEADING_RE.match(raw_j)
                if not m2:
                    break
                c2 = m2.group(1).strip()
                if TITLE_STOP_HEADINGS.match(c2):
                    break
                if re.match(r"^\d{4}-\d{2}-\d{2}", c2):
                    break
                if c2.startswith("National ") or c2.startswith("Determination "):
                    break
                if re.match(r"^[\d０-９]+[\s\.．]", c2) and "范围" in c2:
                    break
                parts.append(c2)
                j += 1
            if parts:
                merged = "食品安全国家标准 " + " ".join(parts)
                merged = re.sub(r" +", " ", merged)
                return merged
            i = j
            continue
        i += 1
    return None


def ensure_food_safety_prefix(title: str) -> str:
    t = title.strip().replace("\u3000", " ")
    t = re.sub(r" +", " ", t)
    # 「…纤维素第1号修改单」→「…纤维素 第1号修改单」
    t = re.sub(r"([^\s])(第\d+号修改单)", r"\1 \2", t)
    if t.startswith("食品安全国家标准"):
        return t
    return "食品安全国家标准 " + t


def build_target_basename(
    gb_code: str, full_title: str, amendment: bool, amend_n: str | None
) -> str:
    title = ensure_food_safety_prefix(full_title)
    if amendment and amend_n and not AMENDMENT_RE.search(title):
        title = f"{title} 第{amend_n}号修改单"
    base = f"GB {gb_code} {title}"
    return sanitize_filename(base) + ".md"


def load_gb_mapping(path: Path | None) -> dict[str, str]:
    """TSV：source_relative<TAB>gb_code（可含表头 source_relative）。"""
    if path is None or not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            k, v = row[0].strip(), row[1].strip()
            if not k or k.lower() == "source_relative" or k.startswith("#"):
                continue
            out[k] = normalize_gb_code(v)
    return out


def resolve_target(
    path: Path,
    body: str,
    *,
    rel_key: str,
    max_body_lines: int = 200,
    mapping: dict[str, str] | None = None,
) -> tuple[str | None, str | None, bool, str | None, str]:
    """返回 (gb_code, full_title, amendment, amend_n, skip_reason)。"""
    stem_orig = path.stem
    stem_for_gb = normalize_stem_three_digit_year(stem_orig, body)
    code, rest = extract_gb_from_filename(stem_for_gb)

    if not code:
        code = extract_gb_from_body(body, max_lines=max_body_lines)
    if not code and mapping:
        code = mapping.get(rel_key)
    if not code:
        code = infer_residue_gb_2013(stem_orig, body)
    if not code:
        return None, None, False, None, "no_gb_in_filename_or_body"

    title = title_from_filename_remainder(rest, code)
    if not title:
        title = parse_title_from_headings(body)

    if not title:
        return code, None, False, None, "no_title_derived"

    amd, amd_n = amendment_from_filename(stem_orig)
    return code, title, amd, amd_n, ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--src",
        type=Path,
        default=Path("_all_md"),
        help="源目录（递归收集 .md）",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("_all_md_renamed"),
        help="目标目录（复制到此，保持相对路径）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计并写入 --dst 下的 TSV/CSV 清单，不复制 .md（仍会创建 --dst 目录）",
    )
    p.add_argument(
        "--on-collision",
        choices=("skip", "suffix"),
        default="skip",
        help="目标文件名冲突时：跳过第二个或追加 __2、__3",
    )
    p.add_argument(
        "--max-body-lines",
        type=int,
        default=200,
        help="从正文解析 GB 编号时扫描的最大行数",
    )
    p.add_argument(
        "--mapping",
        type=Path,
        default=None,
        metavar="PATH",
        help="TSV：source_relative<TAB>gb_code，仅在文件名与正文均无法解析编号时使用",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    if not src_root.is_dir():
        print(f"源目录不存在: {src_root}", file=sys.stderr)
        return 1

    md_files = sorted(src_root.rglob("*.md"))
    if not md_files:
        print(f"未找到 .md: {src_root}", file=sys.stderr)
        return 1

    gb_mapping = load_gb_mapping(args.mapping.resolve() if args.mapping else None)

    skipped: list[tuple[str, str]] = []
    conflicts: list[tuple[str, str, str]] = []
    log_rows: list[tuple[str, str, str]] = []
    used_targets: dict[str, str] = {}
    ok = 0

    dst_root.mkdir(parents=True, exist_ok=True)

    for path in md_files:
        rel = path.relative_to(src_root)
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            skipped.append((str(rel), f"read_error:{e}"))
            log_rows.append((str(rel), "", f"skip:read_error"))
            continue

        code_t, title_t, amd, amd_n, reason = resolve_target(
            path,
            body,
            rel_key=rel.as_posix(),
            max_body_lines=args.max_body_lines,
            mapping=gb_mapping or None,
        )
        if reason == "no_gb_in_filename_or_body":
            skipped.append((str(rel), reason))
            log_rows.append((str(rel), "", "skip:no_gb"))
            continue
        if reason == "no_title_derived":
            skipped.append((str(rel), reason))
            log_rows.append((str(rel), f"GB {code_t}", "skip:no_title"))
            continue

        assert code_t is not None and title_t is not None
        basename = build_target_basename(code_t, title_t, amd, amd_n)
        target_rel = rel.with_name(basename)
        # 若源在子目录，仍把文件放在 dst 下相同相对路径
        target_path = dst_root / target_rel

        key = target_rel.as_posix()
        if key in used_targets:
            if args.on_collision == "suffix":
                stem = target_path.stem
                parent = target_path.parent
                suf = 2
                while True:
                    cand = parent / f"{stem}__{suf}{target_path.suffix}"
                    k2 = cand.relative_to(dst_root).as_posix()
                    if k2 not in used_targets:
                        target_path = cand
                        key = k2
                        break
                    suf += 1
            else:
                conflicts.append(
                    (str(rel), str(used_targets[key]), basename),
                )
                log_rows.append((str(rel), key, "skip:collision"))
                skipped.append((str(rel), "target_collision"))
                continue

        used_targets[key] = str(rel)

        if not args.dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_path)

        log_rows.append((str(rel), key, "ok"))
        ok += 1

    report_dir = dst_root
    skip_path = report_dir / "skipped_no_gb.tsv"
    conflict_path = report_dir / "rename_conflicts.tsv"
    log_path = report_dir / "rename_log.csv"

    with skip_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source_relative", "reason"])
        w.writerows(skipped)
    with conflict_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source_relative", "conflicts_with_source", "target_basename"])
        w.writerows(conflicts)
    with log_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_relative", "target_relative", "status"])
        w.writerows(log_rows)

    print(
        f"files={len(md_files)} ok={ok} skipped={len(skipped)} "
        f"conflicts={len(conflicts)} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
