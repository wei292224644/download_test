#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 https://sppt.cfsa.net.cn:8086/db 爬取并下载所有国家标准（标准文本）。
步骤：打开页面 → 切换到「标准文本」→ 依次点击左侧筛选标签 → 每个标签下点击「展开全部」→ 检索并下载所有 PDF。
"""

import os
import random
import re
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


BASE_URL = "https://sppt.cfsa.net.cn:8086/db"
DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_GB_DIR", "downloads"))
# 每次下载后的等待时间（秒），配合随机波动，降低被识别为爬虫的概率
DOWNLOAD_DELAY_MIN = 2.0
DOWNLOAD_DELAY_MAX = 5.0
DEFAULT_TIMEOUT = 30000

# 左侧筛选标签，按顺序点击并在每个分类下展开全部后下载
CATEGORIES = [
    "食品添加剂",
    "食品产品",
    "食品相关产品",
    "营养与特殊膳食食品",
    "食品营养强化剂",
]


def sanitize_filename(name: str) -> str:
    """去掉文件名中的非法字符。"""
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip() or "unnamed"


def get_resource_name_from_info_item(button_locator) -> str:
    """
    从 .infoItem 容器内获取资源名称（按钮在 .infoItem > .infoItemRight 下）。
    取同一 .infoItem 的文本，去掉「下载」等按钮文字。
    """
    try:
        parent = button_locator.locator(
            "xpath=ancestor::*[contains(@class,'infoItem')]"
        ).first
        if parent.count() > 0:
            raw = parent.inner_text()
            if raw:
                for strip in ("下载", "展开", ">>", "查看"):
                    raw = raw.replace(strip, " ")
                raw = " ".join(raw.split()).strip()
                if raw and len(raw) > 1:
                    return raw[:200]
    except Exception:
        pass
    return ""


def get_download_infos(page):
    """
    获取当前页面上所有下载按钮的 (key, locator, resource_name) 列表。
    下载按钮结构：.infoItem > .infoItemRight 下的 <button>下载</button>。
    若 .infoItem > .infoItemCenter 中包含 "-- 请输入标准名称 --"，则该下载按钮无效，跳过。
    """
    selector = ".infoItem > .infoItemRight button:has-text('下载')"
    result = []
    try:
        loc = page.locator(selector)
        n = loc.count()
        for i in range(n):
            node = loc.nth(i)
            try:
                # 若同一条 .infoItem 下的 .infoItemCenter 含 "-- 请输入标准名称 --"，则为无效项，跳过
                info_item = node.locator("xpath=ancestor::*[contains(@class,'infoItem')]").first
                if info_item.count() > 0:
                    center = info_item.locator(".infoItemCenter").first
                    if center.count() > 0:
                        center_text = center.inner_text() or ""
                        if "-- 请输入标准名称 --" in center_text:
                            continue
                key = (selector, i)
                resource_name = get_resource_name_from_info_item(node)
                result.append((key, node, resource_name))
            except Exception:
                continue
    except Exception:
        pass
    return result


def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载目录: {DOWNLOAD_DIR.absolute()}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            accept_downloads=True,
            ignore_https_errors=True,
        )
        context.set_default_timeout(DEFAULT_TIMEOUT)
        page = context.new_page()

        try:
            print("正在打开页面...")
            page.goto(BASE_URL, wait_until="networkidle")
            time.sleep(1)

            print("切换到「标准文本」...")
            tab_std = page.get_by_text("标准文本", exact=False).first
            tab_std.wait_for(state="visible")
            tab_std.click()
            time.sleep(1.5)

            total_success = 0
            total_count = 0
            total_skipped = 0
            # 跨分类去重：同一文件可能出现在多个标签下，已下载过的文件名则跳过
            # 启动时扫描 downloads 目录，将之前已下载的文件名加入集合，避免重复下载
            downloaded_names = set()
            if DOWNLOAD_DIR.exists():
                for p in DOWNLOAD_DIR.rglob("*"):
                    if p.is_file():
                        downloaded_names.add(p.name)
            if downloaded_names:
                print(f"已扫描到 {len(downloaded_names)} 个已存在文件，将跳过重复下载")

            for cat_index, category in enumerate(CATEGORIES):
                print(f"\n[{cat_index + 1}/{len(CATEGORIES)}] 切换到左侧筛选「{category}」...")
                cat_btn = page.locator(".infoContent").get_by_text(category, exact=True).first
                try:
                    cat_btn.wait_for(state="visible", timeout=5000)
                    cat_btn.click()
                except Exception as e:
                    print(f"  无法点击「{category}」: {e}，跳过该分类")
                    continue
                time.sleep(1.5)

                print("  点击「展开全部>>」...")
                expand_btn = page.get_by_text("展开全部", exact=False).first
                try:
                    expand_btn.wait_for(state="visible", timeout=5000)
                    expand_btn.click()
                except Exception as e:
                    print(f"  未找到或无法点击「展开全部」: {e}，跳过该分类")
                    continue
                # 等待 5 秒，确保服务器返回并渲染完列表
                print("  等待 5 秒，确保服务器返回并渲染完列表")
                time.sleep(5)

                download_infos = get_download_infos(page)
                total = len(download_infos)
                print(f"  共找到 {total} 个下载项，开始下载...")

                if total == 0:
                    continue

                # 按分类建子目录存放
                category_dir = DOWNLOAD_DIR / sanitize_filename(category)
                category_dir.mkdir(parents=True, exist_ok=True)

                success = 0
                for i, (key, handle, resource_name) in enumerate(download_infos):
                    try:
                        with page.expect_download(timeout=60000) as download_info:
                            handle.click()
                        download = download_info.value
                        # 优先使用服务器返回的文件名（与手动下载一致，如 GB 1886.386-2025 食品安全国家标准 食品添加剂 碳酸铵.pdf）
                        suggested = download.suggested_filename or ""
                        if suggested and suggested.strip():
                            safe_name = sanitize_filename(suggested.strip())
                        elif resource_name:
                            ext = Path(suggested).suffix if suggested else ".pdf"
                            base = sanitize_filename(resource_name)
                            safe_name = base + ext if not base.lower().endswith(ext.lower()) else base
                        else:
                            safe_name = sanitize_filename(suggested) or f"file_{i+1}.pdf"

                        # 跨分类去重：已在其他标签下下载过的同名文件则跳过
                        if safe_name in downloaded_names:
                            with tempfile.NamedTemporaryFile(
                                delete=True, suffix=Path(safe_name).suffix
                            ) as tmp:
                                download.save_as(tmp.name)
                            total_skipped += 1
                            print(f"    [{i+1}/{total}] 已存在，跳过: {safe_name}")
                        else:
                            path = category_dir / safe_name
                            if path.exists():
                                stem, suf = path.stem, path.suffix
                                k = 1
                                while path.exists():
                                    path = category_dir / f"{stem}_{k}{suf}"
                                    k += 1
                            download.save_as(path)
                            downloaded_names.add(safe_name)
                            success += 1
                            total_success += 1
                            print(f"    [{i+1}/{total}] 已保存: {path.name}")
                    except PlaywrightTimeout:
                        print(f"    [{i+1}/{total}] 超时，跳过")
                    except Exception as e:
                        print(f"    [{i+1}/{total}] 失败: {e}")
                    delay = random.uniform(DOWNLOAD_DELAY_MIN, DOWNLOAD_DELAY_MAX)
                    time.sleep(delay)
                total_count += total
                print(f"  分类「{category}」完成，成功 {success}/{total}")

            print(f"\n全部完成。成功下载 {total_success} 个，跳过重复 {total_skipped} 个，保存到: {DOWNLOAD_DIR.absolute()}")

        except Exception as e:
            print(f"运行出错: {e}")
            raise
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    main()
