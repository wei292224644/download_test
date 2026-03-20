# 国家标准下载爬虫

从 **https://sppt.cfsa.net.cn:8086/db** 自动切换至「标准文本」、展开全部后，轮询所有下载按钮并保存到本地。

## 步骤说明

1. 打开上述网址（默认在「最新公告」）
2. 点击 **「标准文本」** 标签
3. 点击 **「展开全部>>」**
4. 查找页面上所有下载链接/按钮，逐个点击并保存文件

## 环境要求

- 使用 **uv** 做 Python 版本与依赖管理（见 [uv](https://docs.astral.sh/uv/)）
- 项目锁定 Python 3.12（`.python-version`），需 Playwright（Chromium）

## 安装与运行

```bash
# 进入项目目录
cd /Users/wwj/Desktop/self/download_gb

# 使用 uv 创建虚拟环境并安装依赖（会按 .python-version 使用 Python 3.12）
uv sync

# 安装 Playwright 浏览器（仅首次需要）
uv run playwright install chromium

# 运行爬虫（默认有界面浏览器，便于观察）
uv run python download_gb.py
```

## 配置

- **Python 版本**：当前由 `.python-version` 指定为 3.12。修改版本可执行 `uv python pin 3.11` 等，或直接编辑 `.python-version` 后重新 `uv sync`。
- **下载目录**：默认 `./downloads`，可通过环境变量覆盖：
  ```bash
  export DOWNLOAD_GB_DIR=/path/to/save
  python download_gb.py
  ```
- **无头模式**：在 `download_gb.py` 中将 `headless=False` 改为 `headless=True` 可后台运行（不弹出浏览器窗口）。
- **请求间隔**：脚本内 `DOWNLOAD_DELAY = 1.0`（秒），可按需修改，避免请求过快。

## 若未找到下载按钮

若控制台提示「共找到 0 个下载项」，说明页面上的选择器与当前网站结构不一致。可以：

1. 保持 `headless=False`，运行脚本观察页面是否成功切换到「标准文本」并展开全部。
2. 在浏览器中按 F12 打开开发者工具，在「标准文本」展开后查看下载链接的 HTML（标签名、class、文字等）。
3. 在 `download_gb.py` 的 `get_download_infos()` 中，根据实际 DOM 增改 `selectors` 列表中的选择器（或增加新的 `page.get_by_text(...)` / `page.locator(...)` 逻辑）。

## 整理 MD 标题结构（reorganize_md.py）

使用 LLM（Ollama 优先，不可用时回退到 OpenAI）整理 Markdown 的**标题层级与顺序**，仅改结构、不改正文，结果写入新文件（默认 `xxx.reorganized.md`）。

### 用法示例

```bash
# 仅列出将要处理的文件（不调用 LLM）
uv run python reorganize_md.py --dir output/ --dry-run

# 处理 output/ 下所有 .md（默认目录）
uv run python reorganize_md.py

# 只处理单个文件
uv run python reorganize_md.py --file "output/某文档/vlm/某文档.md"

# 指定输出目录、后缀、超时
uv run python reorganize_md.py --dir output/ --out-suffix ".reorganized" --out-dir ./out --timeout 120
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `OLLAMA_BASE_URL` | Ollama API 地址，默认 `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Ollama 模型名，默认 `llama3.2` |
| `OPENAI_API_KEY` | OpenAI 或兼容 API 的 Key（Ollama 失败时使用） |
| `OPENAI_BASE_URL` | 可选，兼容 API 的 base URL |
| `OPENAI_MODEL` | 可选，兼容 API 的模型名，默认 `gpt-4o-mini` |

正文保护：若 LLM 改动了非标题内容，脚本会拒绝写入并记入失败数。

---

## 注意

- 请遵守网站使用条款与 robots 规则，仅用于个人学习或合规用途。
- 若网站需要登录或验证码，当前脚本未做处理，需自行扩展。
