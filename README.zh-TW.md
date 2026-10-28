<div align="center">

# NASDAQ 當日漲幅冠軍 Agent

**一個自主運作的 LangGraph Agent：找出當日 NASDAQ 漲幅最大的股票，自行撰寫並執行分析程式碼，再以電子郵件寄出報告；報告中的每一個數字都經過獨立重新計算。**

<p>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/agent-LangGraph-1C3C3C">
  <img alt="Docker sandbox" src="https://img.shields.io/badge/code%20execution-Docker%20sandbox-2496ED?logo=docker&logoColor=white">
  <img alt="Next.js dashboard" src="https://img.shields.io/badge/dashboard-Next.js-000000?logo=nextdotjs&logoColor=white">
</p>

<p>
  <a href="README.md">English</a> · <b>繁體中文</b>
</p>

<p>
  <a href="#五分鐘快速審閱">快速審閱</a> ·
  <a href="#運作方式">運作方式</a> ·
  <a href="#產品架構">產品架構</a> ·
  <a href="#報告如何產生">報告設計</a> ·
  <a href="#資料檢查與排名狀態">資料檢查</a> ·
  <a href="#安裝">安裝</a> ·
  <a href="#執行">執行</a> ·
  <a href="#測試與評測">測試</a>
</p>

<img src="docs/agent-chart.png" width="640" alt="AIOK 與標普 500、NASDAQ 綜合指數近五個交易日的走勢，以 100 為基準">
<br><sub>由 Agent 自行撰寫的程式碼繪製：2026 年 10 月 23 日篩選後漲幅第一的 AIOK。</sub>

</div>

---

## 五分鐘快速審閱

<table>
<tr><td width="150"><b>功能</b></td><td>一個 LangGraph Agent 搭配七個工具：擬定計畫、篩選漲幅榜、取得股價與新聞、撰寫並執行分析程式碼、評估新聞情緒、寄送郵件。由模型決定呼叫順序與參數；由程式碼核對每一個數字，並控管郵件的寄送。完整設計請見 <a href="REPORT.md">REPORT.md</a>（英文）。</td></tr>
<tr><td><b>從哪裡看起</b></td><td><a href="nasdaq_agent/agent/graph.py"><code>graph.py</code></a>：Agent 主迴圈、寄送前檢查、備援流程<br><a href="nasdaq_agent/tools/gainers.py"><code>gainers.py</code></a> + <a href="nasdaq_agent/data/ranking.py"><code>ranking.py</code></a>：篩選與排名狀態<br><a href="nasdaq_agent/tools/code_exec.py"><code>code_exec.py</code></a> + <a href="nasdaq_agent/sandbox/runner.py"><code>sandbox/runner.py</code></a>：執行模型撰寫的程式碼<br><a href="nasdaq_agent/reporting/evidence_report.py"><code>evidence_report.py</code></a> + <a href="nasdaq_agent/reporting/readable.py"><code>readable.py</code></a>：報告正文</td></tr>
<tr><td><b>報告的主張範圍</b></td><td>報告指出「<i>經核實的第一名候選</i>」並標示排名狀態。「已核實」代表沒有任何可能排名更前、卻仍無法確認的榜單資料；這不代表上游資料絕無錯誤。第二家資料商的價格一致，只能支持價格正確，不能證明排名。</td></tr>
</table>

```powershell
uv sync --extra jev
docker build -t nasdaq-analysis:local -f nasdaq_agent/sandbox/Dockerfile .
uv run python main.py          # 預設為 dry-run：郵件只存檔，不寄出
```

> [!NOTE]
> 沒有 Docker 時流程仍會完成：模型撰寫的程式碼不會被執行，郵件改為由固定的參考實作計算出的降級報告。

## 運作方式

<p align="center"><img src="docs/diagrams/agent-flow.svg" width="900" alt="Agent 流程：LLM Agent 呼叫七個工具；execute_python 在 Docker 沙盒中執行；send_email 須通過寄送前檢查，否則退回 Agent；達到上限或服務中斷時進入確定性備援"></p>
<p align="center"><sub>圖中文字為英文，與程式碼中的名稱一致。</sub></p>

| 工具 | 作用 |
|---|---|
| `record_plan` | 記錄或修訂計畫（可稽核，不強制執行） |
| `get_top_nasdaq_gainers` | Yahoo 漲幅榜，以 Nasdaq.com 備援；回傳帶日期與標記的候選、未篩選的第一名，以及排名狀態 |
| `get_price_history` | 經驗證的個股與指數收盤價，存成程式碼可讀取的資料集 |
| `execute_python` | 在 Docker 沙盒中執行模型撰寫的程式碼，並獨立重算每一個結果 |
| `get_news` / `analyze_sentiment` | 帶編號的新聞標題；情緒分數標示為模型估計 |
| `send_email` | 接收證據編號與 2–4 條依重要性排序的關鍵結論；由程式碼撰寫並寄出報告 |

### 產品架構

<p align="center"><img src="docs/diagrams/architecture.svg" width="900" alt="產品架構：命令列與儀表板入口、FastAPI 與工作佇列、執行器與 LangGraph Agent、外部資料源、模型、Docker 沙盒、郵件，以及本機儲存"></p>

| 區塊 | 說明 |
|---|---|
| 入口 | 命令列 `python main.py`；選用的 Next.js 儀表板 |
| 服務 | FastAPI（`nasdaq_agent/api.py`）與 `jobs.sqlite` 工作佇列，支援租約與續跑 |
| Agent 執行環境 | 執行器、LangGraph Agent、`checkpoints.sqlite` 檢查點 |
| 外部服務 | Yahoo Finance（主要）、Nasdaq.com（備援）、Google News RSS；MiMo（主要）、DeepSeek（備援）、TypeSafe Jev（選用）；Docker 沙盒；SMTP 郵件 |
| 本機儲存 | `run_*/` 執行紀錄、`delivery.sqlite` 寄送紀錄、`usage.sqlite` 模型用量、`.cache/` 資料快取 |

## 報告如何產生

<p align="center"><img src="docs/diagrams/report-design.svg" width="900" alt="報告設計：三種方案比較，以及最終方案從 Agent 選擇、驗證、渲染到產出報告的流程"></p>

由模型決定**哪些事實最重要**；由程式碼根據已核實的資料寫出**每一句話**。報告開頭是一句話摘要與 Agent 選出的關鍵結論，例如（報告本身為英文）：

> **In brief.** AIOK jumped 183% on Oct 23, with volume about 880 times its recent average.
>
> **Key findings.**
> 1. Almost all of the 5-session gain came on Oct 23, when the stock rose 183%; the 4 earlier sessions together added 1.2%.
> 2. Oct 23 volume was about 880 times the previous 5-session average of 48.6K shares a day — an extreme spike in trading.
> 3. GlobeNewswire published the headline: "AI OKTO Corp. Declares Special Dividend of $3.50 Per Share".
> 4. Over the same dates the S&P 500 fell 0.47% and the NASDAQ Composite fell 0.71%; AIOK's 186% 5-session gain was much larger than either index's move.

用字依數值的固定門檻選擇：例如最後一天占比達 90% 以上才寫「almost all」（幾乎全部），與大盤差距達 10 個百分點以上才寫「well ahead of」（大幅領先）。未被選為重點的事實仍保留在正文中；若排名、篩選資格或價格核對有任何未確認，最主要的限制必須列為其中一條結論。契約細節請見 [docs/evidence-report.md](docs/evidence-report.md)（英文）。

### 為什麼這樣設計：三種方案比較

| | 1. 模型自由撰寫正文 | 2. 全部由程式碼產生 | **3. 最終方案：模型選重點，程式碼寫句子** |
|---|---|---|---|
| 模型的角色 | 撰寫全部文字 | 只挑選新聞編號 | 挑選新聞，並選出 2–4 條關鍵結論與排序（也決定一句話摘要的內容） |
| 事實可靠性 | ❌ 實測未通過 | ✅ 每句話都能對應回資料 | ✅ 與方案 2 相同 |
| 可讀性 | ✅ 最自然 | ❌ 像技術說明、逐條羅列 | ⚠️ 接近分析師短評，但句型有限 |
| 能否看出 Agent 的判斷 | ✅ 最明顯 | ❌ 幾乎看不出 | ✅ 看得出哪些事實最重要、順序如何 |
| 能否隱藏不利資訊 | 能（不寫即可） | 不能 | 不能：未選的事實仍在正文中，且必須列出主要限制 |
| 寄送前檢查 | 需另一個模型判斷是否有依據，結果不確定 | 確定性規則 | 確定性規則，外加結論相關的檢查 |
| 驗證成本 | 高（盲評、人工複核） | 低（單元測試） | 低到中（單元測試加少量真實模型測試） |

| 依據 | 結果 |
|---|---|
| 1. 模型自由撰寫 | 抽查三份已放行的報告，兩份含無依據的說法（例如實際並非如此的「全部漲幅」、臆測的動機）；24 份盲評報告只有 12 份能可靠判斷，驗收未通過 |
| 2. 全部由程式碼產生 | 兩個模型各通過 4 個情境，沒有因措辭被退回；相同證據產生完全相同的正文 |
| 3. 最終方案 | 8/8 個合成情境都採用 Agent 自選的結論（兩次初選被退回後一次修正）；在 10 月 23 日的真實資料上，兩個模型都主動選出「成交量放大 876 倍」 |

<details>
<summary><b>最終方案的取捨</b></summary>

| 得到 | 付出 |
|---|---|
| 報告不會出現無依據的說法，保證與方案 2 相同 | 句型來自範本，不如人寫的靈活 |
| 一句話摘要與結論順序反映 Agent 的判斷 | 模型對這類參數不熟悉，需要相容多種寫法並以錯誤訊息引導，偶爾多一輪呼叫 |
| 讀起來像分析師短評：摘要、重點、細節 | 用字門檻（「幾乎全部」「大幅領先」）是工程設定，尚未校準 |
| 關鍵指標與限制不會被省略 | 正文接近 250 字上限，篇幅不足時自動精簡 |
| 完全確定性，單元測試即可驗證 | 合成情境中選擇較為相似，樣本太少，無法判斷原因 |

</details>

## 資料檢查與排名狀態

- **報價日期。** Yahoo 漲幅榜的每一列都必須是交易當日的報價；較早交易日的報價會被排除，並列在警告中。Nasdaq.com 的資料沒有時間戳記，因此逐檔與該股自己的日線比對：價格與漲跌幅都符合交易當日才算「最新」，都符合前一交易日才算「過期」，其餘一律「無法確認」。
- **篩選門檻邊界。** 價格或成交量落在門檻 ±10% 內的股票，先以日線資料判定，再決定是否淘汰；若來源之間不一致，該股保留但標示「資格未確認」。10% 是尚未校準的工程預設值（`data_checks.filter_boundary_band`）。
- **排名狀態。** 沒有任何可能排名更前的資料尚待確認時為 `verified`（已核實）；否則為 `uncertain`（報告會說明這不是確認的第一名，並列出相關股票）；盤中為 `provisional_intraday`（暫定）。
- **價格一致性。** 以另一家資料商核對所選股票的收盤價與漲跌幅（`consistent`、`single_source`、`conflict`）。資料商可能共用上游來源，因此一致只代表資料相符，不代表獨立證明。
- **請求預算。** 每次篩選的逐檔核對最多 12 個請求、25 秒，並使用快取。

<details>
<summary><b>產品預設值</b></summary>

產生報告的對話框預設勾選兩個篩選條件：排除股價低於 1 美元的股票，以及排除當日成交量低於 10 萬股的股票，兩者都可取消。預設範圍排除認股權證、權利、單位與特別股。報告會標示為「**篩選後 NASDAQ 漲幅第一**」並列出套用的條件；模型無法更改篩選範圍或分析天數。歷史資料不足會列為限制，若改分析其他股票則標示為補充分析。

</details>

## 安裝

需要 Python 3.11 以上、[uv](https://docs.astral.sh/uv/)、支援 Linux 容器的 Docker、相容 OpenAI 介面的模型金鑰；若要使用儀表板，另需 Node.js 20 以上。

```powershell
uv sync --extra ui          # 儀表板後端，已包含 Jev
Copy-Item .env.example .env
docker build -t nasdaq-analysis:local -f nasdaq_agent/sandbox/Dockerfile .
```

在 `.env` 填入模型憑證；只有真正寄信時才需要 SMTP 憑證。業務設定位於 [config.yaml](config.yaml)。

> [!TIP]
> **建議安裝 Jev（TypeSafe）** 用於新聞情緒分析。`--extra ui` 已包含，儀表板預設啟用；只使用命令列時請執行 `uv sync --extra jev` 並設定 `TYPESAFE_API_KEY`。若未安裝，情緒分數改由聊天模型產生，產生報告的對話框會顯示目前使用的方式。

<details>
<summary><b>沙盒細節與 Docker 鏡像</b></summary>

沙盒無網路、根目錄與輸入檔唯讀、以非 root 使用者執行，並限制 CPU、記憶體、行程數、輸出與執行時間，輸出存放在 32 MiB 的暫存檔案系統。若未安裝 Docker、服務未啟動或映像檔尚未建置，Agent 會被告知程式碼未執行，流程直接改用參考計算。`--sandbox local`（或 `SANDBOX_BACKEND=trusted_local`）僅供開發除錯：它以你的使用者權限執行模型撰寫的程式碼，不能證明隔離效果。

若無法使用 Docker Hub，可改用公開的 ECR 鏡像：

```powershell
docker build -t nasdaq-analysis:local --build-arg PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.13-slim-bookworm -f nasdaq_agent/sandbox/Dockerfile .
```

</details>

## 執行

```powershell
uv run python main.py           # dry-run（預設）：存於 outputs/run_*/email.eml
uv run python main.py --send    # 真正寄出；需要 EMAIL_TO 與 SMTP 憑證
```

| 參數 | 作用 |
|---|---|
| `--days 10` | 分析 10 個交易日 |
| `--provider deepseek` | 更換主要模型，保留備援順序 |
| `--to a@example.com,b@example.com` | 覆寫收件人 |
| `--no-jev` | 改用聊天模型評估情緒，不使用 Jev |
| `--sandbox local` | 僅供開發：在無容器隔離的情況下執行程式碼 |
| `--resume <run_id>` | 從中斷的檢查點繼續 |
| `--quiet` | 只輸出最後的摘要 |

結束代碼：**0** 完成、**2** 降級／備援、**1** 失敗。每次執行都會在 `outputs/` 下保存執行軌跡、各版程式碼、清理後的資料、圖表、郵件與摘要。

<details>
<summary><b>選用：儀表板與後端</b></summary>

```powershell
uv run --extra ui uvicorn nasdaq_agent.api:app --host 127.0.0.1 --port 8000
```

在另一個終端機中：

```powershell
cd web
npm ci
npm run dev
```

開啟 http://localhost:3000（`NEXT_PUBLIC_API_URL` 可更改 API 位址）。排隊中的工作、租約與檢查點在重新啟動後仍會保留；介面事件可透過 SSE 以 `Last-Event-ID` 重播。寄信對話框最多可填五位收件人；SQLite 寄送紀錄可防止重試或續跑時重複寄出，SMTP 結果不明時會阻擋自動重寄。

本機使用請保留預設的回送位址綁定。遠端部署請設定 `NASDAQ_API_TOKEN` 並帶上 `Authorization: Bearer <token>`；請見 [Docker Compose 與 Coolify 部署說明](docs/deployment.md)（英文）。多使用者的租戶隔離不在此原型的範圍內。

</details>

## 測試與評測

```powershell
uv run --extra ui pytest -q                                   # 離線測試：不需金鑰與網路
$env:RUN_DOCKER_TESTS = '1'; uv run --extra ui pytest tests/test_docker_sandbox.py -q   # 需要已建置的映像檔
```

離線測試（356 項）不會讀取 `.env`：它設定 `NASDAQ_ENV_FILE=none`、移除環境中已匯出的憑證並改用預留值，且設定物件的 `repr` 不包含郵件地址，測試失敗時也不會印出。回歸測試包含 10 月 23 日實際記錄的漲幅榜與日線資料。GitHub Actions 會執行離線測試、資料契約檢查、Docker 隔離測試與前端建置。

<details>
<summary><b>合成、真實模型與歷史評測</b></summary>

40 個合成市場情境涵蓋上漲、下跌、盤整、高波動、單日暴漲、IPO 歷史不足、主要資料源過期與無新聞等情況；校準與測試分組使用互不重疊的日期與代號。

```powershell
uv run python scripts/eval_agent.py --mode contract --split test                    # 資料契約
uv run python scripts/eval_agent.py --mode live --split test --limit 3              # 真實模型，dry-run
uv run python scripts/eval_historical.py --output outputs/eval/historical-contracts.json
```

真實模型評測逐一使用各模型、不跨模型備援，每個情境完成後即存檔；資料、程式碼、提示詞或設定的指紋改變時拒絕續跑。六段 2022–2024 年的歷史記錄保留原始 Yahoo 回應與獨立計算的標準答案；由於當時未記錄全市場排名，只能測試選股後的分析階段（[範圍說明](tests/fixtures/historical/README.md)，英文）。產生的評測結果保存在本機 `outputs/`，不納入版本控制。

舊的自由撰寫檢查器保留在 `scripts/legacy/` 供稽核：它曾放行無依據的因果、比較與「涵蓋全部」類說法，這正是目前報告契約存在的原因。

</details>

## 專案結構

```text
nasdaq_agent/agent/       Agent 圖、狀態、快照驗證、寄送前檢查、備援、執行器
nasdaq_agent/tools/       七個型別化工具
nasdaq_agent/data/        交易日曆、價格驗證、排名狀態、參考指標
nasdaq_agent/providers/   Yahoo、Nasdaq.com、Google News
nasdaq_agent/sandbox/     AST 檢查、Docker 執行器與映像檔
nasdaq_agent/infra/       HTTP、快取、熔斷、工作佇列、寄送紀錄、郵件、追蹤
nasdaq_agent/reporting/   報告契約、可讀措辭、郵件渲染、圖表
web/                      Next.js 儀表板
scripts/                  評測腳本（舊檢查器位於 scripts/legacy/）
tests/                    離線測試與記錄的測試資料
```

<div align="center"><sub>公開資料端點皆搭配快取、速率限制、逾時與備援使用，不會繞過反機器人驗證。本專案僅供參考，不構成投資建議。</sub></div>
