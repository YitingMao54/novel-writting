# novel-agent — 百万字级 AI 长篇小说写作 Agent

一个面向**超长篇连载小说**（百万字级别）的自动写作 Agent。通过「三级规划 + 分层记忆 + 一致性审查」的流水线，突破单次模型上下文的限制，持续产出剧情连贯、人物不崩的长篇小说。

支持 **DeepSeek、通义千问、Kimi、智谱 GLM、MiniMax、混元、豆包、硅基流动** 等任意 OpenAI 兼容的国产大模型，不同环节可路由到不同厂商，兼顾质量与成本。

## 核心设计：为什么能写百万字？

任何模型都装不下百万字上下文，本项目的答案是**分层记忆 + 局部上下文组装**：

```
全书大纲（10卷）        ← 规划一次，定主线，保证"不写歪"
   └── 卷大纲（每卷40章）  ← 每卷规划一次，保证"节奏不崩"
          └── 章细纲（3-5场景）← 写前现场生成，融合最新记忆
                 └── 场景正文   ← 逐场景生成，流式输出
```

写第 N 章时，自动组装一份紧凑且强相关的上下文：

| 记忆层 | 内容 | 更新时机 |
|---|---|---|
| 静态层 | 世界观设定、人物卡（含**状态增量更新**） | 每章写完自动合并 |
| 滚动层 | 前情提要：早期卷用卷级摘要，近 8 章用逐章摘要 | 每章写完自动生成 |
| 线索层 | 伏笔台账：谁在第几章埋的、是否已回收 | 每章写完自动识别 |
| 衔接层 | 上一章结尾原文（默认 600 字） | 自动携带 |

单章流水线：**细纲 → 逐场景生成 → 拼接 → 一致性审查（LLM 编辑）→ 定点修订 → 记忆回写 → 定稿**。

## 安装

```bash
pip install -r requirements.txt
# 或安装为命令行工具
pip install .
```

要求 Python ≥ 3.10。

## 快速开始

```bash
# 1. 配置 API Key（按需，用哪家配哪家）
export DEEPSEEK_API_KEY=sk-xxx        # DeepSeek
export DASHSCOPE_API_KEY=sk-xxx       # 阿里云百炼·通义千问
export MOONSHOT_API_KEY=sk-xxx        # Kimi
export ZHIPU_API_KEY=xxx              # 智谱 GLM

# 2. 初始化项目
novel-agent init ~/novels/my-book --title "剑指苍穹" --genre "东方玄幻"

# 3. 写世界观设定（也可以直接编辑项目内文件后导入）
novel-agent bible set-world ~/novels/my-book "九州大陆，灵气源自天外坠落的九座天碑……"

# 4. 建主要人物卡
novel-agent character add ~/novels/my-book 林昭 \
  "姓名：林昭 / 身份：将门遗孤 / 性格：外冷内热 / 背景：家族蒙冤…"

# 5. 三级规划：全书大纲 → 第1卷大纲 → 第1章细纲（自动）
novel-agent plan book ~/novels/my-book
novel-agent plan volume ~/novels/my-book 1

# 6. 开写！单章 / 批量均可
novel-agent write ~/novels/my-book 1          # 写第1章（流式打印）
novel-agent write ~/novels/my-book 1 40       # 连写第1-40章（断点续写）

# 7. 查看 / 导出
novel-agent status ~/novels/my-book
novel-agent export ~/novels/my-book --format md
```

> 没有全局安装时，用 `python -m novel_agent <命令>` 等价执行。

## Web 工作台（图形界面）

不想敲命令行？启动本地 Web 工作台，在浏览器里完成整个创作流程：

```bash
novel-agent web ~/novels/my-book --port 8765
# 打开 http://127.0.0.1:8765
```

界面包含五个工作区：

| 工作区 | 功能 |
|---|---|
| **设定** | 世界观 / 文风 / 题材编辑，人物卡可视化管理（新建 / 编辑） |
| **大纲** | 一键生成全书大纲、卷大纲；分卷树浏览全部章节梗概 |
| **写作台** | 单章 / 批量写作，正文**实时流式显示** + 流程日志，审查开关 |
| **阅读** | 书页式章节阅读器，展示审查评分与问题批注 |
| **伏笔·用量** | 伏笔台账（埋设/回收状态）、按角色的 Token 用量统计 |

左侧边栏为卷分组章节树（状态圆点：灰=有梗概 / 蓝=有细纲 / 绿=已定稿）与全书进度条。

- 流式输出基于 SSE，无需刷新即可看到逐字生成；
- 写作期间关闭页面任务会在后台继续完成并写入数据库；
- 仅使用 Python 标准库 `http.server`，不引入任何 Web 框架依赖，默认仅监听本机 127.0.0.1。

## 模型路由配置（config.yaml）

不同任务环节自动路由到不同厂商/模型，这是控制百万字成本的关键：

```yaml
roles:
  planner:     # 大纲/细纲：需要强结构化与推理
    provider: deepseek
    model: deepseek-v4-flash
  writer:      # 正文：重文风，温度调高
    provider: deepseek
    model: deepseek-v4-flash
    temperature: 0.9
  reviewer:    # 一致性审查：低随机性
    provider: qwen
    model: qwen3.7-plus
  summarizer:  # 摘要/记忆提取：走最便宜档
    provider: qwen
    model: qwen3.7-flash

writing:
  chapter_words: 3000      # 每章目标字数
  chapters_per_volume: 40  # 每卷章数（40×3000 = 12万字/卷）
  volume_count: 10         # 卷数（10卷 ≈ 120万字）
  review_before_final: true
```

### 内置厂商预设

| provider | base_url | api_key 环境变量 | 推荐模型（2026-09） |
|---|---|---|---|
| `deepseek` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | `deepseek-v4-flash`（1M 上下文）、`deepseek-v4-pro` |
| `qwen` | `dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | `qwen3.8-max` / `qwen3.7-plus` / `qwen3.7-flash` |
| `kimi` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` | `kimi-k3`（1M 上下文） |
| `zhipu` | `open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` | `glm-5.2` / `glm-5.3-flash`（超低价） |
| `minimax` | `https://api.minimaxi.com/v1` | `MINIMAX_API_KEY` | MiniMax-Text 系列 |
| `hunyuan` | `api.hunyuan.cloud.tencent.com/v1` | `HUNYUAN_API_KEY` | `hunyuan-turbos` |
| `doubao` | `ark.cn-beijing.volces.com/api/v3` | `ARK_API_KEY` | 豆包 / 方舟接入点 ID |
| `siliconflow` | `https://api.siliconflow.cn/v1` | `SILICONFLOW_API_KEY` | DeepSeek/Qwen/GLM 开源模型聚合 |

> ⚠️ DeepSeek 旧模型名 `deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 退役，请使用 `deepseek-v4-flash` / `deepseek-v4-pro`；Kimi 旧模型 `moonshot-v1-*`、`kimi-k2.5` 已于 2026-08-31 下线，请使用 `kimi-k3`。
>
> 任意 OpenAI 兼容端点：`provider: custom` + 直接填 `base_url` / `api_key`。

## 全部命令

| 命令 | 说明 |
|---|---|
| `init <dir> [--title] [--genre]` | 初始化项目（生成 config.yaml + 数据库） |
| `bible show / set-title / set-genre / set-world / set-style` | 世界观设定管理 |
| `character add/list/show` | 人物卡管理（支持多行输入） |
| `plan book` | 生成全书大纲（主题/主线/分卷） |
| `plan volume <n>` | 生成第 n 卷大纲（逐章梗概自动落库） |
| `plan chapter <n>` | 生成第 n 章细纲（场景级） |
| `plan outline-for <from> <to>` | 批量预生成细纲 |
| `write <n>` | 写第 n 章（完整流水线，流式输出） |
| `write <from> <to>` | 批量连写，断点续写（已定稿自动跳过） |
| `review <n>` / `revise <n>` | 手动审查 / 按审查意见修订 |
| `volume-summary <n>` | 聚合生成卷级摘要（压缩早期记忆） |
| `status` / `usage` / `export` | 进度 / Token 用量 / 导出 md·txt |
| `web <dir> [--port 8765]` | 启动本地 Web 工作台（见上文） |

## 百万字工程实践建议

1. **节奏**：默认 10 卷 × 40 章 × 3000 字 ≈ 120 万字，可在 config.yaml 调整。
2. **成本**：`summarizer`/`reviewer` 用低价模型（如 `qwen3.7-flash`、`glm-5.3-flash`），`writer` 用主力模型；写作时大约每章 3-6 次调用。
3. **记忆管理**：每写完一卷跑一次 `volume-summary`，早期逐章摘要即可被卷摘要替代，上下文永远可控。
4. **质量控制**：审查发现 `high/medium` 问题会自动定点修订；也可随时手动 `review`/`revise`。重大剧情拐点建议人工过目细纲（`plan chapter` 后检查）再 `write`。
5. **断点续写**：所有状态在 SQLite（`novel.db`），中断后重跑 `write <from> <to>` 自动从断点继续；备份项目目录即备份全书。
6. **伏笔**：写细纲时 Agent 会自动带上"未回收伏笔清单"提示回收；`bible` / 人物卡均可用 LLM 之外的纯文本手工干预。

## 项目结构

```
novel_agent/
├── cli.py        # 命令行入口
├── config.py     # 厂商预设 + 角色路由
├── llm.py        # OpenAI 兼容客户端（流式/重试/Mock）
├── store.py      # SQLite 存储（卷/章/人物/伏笔/时间线/用量）
├── memory.py     # 分层记忆：上下文组装 + 写后记忆更新
├── planner.py    # 全书大纲 → 卷大纲 → 章细纲
├── writer.py     # 写作主流程编排
├── reviewer.py   # 一致性审查与定点修订
└── prompts.py    # 全部 Prompt 模板
```

## 无 Key 体验（Mock 模式）

```bash
NOVEL_AGENT_MOCK=1 python -m novel_agent init /tmp/demo --title 测试
NOVEL_AGENT_MOCK=1 python -m novel_agent plan book /tmp/demo
NOVEL_AGENT_MOCK=1 python -m novel_agent write /tmp/demo 1 3
```

## 免责声明

生成内容为 AI 创作虚构作品，请在遵守当地法律法规的前提下使用，禁止用于生成违法违规内容。
