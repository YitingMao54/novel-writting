"""novel-agent —— 百万字级长篇小说自动写作 Agent。

架构：分层规划（全书大纲 → 卷大纲 → 章细纲）+ 长期记忆（Story Bible /
滚动前情摘要 / 伏笔台账）+ 场景级生成 + 一致性审查，配合任意
OpenAI 兼容的国产大模型（DeepSeek / 通义千问 / Kimi / 智谱 GLM 等）。
"""

__version__ = "1.0.0"
