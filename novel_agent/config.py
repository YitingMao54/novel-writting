"""配置管理 —— 厂商预设、角色→模型路由、项目配置加载。"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# 国产大模型厂商预设（全部 OpenAI Chat Completions 兼容）
# base_url 可被项目配置覆盖；api_key 默认从环境变量读取
# ---------------------------------------------------------------------------

PROVIDER_PRESETS: dict[str, dict] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "note": "DeepSeek V4（deepseek-v4-flash / deepseek-v4-pro）",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "note": "阿里云百炼·通义千问（qwen3.8-max / qwen3.7-plus / qwen3.7-flash）",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "note": "月之暗面 Kimi（kimi-k3，1M 上下文）",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY",
        "note": "智谱 GLM（glm-5.2 / glm-5.3-flash / glm-4.7-flash 免费）",
    },
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "note": "MiniMax（abab / MiniMax-Text 系列）",
    },
    "hunyuan": {
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "api_key_env": "HUNYUAN_API_KEY",
        "note": "腾讯混元（hunyuan-turbos / hunyuan-large）",
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env": "ARK_API_KEY",
        "note": "火山方舟·豆包（endpoint 或 doubao-* 模型 ID）",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "note": "硅基流动聚合（DeepSeek/Qwen/GLM 开源模型一站式）",
    },
    "custom": {
        "base_url": "",
        "api_key_env": "",
        "note": "任意 OpenAI 兼容端点，直接填 base_url 与 api_key",
    },
}

# 四类任务角色：不同环节可路由到不同厂商/模型，兼顾质量与成本
ROLE_KEYS = ("planner", "writer", "reviewer", "summarizer")

DEFAULT_ROLE_ROUTES: dict[str, dict] = {
    # 大纲/细纲需要结构化与推理能力
    "planner": {"provider": "deepseek", "model": "deepseek-v4-flash",
                "temperature": 0.7, "thinking": False},
    # 正文写作重文风，温度调高
    "writer": {"provider": "deepseek", "model": "deepseek-v4-flash",
               "temperature": 0.9, "thinking": False},
    # 审查需要严谨、低随机性
    "reviewer": {"provider": "qwen", "model": "qwen3.7-plus",
                 "temperature": 0.2, "thinking": False},
    # 摘要/记忆提取走最便宜的档位
    "summarizer": {"provider": "qwen", "model": "qwen3.7-flash",
                   "temperature": 0.3, "thinking": False},
}


@dataclass
class RoleRoute:
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    temperature: float = 0.8
    max_tokens: int = 8192
    thinking: bool = False
    extra: dict = field(default_factory=dict)


class Config:
    """项目配置：读取 <project>/config.yaml，合并厂商预设与角色路由。"""

    def __init__(self, project_dir: str | Path):
        self.project_dir = Path(project_dir).resolve()
        self.path = self.project_dir / "config.yaml"
        self.data: dict = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = yaml.safe_load(f) or {}

    # -- 基本信息 -----------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self.project_dir / "novel.db"

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "output"

    def get_writing(self, key: str, default=None):
        return (self.data.get("writing") or {}).get(key, default)

    # -- 角色路由 -------------------------------------------------------------

    def resolve_role(self, role: str) -> RoleRoute:
        """把角色名解析成完整的调用参数（合并厂商预设 + 配置覆盖 + 环境变量）。"""
        if role not in ROLE_KEYS:
            raise ValueError(f"未知任务角色: {role}，可选 {ROLE_KEYS}")
        merged = copy.deepcopy(DEFAULT_ROLE_ROUTES[role])
        cfg_roles = self.data.get("roles") or {}
        if role in cfg_roles and isinstance(cfg_roles[role], dict):
            merged.update(cfg_roles[role])
        provider_name = merged.pop("provider", "custom")
        preset = dict(PROVIDER_PRESETS.get(provider_name, PROVIDER_PRESETS["custom"]))
        base_url = merged.pop("base_url", None) or preset["base_url"]
        if not base_url:
            raise ValueError(f"角色 {role} 的 provider={provider_name} 未配置 base_url")
        api_key_env = merged.pop("api_key_env", None) or preset["api_key_env"]
        api_key = merged.pop("api_key", None) or os.environ.get(api_key_env or "", "")
        return RoleRoute(
            provider=provider_name,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            api_key_env=api_key_env,
            **merged,
        )

    # -- 写入 -----------------------------------------------------------------

    def save_default_config(self, title: str = "未命名小说"):
        """init 时生成项目默认 config.yaml（DeepSeek/千问双厂商示例路由）。"""
        if self.path.exists():
            return
        doc = {
            "title": title,
            "providers": {
                # 在此覆盖 base_url / api_key；api_key 建议走环境变量
                "deepseek": {"api_key_env": "DEEPSEEK_API_KEY"},
                "qwen": {"api_key_env": "DASHSCOPE_API_KEY"},
            },
            "roles": {
                "planner": {"provider": "deepseek", "model": "deepseek-v4-flash"},
                "writer": {"provider": "deepseek", "model": "deepseek-v4-flash",
                           "temperature": 0.9},
                "reviewer": {"provider": "qwen", "model": "qwen3.7-plus"},
                "summarizer": {"provider": "qwen", "model": "qwen3.7-flash"},
            },
            "writing": {
                "chapter_words": 3000,        # 每章目标字数
                "min_scenes": 3,              # 每章最少场景数
                "max_scenes": 5,              # 每章最多场景数
                "chapters_per_volume": 40,    # 每卷章数（40章×3000字=12万字/卷）
                "volume_count": 10,           # 卷数（10卷≈120万字）
                "recent_summaries": 8,        # 前情提要携带的最近章节数
                "prev_tail_words": 600,       # 上一章结尾携带字数
                "review_before_final": True,  # 定稿前自动一致性审查
                "style": "网络连载文风：节奏快、代入感强、多用对话与动作推进，"
                         "叙述凝练，每章结尾留钩子。",
            },
        }
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
