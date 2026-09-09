"""LLM 适配层 —— 统一的 OpenAI Chat Completions 兼容客户端。

特性：
- 按任务角色（planner/writer/reviewer/summarizer）路由到不同厂商与模型；
- 支持流式输出（写正文时实时打印）；
- 指数退避重试（429/5xx/网络错误）；
- NOVEL_AGENT_MOCK=1 时进入 Mock 模式，无需 API Key 即可跑通全流程。
"""

from __future__ import annotations

import json
import os
import re
import time

import httpx

from .config import Config, RoleRoute

MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: Config, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.mock = os.environ.get("NOVEL_AGENT_MOCK", "").strip() in ("1", "true", "yes")
        self._routes: dict[str, RoleRoute] = {}

    def route(self, role: str) -> RoleRoute:
        if role not in self._routes:
            self._routes[role] = self.cfg.resolve_role(role)
        return self._routes[role]

    # ------------------------------------------------------------------ #

    def chat(self, role: str, system: str, user: str, *,
             stream: bool = False, max_tokens: int | None = None,
             json_mode: bool = False) -> tuple[str, dict]:
        """调用一次对话补全。返回 (回复文本, usage)。"""
        if self.mock:
            return mock_complete(role, user), {}

        r = self.route(role)
        payload: dict = {
            "model": r.model,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": user}],
            "temperature": r.temperature,
            "max_tokens": max_tokens or r.max_tokens,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if r.thinking:
            # DeepSeek V4 / GLM 等模型的思考模式开关（OpenAI 兼容扩展字段）
            payload["extra_body"] = {"thinking": {"type": "enabled"}}
        payload.update(r.extra or {})

        headers = {"Content-Type": "application/json"}
        if r.api_key:
            headers["Authorization"] = f"Bearer {r.api_key}"

        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                if stream:
                    return self._stream_once(r, payload, headers)
                with httpx.Client(timeout=300.0) as client:
                    resp = client.post(f"{r.base_url}/chat/completions",
                                       json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                return text.strip(), data.get("usage") or {}
            except (httpx.HTTPError, LLMError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                if self.verbose:
                    print(f"  [llm] {role} 第{attempt + 1}次调用失败: {e}，{delay:.0f}s 后重试")
                time.sleep(delay)
        raise LLMError(f"角色 {role} 调用失败（已重试 {MAX_RETRIES} 次）: {last_err}")

    def _stream_once(self, r: RoleRoute, payload: dict, headers: dict) -> tuple[str, dict]:
        """流式接收，实时打印增量文本。"""
        chunks: list[str] = []
        usage = {}
        with httpx.Client(timeout=600.0) as client:
            with client.stream("POST", f"{r.base_url}/chat/completions",
                               json=payload, headers=headers) as resp:
                if resp.status_code in (429, 500, 502, 503, 504):
                    body = resp.read().decode("utf-8", "ignore")[:200]
                    raise LLMError(f"HTTP {resp.status_code}: {body}")
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    try:
                        delta = obj["choices"][0].get("delta") or {}
                        piece = delta.get("content") or ""
                    except (KeyError, IndexError):
                        piece = ""
                    if piece:
                        chunks.append(piece)
                        if self.verbose:
                            print(piece, end="", flush=True)
        if self.verbose:
            print()
        return "".join(chunks).strip(), usage

    # ------------------------------------------------------------------ #

    def chat_json(self, role: str, system: str, user: str,
                  max_tokens: int | None = None) -> dict:
        """要求 JSON 输出并稳健解析（容忍 ```json 代码块包裹）。"""
        text, _ = self.chat(role, system, user, max_tokens=max_tokens, json_mode=True)
        return parse_json(text)

    def log_usage(self, store, role: str, usage: dict):
        if not usage:
            return
        r = self.route(role)
        store.log_usage(role, r.model, usage)


# ---------------------------------------------------------------------------
# JSON 稳健解析
# ---------------------------------------------------------------------------

def parse_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象：容忍 markdown 代码块与前后噪声。"""
    text = text.strip()
    # 1) 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) 提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3) 提取第一个平衡的 {...}
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"无法从模型输出中解析 JSON:\n{text[:400]}")


# ---------------------------------------------------------------------------
# Mock 模式 —— 无 Key 跑通全流程（根据 prompt 特征返回不同类型假数据）
# ---------------------------------------------------------------------------

MOCK_PROSE = (
    "夜色像一层浸了水的绒布，沉沉压在城头上。{name}按着剑柄，指节因用力而泛白。"
    "风从北面来，带着铁锈与血混杂的气味，火把在垛口后明明灭灭，把每个人的影子拉得老长。"
    "\n\n“来了。”老周的声音压得极低，像是怕惊动什么。{name}没有答话，只是眯起眼望向远处"
    "那条起伏的黑线——那不是山脊，是人。数不清的人，正踩着枯草与碎石，朝这座孤城缓缓合围。"
    "\n\n三日前他还站在师门的演武场上，听师父讲“心正则剑正”的道理；如今那座山门已成了一片"
    "焦黑的瓦砾，师父的话也成了他怀里唯一带出来的东西。他把这口气压进丹田，缓缓拔剑。"
    "\n\n剑身映出火光，也映出他的眼睛。那双曾经总被同门笑作“过分温和”的眼睛，此刻静得像结了冰的湖面。"
    "\n\n“守不住，就死在这。”他说给自己听。城下的号角声忽然拔高，如同一把钝刀劈开了夜幕。"
    "{name}深吸一口气，迎着那片翻涌而来的黑潮，迈出了第一步。"
)


def mock_complete(role: str, user: str) -> str:
    """按 prompt 特征返回结构正确的假响应，保证端到端流程可验证。"""
    if "全书大纲" in user and "volumes" in user:
        vols = [{"number": i + 1, "title": f"第{i + 1}卷·命运齿轮",
                 "summary": f"第{i + 1}卷概述：主角历经磨砺，主线向前推进一大步。",
                 "key_events": [f"卷{i + 1}关键事件A", f"卷{i + 1}关键事件B",
                                f"卷{i + 1}关键事件C", f"卷{i + 1}关键事件D"]}
                for i in range(2)]
        return json.dumps({
            "theme": "平凡少年在乱世中寻找自我与救赎",
            "main_plot": "边城少年林昭因家族蒙冤踏上修行路，在乱世中逐渐揭开当年真相，"
                         "从复仇者成长为守护者，最终在旧怨与苍生之间做出抉择。",
            "ending": "林昭查明真相后放下私仇，以身守护边城，完成从少年到英雄的蜕变。",
            "volumes": vols,
        }, ensure_ascii=False)
    if "卷大纲" in user and "chapters" in user:
        import re as _re
        m = _re.search(r"第 (\d+) 章至第 (\d+) 章", user)
        start, end = (int(m.group(1)), int(m.group(2))) if m else (1, 3)
        chs = [{"number": n, "title": f"风起第{n}章",
                "brief": f"第{n}章梗概：危机逼近，主角做出关键抉择，局势发生转折。"}
               for n in range(start, end + 1)]
        return json.dumps({"title": "卷标题·风起", "summary": "本卷概述：主角初入江湖，各方势力登场。",
                           "chapters": chs}, ensure_ascii=False)
    if "章《" in user and "scenes" in user:
        return json.dumps({
            "title": "孤城夜守",
            "pov": "林昭",
            "chapter_goal": "推进守城危机，展现主角成长",
            "scenes": [
                {"location": "城头", "time": "深夜", "characters": ["林昭", "老周"],
                 "event": "敌军夜袭，林昭首次独当一面指挥防御", "goal": "制造危机并展现成长",
                 "emotion": "紧张压抑中透出决意"},
                {"location": "城内军帐", "time": "凌晨", "characters": ["林昭", "苏青"],
                 "event": "苏青带来援军绝无的坏消息，两人定下险计", "goal": "推进主线冲突",
                 "emotion": "焦虑与信任交织"},
                {"location": "城头", "time": "黎明", "characters": ["林昭"],
                 "event": "林昭率众死守，黎明前最凶险一波攻势被打退", "goal": "章末留钩子",
                 "emotion": "悲壮与希望"},
            ],
        }, ensure_ascii=False)
    if "提取记忆要素" in user:
        return json.dumps({
            "summary": "敌军夜袭孤城，林昭首次独立指挥防御；苏青带来援军无望的消息，"
                       "两人定下以少胜多的险计；黎明前打退最凶险一波攻势。",
            "characters": ["林昭", "老周", "苏青"],
            "character_updates": [
                {"name": "林昭", "changes": "首次独立指挥作战，信心与威望提升，略受轻伤"}
            ],
            "new_characters": [],
            "new_foreshadows": ["苏青提及的神秘援军信物下落不明"],
            "resolved_foreshadows": [],
            "timeline_event": "守城第三夜，敌军夜袭被击退，故事时间推进两天"
        }, ensure_ascii=False)
    if "连贯性问题" in user:
        return json.dumps({"pass": True, "score": 9, "issues": []}, ensure_ascii=False)
    if "修订" in user and "问题清单" in user:
        body = user.split("【原文】")[-1]
        return body.strip() or MOCK_PROSE
    if "人物卡（旧）" in user:
        return ("姓名：林昭\n身份：边城守军统领\n外貌：剑眉星目，身形挺拔\n"
                "性格：外冷内热，遇事沉静果决\n背景：将门之后，家族蒙冤\n"
                "关系：苏青-生死之交；老周-部下兼长辈\n"
                "当前状态：首次独立指挥作战后威望提升，左臂轻伤未愈")
    # 默认：正文
    name = "林昭"
    m = re.search(r"人物名[：:]\s*(\S+)", user)
    if m:
        name = m.group(1)
    return MOCK_PROSE.format(name=name) * 3
