import streamlit as st
import json
import faiss
import numpy as np
import requests
import time
import os
import re
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

QWEN_EMBED_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
QWEN_EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def safe_dirname(name: str) -> str:
    """清理文件名，去掉 Windows 不允许的字符和 JSON 包裹"""
    if not name:
        return "novel"
    # 去掉可能的 JSON 包裹
    name = re.sub(r'^[\{\[].*?"(?:title|name)"\s*:\s*"?', '', name)
    name = re.sub(r'"?[\}\]]$', '', name)
    name = name.strip().strip("《》\"'")
    # 替换 Windows 禁止字符
    name = re.sub(r'[\\/:*?"<>|\n\r\t]', '_', name)
    return name[:80]


# ═══════════════════════════════════════════════════════════
# API 客户端
# ═══════════════════════════════════════════════════════════

class DeepSeekClient:
    """DeepSeek API 客户端"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: List[Dict], temperature: float = 0.7,
             max_tokens: int = 4096) -> str:
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{DEEPSEEK_BASE_URL}/chat/completions",
                    headers=self.headers, json=payload, timeout=300
                )
                if resp.status_code == 429:
                    st.warning(f"API 限流，等待 {10 * (attempt + 1)}s 后重试...")
                    time.sleep(10 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.exceptions.Timeout:
                st.warning(f"请求超时，第 {attempt + 1} 次重试...")
                time.sleep(5)
            except requests.exceptions.ConnectionError:
                st.warning(f"连接失败，第 {attempt + 1} 次重试...")
                time.sleep(5)
        raise RuntimeError("DeepSeek API 调用失败，已重试 3 次")


class QwenEmbedClient:
    """千问 text-embedding-v3 嵌入客户端"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        all_emb = []
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "model": QWEN_EMBED_MODEL,
                "input": {"texts": batch},
                "parameters": {"dimension": EMBED_DIM, "text_type": "document"}
            }
            for attempt in range(3):
                try:
                    resp = requests.post(
                        QWEN_EMBED_URL, headers=self.headers,
                        json=payload, timeout=60
                    )
                    if resp.status_code == 429:
                        time.sleep(3 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    batch_emb = [item["embedding"] for item in data["output"]["embeddings"]]
                    all_emb.extend(batch_emb)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    time.sleep(2)
        return all_emb

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ═══════════════════════════════════════════════════════════
# 记忆管理 (FAISS 向量存储)
# ═══════════════════════════════════════════════════════════

class MemoryManager:
    """
    核心记忆系统：用 FAISS 存储所有章节摘要、角色状态、世界细节的嵌入，
    写作时通过语义检索获取最相关的历史上下文，支撑几十万字长篇的一致性。
    """

    def __init__(self, embedder: QwenEmbedClient):
        self.embedder = embedder
        self.entries: List[Dict] = []
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self._next_id = 0

    @staticmethod
    def _normalize(vecs: List[List[float]]) -> np.ndarray:
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-10, None)

    def add(self, content: str, entry_type: str, chapter: int,
            volume: int, importance: float = 0.5):
        emb = self.embedder.embed_single(content)
        norm = self._normalize([emb])
        self.entries.append({
            "id": self._next_id,
            "content": content,
            "type": entry_type,
            "chapter": chapter,
            "volume": volume,
            "importance": importance,
            "embedding": emb
        })
        self.index.add(norm)
        self._next_id += 1

    def search(self, query: str, top_k: int = 8,
               entry_type: Optional[str] = None,
               min_chapter: int = 0) -> List[Tuple[Dict, float]]:
        if not self.entries:
            return []
        qvec = self._normalize([self.embedder.embed_single(query)])
        k = min(top_k, len(self.entries))
        scores, indices = self.index.search(qvec, k)
        results = []
        for s, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            e = self.entries[idx]
            if entry_type and e["type"] != entry_type:
                continue
            if e["chapter"] < min_chapter:
                continue
            results.append((e, float(s)))
        return results

    def get_recent_summaries(self, n: int = 3) -> List[Dict]:
        sums = [e for e in self.entries if e["type"] == "chapter_summary"]
        return sorted(sums, key=lambda x: x["chapter"], reverse=True)[:n]

    def rebuild_index(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        if self.entries:
            vecs = [e["embedding"] for e in self.entries]
            self.index.add(self._normalize(vecs))

    def to_serializable(self) -> List[Dict]:
        return self.entries

    def from_serializable(self, data: List[Dict]):
        self.entries = data
        self._next_id = max((e["id"] for e in data), default=-1) + 1
        self.rebuild_index()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.to_serializable(), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.from_serializable(data)


# ═══════════════════════════════════════════════════════════
# Agent 基类
# ═══════════════════════════════════════════════════════════

class Agent:
    def __init__(self, llm: DeepSeekClient, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt

    def run(self, user_msg: str, temperature: float = 0.7,
            max_tokens: int = 4096, context: str = "") -> str:
        msgs = [{"role": "system", "content": self.system_prompt}]
        if context:
            msgs.append({"role": "user", "content": f"【参考上下文】\n{context}"})
            msgs.append({"role": "assistant", "content": "好的，我已了解上述上下文信息。"})
        msgs.append({"role": "user", "content": user_msg})
        return self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)

    @staticmethod
    def _parse_json(text: str) -> Any:
        text = text.strip()
        # 去掉 markdown 代码块
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # 找到第一个 { 或 [ 的位置
        start = len(text)
        for ch in ["{", "["]:
            idx = text.find(ch)
            if 0 <= idx < start:
                start = idx
        if start == len(text):
            raise ValueError("未找到 JSON")

        # 精确匹配闭合括号，处理嵌套和字符串
        open_ch = text[start]
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\" and in_str:
                esc = True
                continue
            if c == '"' and not esc:
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            end = len(text) - 1

        json_str = text[start:end + 1]
        return json.loads(json_str)


# ═══════════════════════════════════════════════════════════
# 规划 Agent
# ═══════════════════════════════════════════════════════════

PLANNER_PROMPT = """你是一位顶级小说策划大师，擅长构建宏大自洽的小说框架。
所有输出必须严格为 JSON 格式，不要输出任何 JSON 之外的内容。"""


class PlannerAgent(Agent):
    def __init__(self, llm: DeepSeekClient):
        super().__init__(llm, PLANNER_PROMPT)

    def gen_world(self, genre: str, theme: str, style: str, extra: str) -> Dict:
        return self._parse_json(self.run(
            f"为一部{genre}小说设计世界观。\n主题：{theme}\n风格：{style}\n额外要求：{extra}\n\n"
            '{"era":"时代","location":"地点","power_system":"力量体系(详细分级)",'
            '"social_structure":"社会结构","history":"历史","rules":"世界规则"}',
            temperature=0.8, max_tokens=4096
        ))

    def gen_characters(self, world: Dict, genre: str,
                       main_n: int = 3, support_n: int = 5) -> List[Dict]:
        return self._parse_json(self.run(
            f"基于世界观设计角色。\n世界观：{json.dumps(world, ensure_ascii=False)}\n"
            f"类型：{genre}\n{main_n}个主角 + {support_n}个重要配角。\n\n"
            '输出JSON数组，每项：{"name":"名","role":"主角/配角/反派/导师",'
            '"personality":"性格(≥50字)","appearance":"外貌(≥30字)",'
            '"background":"背景(≥50字)","abilities":"能力",'
            '"relationships":{"他者":"关系"},"arc":"成长弧线"}',
            temperature=0.8, max_tokens=8192
        ))

    def gen_outline(self, world: Dict, chars: List[Dict], genre: str,
                    theme: str, n_vols: int, ch_per_vol: int,
                    words_per_ch: int) -> List[Dict]:
        char_brief = "\n".join(
            f"- {c['name']}({c['role']}): {c['personality'][:40]}… 弧线: {c['arc'][:60]}"
            for c in chars
        )
        return self._parse_json(self.run(
            f"设计{genre}小说分卷大纲。\n主题：{theme}\n{n_vols}卷，每卷约{ch_per_vol}章，"
            f"每章{words_per_ch}字。\n世界观：{json.dumps(world, ensure_ascii=False)}\n"
            f"角色：\n{char_brief}\n\n"
            '输出JSON数组，每卷：{"volume_num":序号,"title":"卷名","summary":"概要(100-200字)",'
            '"chapter_outlines":[{"chapter_num":全局章号,"title":"章名",'
            '"summary":"概要(50-100字)","key_events":["事件"],"characters_involved":["角色"],'
            '"emotional_tone":"基调","word_count_target":字数}]}',
            temperature=0.7, max_tokens=16384
        ))

    def gen_title(self, genre: str, theme: str, style: str) -> str:
        t = self.run(
            f"为{genre}小说起一个吸引人的标题。主题：{theme}，风格：{style}。只输出标题文字。",
            temperature=0.9, max_tokens=50
        ).strip().strip("《》")
        return t


# ═══════════════════════════════════════════════════════════
# 写作 Agent
# ═══════════════════════════════════════════════════════════

WRITER_PROMPT = """你是一位文笔精湛的小说家。根据章节大纲和上下文，写出高质量正文。

要求：
1. 文笔流畅有画面感，善用细节和五感描写
2. 对话自然生动，严格符合角色性格
3. 叙事有张有弛，注意节奏变化
4. 严格遵循大纲关键事件
5. 与上下文保持角色性格、世界观、情节的一致性
6. 适当使用修辞增强文学性
7. 段落长短错落，营造阅读韵律

关键：直接输出小说正文，不要标题、注释、分隔线等任何多余内容。"""


class WriterAgent(Agent):
    def __init__(self, llm: DeepSeekClient, memory: MemoryManager):
        super().__init__(llm, WRITER_PROMPT)
        self.memory = memory

    def write_chapter(self, outline: Dict, title: str,
                      world: Dict, chars: List[Dict],
                      prev_summary: str = "",
                      word_target: int = 3000) -> str:
        involved = outline.get("characters_involved", [])
        char_info = ""
        for c in chars:
            if c["name"] in involved:
                char_info += (f"\n【{c['name']}】{c['role']}：{c['personality']}\n"
                              f"  外貌：{c['appearance']}\n  能力：{c['abilities']}\n")

        query = f"{outline['summary']} {' '.join(outline.get('key_events', []))}"
        memories = self.memory.search(query, top_k=6)
        mem_text = ""
        if memories:
            mem_text = "\n【相关历史情节】\n"
            for entry, score in memories:
                if score > 0.25:
                    mem_text += f"- (第{entry['chapter']}章) {entry['content']}\n"

        char_states = self.memory.search(
            "角色状态变化 性格 装备 关系", top_k=4, entry_type="character_state"
        )
        state_text = ""
        if char_states:
            state_text = "\n【角色最新状态】\n"
            for entry, _ in char_states:
                state_text += f"- {entry['content']}\n"

        prompt = (
            f"撰写《{title}》第{outline['chapter_num']}章。\n"
            f"【大纲】标题：{outline['title']}，概要：{outline['summary']}，"
            f"事件：{'、'.join(outline.get('key_events', []))}，"
            f"基调：{outline['emotional_tone']}，目标字数：{word_target}字\n"
            f"【涉及角色】{char_info}\n"
            f"【前文摘要】{prev_summary or '（第一章）'}"
            f"{mem_text}{state_text}\n\n直接输出正文。"
        )

        max_tok = min(int(word_target * 1.8), 16384)
        return self.run(prompt, temperature=0.75, max_tokens=max_tok)


# ═══════════════════════════════════════════════════════════
# 摘要 Agent
# ═══════════════════════════════════════════════════════════

SUMMARY_PROMPT = """你是专业文学编辑，负责提炼章节核心内容用于长篇记忆管理。
输出严格 JSON，包含：情节摘要、角色状态变化、新信息、伏笔、关键细节。"""


class SummaryAgent(Agent):
    def __init__(self, llm: DeepSeekClient):
        super().__init__(llm, SUMMARY_PROMPT)

    def summarize(self, text: str, ch_num: int) -> Dict:
        try:
            result = self._parse_json(self.run(
                f"为第{ch_num}章正文生成摘要。\n【正文】{text[:5000]}\n\n"
                '{"plot_summary":"情节摘要(100-200字)",'
                '"character_changes":["角色：变化"],'
                '"new_information":["新信息"],'
                '"foreshadowing":["伏笔"],'
                '"key_details":["需记住的细节"]}',
                temperature=0.2, max_tokens=2048
            ))
            return result
        except Exception:
            return {
                "plot_summary": text[:200],
                "character_changes": [], "new_information": [],
                "foreshadowing": [], "key_details": []
            }


# ═══════════════════════════════════════════════════════════
# 一致性检查 Agent
# ═══════════════════════════════════════════════════════════

CHECKER_PROMPT = """你是严格的文学审校专家，检查小说内容一致性。
检查维度：角色性格、世界观规则、时间线、地理、称谓统一性。
输出严格 JSON。"""


class CheckerAgent(Agent):
    def __init__(self, llm: DeepSeekClient):
        super().__init__(llm, CHECKER_PROMPT)

    def check(self, text: str, outline: Dict, world: Dict,
              chars: List[Dict]) -> Dict:
        try:
            return self._parse_json(self.run(
                f"检查一致性。\n大纲：{json.dumps(outline, ensure_ascii=False)}\n"
                f"世界规则：{world.get('rules', '')[:400]}\n"
                f"角色：{json.dumps([c for c in chars if c['name'] in outline.get('characters_involved',[])], ensure_ascii=False)[:1000]}\n"
                f"正文：{text[:6000]}\n\n"
                '{"has_issue":bool,"issues":[{"type":"类型","description":"描述","suggestion":"建议"}],'
                '"quality":"优秀/良好/需修改"}',
                temperature=0.15, max_tokens=2048
            ))
        except Exception:
            return {"has_issue": False, "issues": [], "quality": "良好"}


# ═══════════════════════════════════════════════════════════
# 会话状态管理
# ═══════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "stage": "idle",
        "plan": None,
        "chapters": {},
        "summaries": {},
        "memory_data": [],
        "total_words": 0,
        "last_written_ch": 0,
        "novel_title": "",
        "check_results": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def build_agents(deepseek_key: str, qwen_key: str, rebuild_mem: bool = True):
    llm = DeepSeekClient(deepseek_key)
    embedder = QwenEmbedClient(qwen_key)
    memory = MemoryManager(embedder)
    if rebuild_mem and st.session_state["memory_data"]:
        memory.from_serializable(st.session_state["memory_data"])
    planner = PlannerAgent(llm)
    writer = WriterAgent(llm, memory)
    summarizer = SummaryAgent(llm)
    checker = CheckerAgent(llm)
    return planner, writer, summarizer, checker, memory


def persist_memory(memory: MemoryManager):
    st.session_state["memory_data"] = memory.to_serializable()


# ═══════════════════════════════════════════════════════════
# 文件保存 / 加载
# ═══════════════════════════════════════════════════════════

def _get_save_dir() -> str:
    """生成安全的保存目录名"""
    raw_title = st.session_state.get("novel_title", "novel")
    return f"novel_{safe_dirname(raw_title)}"


def save_project(directory: str):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(st.session_state["plan"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(directory, "chapters.json"), "w", encoding="utf-8") as f:
        json.dump(st.session_state["chapters"], f, ensure_ascii=False)
    with open(os.path.join(directory, "summaries.json"), "w", encoding="utf-8") as f:
        json.dump(st.session_state["summaries"], f, ensure_ascii=False)
    mm_data = st.session_state["memory_data"]
    with open(os.path.join(directory, "memory.pkl"), "wb") as f:
        pickle.dump(mm_data, f)
    title = safe_dirname(st.session_state.get("novel_title", "novel"))
    full = assemble_novel()
    with open(os.path.join(directory, f"{title}.txt"), "w", encoding="utf-8") as f:
        f.write(full)
    return directory


def load_project(directory: str):
    plan_path = os.path.join(directory, "plan.json")
    if os.path.exists(plan_path):
        with open(plan_path, "r", encoding="utf-8") as f:
            st.session_state["plan"] = json.load(f)
    ch_path = os.path.join(directory, "chapters.json")
    if os.path.exists(ch_path):
        with open(ch_path, "r", encoding="utf-8") as f:
            st.session_state["chapters"] = json.load(f)
    sum_path = os.path.join(directory, "summaries.json")
    if os.path.exists(sum_path):
        with open(sum_path, "r", encoding="utf-8") as f:
            st.session_state["summaries"] = json.load(f)
    mem_path = os.path.join(directory, "memory.pkl")
    if os.path.exists(mem_path):
        with open(mem_path, "rb") as f:
            st.session_state["memory_data"] = pickle.load(f)
    plan = st.session_state.get("plan")
    if plan:
        st.session_state["novel_title"] = plan.get("title", "")
        st.session_state["last_written_ch"] = len(st.session_state.get("chapters", {}))
        st.session_state["total_words"] = sum(len(t) for t in st.session_state["chapters"].values())
        st.session_state["stage"] = "writing"


def assemble_novel() -> str:
    plan = st.session_state.get("plan")
    chapters = st.session_state.get("chapters", {})
    if not plan:
        return ""
    lines = [f"《{plan['title']}》\n"]
    for vol in plan["volumes"]:
        lines.append(f"\n{'═' * 40}\n第{vol['volume_num']}卷 {vol['title']}\n{'═' * 40}\n")
        for ch in vol.get("chapter_outlines", []):
            cn = str(ch["chapter_num"])
            if cn in chapters:
                lines.append(f"\n第{ch['chapter_num']}章 {ch['title']}\n\n")
                lines.append(chapters[cn])
                lines.append("\n")
    return "\n".join(lines)


def get_flat_chapters(plan: Dict) -> List[Dict]:
    result = []
    if not plan:
        return result
    for vol in plan["volumes"]:
        for ch in vol.get("chapter_outlines", []):
            result.append(ch)
    return sorted(result, key=lambda x: x["chapter_num"])


# ═══════════════════════════════════════════════════════════
# Streamlit 界面
# ═══════════════════════════════════════════════════════════

def main():
    init_state()

    st.set_page_config(page_title="长篇小说生成 Agent", page_icon="📖", layout="wide")

    st.markdown("""<style>
    .hero-title{font-size:2.4rem;font-weight:800;
        background:linear-gradient(135deg,#e8b44d 0%,#c0392b 50%,#8e44ad 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.3rem}
    .hero-sub{color:#999;font-size:.95rem;margin-bottom:1.8rem}
    .chapter-body{line-height:2;font-size:1.05rem;color:#2c2c2c;white-space:pre-wrap}
    .stProgress>div>div>div>div{background:linear-gradient(90deg,#e8b44d,#c0392b)}
    div[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#16213e 100%);color:#e0e0e0}
    div[data-testid="stSidebar"] label,div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span{color:#e0e0e0}
    section[data-testid="stSidebar"] .stTextInput>div>div>input,
    section[data-testid="stSidebar"] .stTextArea>div>div>textarea,
    section[data-testid="stSidebar"] .stSelectbox>div>div>select
    {background:#0f3460;color:#fff;border-color:#e94560}
    </style>""", unsafe_allow_html=True)

    # ── 侧边栏 ──
    with st.sidebar:
        st.markdown("## ⚙️ API 配置")
        dk = st.text_input("DeepSeek API Key", type="password", key="_dk",
                           help="platform.deepseek.com")
        qk = st.text_input("千问 API Key", type="password", key="_qk",
                           help="dashscope.console.aliyun.com")
        st.markdown("---")
        st.markdown("### 📝 小说参数")
        genre = st.selectbox("类型", [
            "玄幻", "仙侠", "都市", "科幻", "历史", "悬疑",
            "奇幻", "武侠", "军事", "言情", "恐怖", "无限流"
        ])
        theme = st.text_input("主题", placeholder="例：重生复仇、废土求生…")
        style = st.selectbox("风格", [
            "严肃正剧", "轻松幽默", "热血爽文", "暗黑深沉",
            "文艺抒情", "硬核写实", "诗意唯美"
        ])
        extra = st.text_area("额外要求", height=70, placeholder="特殊设定…")
        c1, c2 = st.columns(2)
        n_vols = c1.number_input("卷数", 1, 50, 5)
        ch_pv = c2.number_input("每卷章数", 3, 50, 10)
        wpc = st.number_input("每章字数", 1000, 10000, 3000, step=500)
        do_check = st.checkbox("一致性检查（慢）", value=False)

        st.markdown("---")
        total_ch = n_vols * ch_pv
        total_w = total_ch * wpc
        st.markdown(f"**预估：{total_ch} 章 / {total_w:,} 字 / {total_w // 10000} 万字**")

        st.markdown("---")
        st.markdown("### 💾 存档")
        if st.button("保存进度", use_container_width=True):
            if st.session_state["plan"]:
                d = save_project(_get_save_dir())
                st.success(f"已保存 → {d}/")
        ld = st.text_input("存档目录", placeholder="novel_xxx")
        if st.button("加载存档", use_container_width=True):
            if ld and os.path.isdir(ld):
                load_project(ld)
                st.success("加载成功！")
                st.rerun()

    # ── 主区域 ──
    st.markdown('<div class="hero-title">长篇小说生成 Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">DeepSeek 主模型 · 千问嵌入记忆 · FAISS 上下文检索 · 支撑几十万字长篇</div>',
                unsafe_allow_html=True)

    stage = st.session_state["stage"]

    # ═══ IDLE ═══
    if stage == "idle":
        if not dk or not qk:
            st.info("💡 请在左侧边栏填入 DeepSeek 和千问的 API Key")
        elif not theme:
            st.info("💡 请输入小说主题后点击开始")
        else:
            st.success("✅ 准备就绪")
            if st.button("🚀 开始创作", type="primary", use_container_width=True):
                with st.spinner("正在规划小说世界观与大纲，请耐心等待（通常 30-90 秒）…"):
                    try:
                        planner, _, _, _, memory = build_agents(dk, qk, rebuild_mem=False)

                        st.info("🏰 构建世界观…")
                        world = planner.gen_world(genre, theme, style, extra)

                        st.info("👤 设计角色…")
                        mn = max(2, min(4, n_vols))
                        sn = max(3, min(8, n_vols * 2))
                        chars = planner.gen_characters(world, genre, mn, sn)

                        for c in chars:
                            txt = (f"角色{c['name']}({c['role']}): {c['personality']} "
                                   f"外貌{c['appearance']} 背景 {c['background']} "
                                   f"能力{c['abilities']} 关系{json.dumps(c.get('relationships',{}), ensure_ascii=False)}")
                            memory.add(txt, "character_state", 0, 0, 1.0)

                        st.info("🗺️ 规划分卷大纲…")
                        outline = planner.gen_outline(world, chars, genre, theme,
                                                      n_vols, ch_pv, wpc)

                        st.info("✏️ 定名…")
                        title = planner.gen_title(genre, theme, style)

                        wt = (f"世界观: 时代{world['era']} 地点{world['location']} "
                              f"力量{world['power_system']} 社会{world['social_structure']} "
                              f"历史{world['history']} 规则{world['rules']}")
                        memory.add(wt, "world_detail", 0, 0, 1.0)

                        plan = {
                            "title": title, "genre": genre, "theme": theme,
                            "style": style, "world": world, "characters": chars,
                            "volumes": outline
                        }
                        st.session_state["plan"] = plan
                        st.session_state["novel_title"] = title
                        persist_memory(memory)
                        st.session_state["stage"] = "planned"
                        st.success(f"✅ 规划完成！《{title}》")
                        st.rerun()
                    except Exception as e:
                        st.error(f"规划失败: {e}")
                        st.session_state["stage"] = "idle"

    # ═══ PLANNED ═══
    elif stage == "planned":
        plan = st.session_state["plan"]
        flat = get_flat_chapters(plan)

        st.markdown(f"## 📋 《{plan['title']}》创作计划")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("类型", plan["genre"])
        m2.metric("卷数", str(len(plan["volumes"])))
        m3.metric("章数", str(len(flat)))
        m4.metric("目标字数", f"{len(flat) * wpc:,}")

        tabs = st.tabs(["🌍 世界观", "👤 角色", "🗺️ 大纲"])
        with tabs[0]:
            w = plan["world"]
            for k, v in w.items():
                label = {"era": "时代", "location": "地点", "power_system": "力量体系",
                         "social_structure": "社会结构", "history": "历史", "rules": "规则"}.get(k, k)
                st.markdown(f"**{label}：** {v}")
        with tabs[1]:
            for c in plan["characters"]:
                icon = "⭐" if c["role"] == "主角" else "🔹"
                with st.expander(f"{icon} {c['name']}（{c['role']}）"):
                    for k, v in c.items():
                        if k == "relationships" and isinstance(v, dict):
                            st.markdown(f"**{k}：**\n" + "\n".join(f"  - {kk}：{vv}" for kk, vv in v.items()))
                        else:
                            st.markdown(f"**{k}：** {v}")
        with tabs[2]:
            for vol in plan["volumes"]:
                with st.expander(f"第{vol['volume_num']}卷《{vol['title']}》（{len(vol.get('chapter_outlines',[]))}章）"):
                    st.markdown(f"*{vol['summary']}*")
                    for ch in vol.get("chapter_outlines", []):
                        ev = " → ".join(ch.get("key_events", [])) or ch["summary"]
                        st.caption(f"**{ch['chapter_num']}. {ch['title']}** `[{ch['emotional_tone']}]` "
                                   f"({ch['word_count_target']}字)  —  {ev}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✍️ 开始写作", type="primary", use_container_width=True):
                st.session_state["stage"] = "writing"
                st.rerun()
        with c2:
            if st.button("↩️ 重新规划", use_container_width=True):
                st.session_state["stage"] = "idle"
                st.session_state["plan"] = None
                st.rerun()

    # ═══ WRITING ═══
    elif stage == "writing":
        plan = st.session_state["plan"]
        flat = get_flat_chapters(plan)
        written = st.session_state["chapters"]
        done_n = len(written)
        total_n = len(flat)
        remaining = total_n - done_n

        if remaining <= 0:
            st.session_state["stage"] = "done"
            st.rerun()
            return

        st.markdown(f"## ✍️ 正在写作《{plan['title']}》")
        p1, p2, p3 = st.columns(3)
        p1.metric("已完成", f"{done_n} / {total_n} 章")
        p2.metric("剩余", f"{remaining} 章")
        p3.metric("已写字数", f"{st.session_state['total_words']:,}")

        pct = done_n / total_n
        st.progress(pct, text=f"总进度 {pct:.1%}")

        mode = st.radio("生成模式", ["逐章生成（推荐，可控）", "批量生成（自动连续）"],
                        horizontal=True, key="_mode")

        next_idx = done_n
        if next_idx >= total_n:
            st.session_state["stage"] = "done"
            st.rerun()
            return

        ch_outline = flat[next_idx]
        ch_num = ch_outline["chapter_num"]

        vol_title = ""
        for vol in plan["volumes"]:
            if any(c["chapter_num"] == ch_num for c in vol.get("chapter_outlines", [])):
                vol_title = f"第{vol['volume_num']}卷《{vol['title']}》"
                break

        st.info(f"📍 即将生成：{vol_title} — 第{ch_num}章《{ch_outline['title']}》")

        batch_n = 1
        auto = False
        if "批量" in mode:
            batch_n = st.number_input("本次批量章数", 1, min(remaining, 20), value=5)
            auto = True

        if st.button(f"✍️ 生成 {'本章' if batch_n == 1 else f'接下来 {batch_n} 章'}",
                     type="primary", use_container_width=True):
            planner, writer, summarizer, checker, memory = build_agents(dk, qk)

            to_write = flat[next_idx:next_idx + batch_n]
            placeholder = st.empty()

            for i, ch_o in enumerate(to_write):
                cn = ch_o["chapter_num"]
                placeholder.info(f"⏳ 正在生成第{cn}章《{ch_o['title']}》({i+1}/{len(to_write)})…")

                prev_sum = ""
                prev_cn = str(cn - 1)
                if prev_cn in st.session_state["summaries"]:
                    prev_sum = st.session_state["summaries"][prev_cn].get("plot_summary", "")

                try:
                    text = writer.write_chapter(
                        ch_o, plan["title"], plan["world"], plan["characters"],
                        prev_summary=prev_sum, word_target=ch_o.get("word_count_target", wpc)
                    )
                except Exception as e:
                    st.error(f"第{cn}章生成失败: {e}")
                    break

                st.session_state["chapters"][str(cn)] = text
                wc = len(text)
                st.session_state["total_words"] += wc

                with st.expander(f"📖 第{cn}章《{ch_o['title']}》（{wc}字）", expanded=True):
                    st.markdown(f'<div class="chapter-body">{text}</div>', unsafe_allow_html=True)

                try:
                    summary = summarizer.summarize(text, cn)
                    st.session_state["summaries"][str(cn)] = summary

                    memory.add(summary["plot_summary"], "chapter_summary",
                               cn, ch_o.get("volume_num", 1), 0.8)
                    for d in summary.get("key_details", []):
                        memory.add(d, "world_detail", cn, ch_o.get("volume_num", 1), 0.6)
                    for c in summary.get("character_changes", []):
                        memory.add(c, "character_state", cn, ch_o.get("volume_num", 1), 0.9)
                except Exception as e:
                    st.warning(f"摘要生成失败: {e}")

                if do_check:
                    try:
                        cr = checker.check(text, ch_o, plan["world"], plan["characters"])
                        st.session_state["check_results"][str(cn)] = cr
                        if cr.get("has_issue"):
                            with st.expander(f"⚠️ 第{cn}章一致性警告"):
                                for iss in cr.get("issues", []):
                                    st.markdown(f"**{iss['type']}**: {iss['description']}")
                                    st.caption(f"建议: {iss['suggestion']}")
                    except Exception:
                        pass

                if cn % 5 == 0:
                    persist_memory(memory)
                    save_project(_get_save_dir())
                    st.caption(f"💾 已自动保存")

                st.session_state["last_written_ch"] = cn
                persist_memory(memory)

                pct_now = len(st.session_state["chapters"]) / total_n
                st.progress(pct_now, text=f"总进度 {pct_now:.1%}")

                time.sleep(0.3)

            placeholder.empty()

            if len(st.session_state["chapters"]) >= total_n:
                st.session_state["stage"] = "done"
                persist_memory(memory)
                save_project(_get_save_dir())
                st.success("🎉 全部完成！已保存")
                st.rerun()
            elif not auto:
                st.success(f"✅ 第{cn}章完成")
                st.rerun()

        if written:
            with st.expander(f"📋 已写章节列表（{done_n}章）"):
                for ch_o in flat[:done_n]:
                    cn = str(ch_o["chapter_num"])
                    has = cn in written
                    mark = "✅" if has else "⬜"
                    wc = f"({len(written[cn])}字)" if has else ""
                    st.markdown(f"{mark} 第{ch_o['chapter_num']}章《{ch_o['title']}》{wc}")

    # ═══ DONE ═══
    elif stage == "done":
        plan = st.session_state["plan"]
        flat = get_flat_chapters(plan)
        chapters = st.session_state["chapters"]
        tw = st.session_state["total_words"]

        st.markdown(f"## 🎉 《{plan['title']}》创作完成")
        d1, d2, d3 = st.columns(3)
        d1.metric("总章数", str(len(chapters)))
        d2.metric("总字数", f"{tw:,}")
        d3.metric("等级", f"{tw // 10000} 万字")

        sel_vol = st.selectbox("选择卷", range(len(plan["volumes"])),
                               format_func=lambda i: f"第{plan['volumes'][i]['volume_num']}卷"
                                                       f"《{plan['volumes'][i]['title']}》")
        vol = plan["volumes"][sel_vol]
        for ch_o in vol.get("chapter_outlines", []):
            cn = str(ch_o["chapter_num"])
            if cn in chapters:
                with st.expander(f"第{ch_o['chapter_num']}章《{ch_o['title']}》（{len(chapters[cn])}字）"):
                    st.markdown(f'<div class="chapter-body">{chapters[cn]}</div>',
                                unsafe_allow_html=True)
                    if cn in st.session_state.get("check_results", {}):
                        cr = st.session_state["check_results"][cn]
                        if cr.get("has_issue"):
                            st.warning(f"一致性问题 ({cr.get('quality','')})")
                            for iss in cr.get("issues", []):
                                st.caption(f"• [{iss['type']}] {iss['description']}")

        st.markdown("### 📥 导出")
        full_text = assemble_novel()
        safe_title = safe_dirname(plan["title"])
        st.download_button("📥 下载完整小说",
                           data=full_text,
                           file_name=f"{safe_title}.txt",
                           mime="text/plain",
                           use_container_width=True)

        st.download_button("📦 下载结构化数据 (JSON)",
                           data=json.dumps({
                               "plan": plan,
                               "chapters": chapters,
                               "summaries": st.session_state["summaries"]
                           }, ensure_ascii=False, indent=2),
                           file_name=f"{safe_title}_data.json",
                           mime="application/json",
                           use_container_width=True)

        if st.button("🔄 开始新小说", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_state()
            st.rerun()


if __name__ == "__main__":
    main()
