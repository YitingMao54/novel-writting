"""记忆系统 —— 百万字长篇的上下文支撑。

核心思想：任何模型的上下文窗口都装不下百万字，因此用"分层记忆"代替
"全文投喂"：
  1. 静态层  Story Bible：世界观设定 + 人物卡（写完每章自动增量更新）；
  2. 滚动层  前情提要：早期卷用卷级摘要，近期章用逐章摘要（滑动窗口）；
  3. 线索层  伏笔台账：埋设/回收全程跟踪，写细纲时主动提示回收；
  4. 衔接层  上一章结尾原文，保证文风与情节无缝衔接。

写第 N 章时由 build_chapter_context() 动态组装出一份紧凑、相关的背景材料。
"""

from __future__ import annotations

from . import prompts
from .llm import LLMClient
from .store import Store


# ---------------------------------------------------------------------------
# 前情提要（滚动窗口）
# ---------------------------------------------------------------------------

def build_recap(store: Store, before_chapter: int, recent: int = 8) -> str:
    """组装 before_chapter 之前的前情提要：
    近 recent 章用逐章摘要；更早的按卷聚合卷级摘要。
    """
    cpv = store.volume_chapters()
    parts: list[str] = []

    first_recent = before_chapter - recent
    if first_recent > 1:
        # 更早部分：卷级摘要
        for vol in store.all_volumes():
            vol_start = (vol["number"] - 1) * cpv + 1
            if vol_start >= before_chapter:
                break
            if vol["summary"]:
                parts.append(f"〔卷{vol['number']}·{vol['title']}〕{vol['summary']}")
            else:
                # 卷大纲尚未生成摘要时，回退到该卷已写章的摘要
                chs = store.chapters_between(vol_start, min(before_chapter - 1, vol_start + cpv - 1))
                briefs = [f"第{c['number']}章《{c['title']}》:{c['summary'] or c['outline']}"
                          for c in chs if c["summary"] or c["outline"]]
                if briefs:
                    parts.append(f"〔卷{vol['number']}〕" + "；".join(briefs))

    # 最近章节：逐章摘要
    recent_chs = store.chapters_between(max(1, first_recent), before_chapter - 1)
    recent_lines = [f"第{c['number']}章《{c['title']}》:{c['summary']}"
                    for c in recent_chs if c["status"] == "final" and c["summary"]]
    if recent_lines:
        parts.append("〔最近章节〕\n" + "\n".join(recent_lines))

    return "\n\n".join(parts) if parts else "（本书开篇，暂无前情）"


# ---------------------------------------------------------------------------
# 上下文组装（写第 N 章用）
# ---------------------------------------------------------------------------

def build_chapter_context(store: Store, cfg, ch_num: int,
                          character_names: list[str] | None = None) -> dict:
    """组装写第 ch_num 章所需的全部背景材料，返回 dict 供各模板填充。

    character_names 为 None 时携带全部人物；否则只带指定人物（可省 token）。
    """
    w = cfg.get_writing
    settings = store.all_settings()
    chars = store.all_characters()
    if character_names:
        picked = {}
        for n in character_names:
            card = chars.get(n)
            if card:
                picked[n] = card
        # 主角兜底：细纲人物查不到卡时，附上全部主要人物的前几张
        if not picked:
            picked = dict(list(chars.items())[:5])
        chars = picked

    context = {
        "title": settings.get("theme", "") and store.get_meta("title", "未命名"),
        "genre": settings.get("genre", "未指定"),
        "world": prompts.truncate(settings.get("world", "（未设定）"), 2500),
        "style": settings.get("style", "") or w("style", "网络连载文风"),
        "characters": prompts.truncate(
            "\n\n".join(f"◆ {n}\n{c}" for n, c in chars.items()) or "（暂无人物卡）", 4000),
        "recap": build_recap(store, ch_num, w("recent_summaries", 8)),
        "foreshadows": _foreshadow_lines(store),
    }
    return context


def _foreshadow_lines(store: Store, limit: int = 15) -> str:
    rows = store.open_foreshadows()
    if not rows:
        return "（当前没有未回收的伏笔）"
    lines = [f"- [第{r['planted_ch']}章埋] {r['content']}" for r in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"- ……另有 {len(rows) - limit} 条未列")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 写后记忆更新：摘要 + 人物卡 + 伏笔 + 时间线
# ---------------------------------------------------------------------------

def update_memory_after_chapter(store: Store, llm: LLMClient, cfg,
                                ch_num: int, content: str) -> dict:
    """第 ch_num 章定稿后：提取记忆要素并写回存储。返回解析出的记忆 dict。"""
    settings = store.all_settings()
    user = prompts.CHAPTER_MEMORY_USER.format(
        world=prompts.truncate(settings.get("world", "（未设定）"), 1500),
        foreshadows=_foreshadow_lines(store),
        chapter_number=ch_num,
        content=prompts.truncate(content, 9000),
    )
    mem = llm.chat_json("summarizer", "你是严谨的小说记忆管理助手，只输出合法 JSON。", user)

    # 1) 章摘要
    summary = (mem.get("summary") or "").strip()
    if summary:
        store.update_chapter(ch_num, summary=summary)

    # 2) 人物卡增量更新
    for upd in mem.get("character_updates") or []:
        name = (upd.get("name") or "").strip()
        changes = (upd.get("changes") or "").strip()
        if not name or not changes or changes == "无":
            continue
        old = store.get_character(name)
        if old:
            merged = _merge_character_card(llm, old, changes)
            store.upsert_character(name, merged)
        else:
            store.upsert_character(name, f"姓名：{name}\n当前状态：{changes}")

    # 3) 新角色建档
    for nc in mem.get("new_characters") or []:
        name = (nc.get("name") or "").strip()
        card = (nc.get("card") or "").strip()
        if name and card and not store.get_character(name):
            store.upsert_character(name, card)

    # 4) 伏笔
    for f in mem.get("new_foreshadows") or []:
        f = (f or "").strip()
        if f:
            store.add_foreshadow(f, planted_ch=ch_num)
    for f in mem.get("resolved_foreshadows") or []:
        f = (f or "").strip()
        if f:
            store.resolve_foreshadow(f[:20], resolved_ch=ch_num)

    # 5) 时间线
    tl = (mem.get("timeline_event") or "").strip()
    if tl:
        store.add_timeline(ch_num, tl)

    return mem


def _merge_character_card(llm: LLMClient, old_card: str, changes: str) -> str:
    try:
        merged = llm.chat(
            "summarizer", "",
            prompts.CHARACTER_MERGE_USER.format(old_card=old_card, changes=changes),
        )[0]
        return merged if len(merged) > len(old_card) * 0.3 else old_card
    except Exception:
        # 合并失败时保守回退：追加变化记录
        return old_card + f"\n[状态更新] {changes}"


# ---------------------------------------------------------------------------
# 卷级摘要（卷写完后聚合，用于替代早期章节的逐章摘要）
# ---------------------------------------------------------------------------

def summarize_volume(store: Store, llm: LLMClient, cfg, vol_number: int) -> str:
    vol = store.get_volume(vol_number)
    if not vol:
        return ""
    cpv = store.volume_chapters()
    start, end = (vol_number - 1) * cpv + 1, vol_number * cpv
    chs = store.chapters_between(start, end)
    text = "\n".join(
        f"第{c['number']}章《{c['title']}》:{c['summary']}"
        for c in chs if c["status"] == "final" and c["summary"])
    if not text:
        return ""
    user = (f"以下是小说第{vol_number}卷各章摘要，请浓缩成 300 字以内的卷级剧情摘要，"
            f"保留主线脉络、关键转折与人物结局走向，只输出摘要文本。\n\n{text}")
    summary, _ = llm.chat("summarizer", "", user, max_tokens=800)
    summary = summary.strip()
    if summary:
        store.upsert_volume(vol_number, summary=summary)
    return summary
