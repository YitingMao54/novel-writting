"""规划流水线 —— 全书大纲 → 卷大纲 → 章细纲 的三级规划。

三级结构是百万字可控性的关键：
- 全书大纲定主线与分卷，保证"不写歪"；
- 卷大纲把主线拆成可执行的章级梗概，保证"节奏不崩"；
- 章细纲结合最新记忆（前情+伏笔）现场生成，保证"局部连贯"。
"""

from __future__ import annotations

import json

from . import prompts
from .llm import LLMClient
from .memory import build_chapter_context
from .store import Store


def _writing_defaults(cfg) -> dict:
    return {
        "chapter_words": cfg.get_writing("chapter_words", 3000),
        "min_scenes": cfg.get_writing("min_scenes", 3),
        "max_scenes": cfg.get_writing("max_scenes", 5),
        "chapters_per_volume": cfg.get_writing("chapters_per_volume", 40),
        "volume_count": cfg.get_writing("volume_count", 10),
    }


# ---------------------------------------------------------------------------
# 全书大纲
# ---------------------------------------------------------------------------

def plan_book(store: Store, llm: LLMClient, cfg) -> dict:
    d = _writing_defaults(cfg)
    total_words = d["chapter_words"] * d["chapters_per_volume"] * d["volume_count"] // 10000
    settings = store.all_settings()

    user = prompts.PLAN_BOOK_USER.format(
        title=store.get_meta("title", "未命名"),
        genre=settings.get("genre", "未指定"),
        target_words=total_words,
        volume_count=d["volume_count"],
        chapters_per_volume=d["chapters_per_volume"],
        chapter_words=d["chapter_words"],
        world=prompts.truncate(settings.get("world", "（未设定）"), 2000),
        characters=prompts.truncate(
            "\n".join(f"- {n}：{card[:120]}" for n, card in store.all_characters().items()),
            2000),
        style=settings.get("style", "") or cfg.get_writing("style", ""),
    )
    outline = llm.chat_json("planner", "你是专业的长篇小说策划编辑，只输出合法 JSON。", user)

    store.set_setting("theme", outline.get("theme", ""))
    store.set_setting("main_plot", outline.get("main_plot", ""))
    store.set_setting("book_outline", json.dumps(outline, ensure_ascii=False, indent=2))
    for v in outline.get("volumes") or []:
        num = int(v.get("number") or 0)
        if num:
            store.upsert_volume(num, title=v.get("title", ""),
                                outline=v.get("summary", ""))
    return outline


# ---------------------------------------------------------------------------
# 卷大纲
# ---------------------------------------------------------------------------

def plan_volume(store: Store, llm: LLMClient, cfg, vol_number: int) -> dict:
    d = _writing_defaults(cfg)
    cpv = d["chapters_per_volume"]
    start_ch = (vol_number - 1) * cpv + 1
    end_ch = vol_number * cpv

    settings = store.all_settings()
    book_outline = settings.get("book_outline", "")
    main_plot = settings.get("main_plot", "")
    recap = _recap_before_volume(store, vol_number, cpv)

    # 已写卷的卷大纲（防止重复规划）
    prior_vols = [v for v in store.all_volumes() if v["number"] < vol_number and v["outline"]]
    prior_lines = "\n".join(f"卷{v['number']}《{v['title']}》:{v['outline'][:150]}"
                            for v in prior_vols)

    user = prompts.PLAN_VOLUME_USER.format(
        title=store.get_meta("title", "未命名"),
        genre=settings.get("genre", "未指定"),
        main_plot=main_plot or book_outline[:1500] or "（先运行 plan book）",
        book_outline=prompts.truncate(book_outline, 2500),
        world=prompts.truncate(settings.get("world", "（未设定）"), 1500),
        characters=prompts.truncate(
            "\n".join(f"- {n}：{card[:120]}" for n, card in store.all_characters().items()),
            2000),
        recap=recap,
        volume_number=vol_number,
        chapters_per_volume=cpv,
        start_ch=start_ch,
        end_ch=end_ch,
        chapter_words=d["chapter_words"],
        volume_hint=(f"【已规划卷概要（勿重复）】\n{prior_lines}" if prior_lines else ""),
    )
    vol = llm.chat_json("planner", "你是专业的长篇小说策划编辑，只输出合法 JSON。", user)

    chapters = vol.get("chapters") or []
    # 卷大纲落库：每章先建记录并写入 brief
    for c in chapters:
        try:
            num = int(c.get("number"))
        except (TypeError, ValueError):
            continue
        store.ensure_chapter(num, volume=vol_number)
        store.update_chapter(num, title=c.get("title", ""), outline=c.get("brief", ""),
                             status="outlined" if store.get_chapter(num)["status"] == "empty"
                             else store.get_chapter(num)["status"])
    store.upsert_volume(vol_number,
                        title=vol.get("title", ""),
                        outline=vol.get("summary", ""),
                        summary=None)
    return vol


def _recap_before_volume(store: Store, vol_number: int, cpv: int) -> str:
    from .memory import build_recap
    return build_recap(store, (vol_number - 1) * cpv + 1, recent=cpv)


# ---------------------------------------------------------------------------
# 章细纲
# ---------------------------------------------------------------------------

def plan_chapter(store: Store, llm: LLMClient, cfg, ch_num: int) -> dict:
    d = _writing_defaults(cfg)
    ch = store.get_chapter(ch_num)
    if not ch:
        store.ensure_chapter(ch_num, volume=store.volume_of(ch_num))
        ch = store.get_chapter(ch_num)
    vol_number = ch["volume"]

    ctx = build_chapter_context(store, cfg, ch_num)
    vol = store.get_volume(vol_number)
    volume_brief = (ch["outline"] or "") if ch["outline"] else ""
    if vol and vol["outline"] and not volume_brief:
        volume_brief = f"（本卷大纲：{vol['outline'][:200]}；请自行设计本章剧情）"

    user = prompts.PLAN_CHAPTER_USER.format(
        chapter_number=ch_num,
        chapter_title_hint=ch["title"] or "（待定）",
        genre=ctx["genre"],
        style=ctx["style"],
        world=ctx["world"],
        characters=ctx["characters"],
        recap=ctx["recap"],
        foreshadows=ctx["foreshadows"],
        volume_brief=volume_brief or "（无预设梗概，请根据前情与伏笔自行设计）",
        min_scenes=d["min_scenes"],
        max_scenes=d["max_scenes"],
    )
    detail = llm.chat_json("planner", "你是专业的小说策划编辑，只输出合法 JSON。", user)

    outline_json = json.dumps(detail, ensure_ascii=False)
    store.update_chapter(ch_num, title=detail.get("title", ch["title"]),
                         outline=outline_json,
                         status="outlined" if ch["status"] == "empty" else ch["status"])
    return detail
