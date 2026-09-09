"""一致性审查与修订 —— 长篇连载的"质检车间"。

审查员只关心硬伤：人物 OOC、设定冲突、时间线错误、前情矛盾、细纲偏离。
发现问题后由修订环节定点修复，避免全文重写带来的风格漂移。
"""

from __future__ import annotations

import json

from . import prompts
from .llm import LLMClient
from .memory import build_chapter_context
from .store import Store

REVIEW_SYSTEM = prompts.REVIEW_SYSTEM


def review_chapter(store: Store, llm: LLMClient, cfg, ch_num: int) -> dict:
    """审查第 ch_num 章，返回 {pass, score, issues:[...]} 并落库。"""
    ch = store.get_chapter(ch_num)
    if not ch or not ch["content"]:
        raise ValueError(f"第 {ch_num} 章尚无正文，无法审查")

    ctx = build_chapter_context(store, cfg, ch_num)
    outline_text = ch["outline"]
    try:
        outline_text = json.dumps(json.loads(ch["outline"]), ensure_ascii=False, indent=1)
    except (json.JSONDecodeError, TypeError):
        pass

    user = prompts.REVIEW_USER.format(
        chapter_number=ch_num,
        world=ctx["world"],
        characters=prompts.truncate(ctx["characters"], 3500),
        recap=prompts.truncate(ctx["recap"], 2500),
        outline=outline_text,
        content=prompts.truncate(ch["content"], 9000),
    )
    result = llm.chat_json("reviewer", REVIEW_SYSTEM, user)
    result.setdefault("issues", [])
    result.setdefault("pass", not result["issues"])
    result.setdefault("score", 8)
    store.update_chapter(ch_num, review_json=json.dumps(result, ensure_ascii=False))
    return result


def revise_chapter(store: Store, llm: LLMClient, cfg, ch_num: int,
                   issues: list[dict] | None = None) -> str:
    """根据问题清单修订正文，返回修订后的全文并落库。"""
    ch = store.get_chapter(ch_num)
    if not ch or not ch["content"]:
        raise ValueError(f"第 {ch_num} 章尚无正文，无法修订")

    if issues is None:
        issues = []
        if ch["review_json"]:
            try:
                issues = json.loads(ch["review_json"]).get("issues", [])
            except json.JSONDecodeError:
                issues = []
    serious = [i for i in issues if i.get("severity") in ("high", "medium")]
    if not serious:
        return ch["content"]  # 无需修订

    ctx = build_chapter_context(store, cfg, ch_num)
    user = prompts.REVISE_USER.format(
        characters=prompts.truncate(ctx["characters"], 3000),
        recap=prompts.truncate(ctx["recap"], 2000),
        issues=json.dumps(serious, ensure_ascii=False, indent=1),
        content=ch["content"],
    )
    revised, _ = llm.chat("writer", prompts.WRITER_SYSTEM, user, stream=False)
    # 修订输出做最小校验：长度不能缩水得离谱（防止截断/跑偏）
    if len(revised) >= len(ch["content"]) * 0.5:
        store.update_chapter(ch_num, content=revised,
                             word_count=len(revised.replace(" ", "").replace("\n", "")))
        return revised
    return ch["content"]
