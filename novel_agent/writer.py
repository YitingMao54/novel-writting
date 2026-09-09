"""写作主流程 —— 单章与批量写作的编排核心。

单章流程：
  确保细纲 → 组装上下文 → 逐场景生成（流式）→ 拼接成章 →
  一致性审查 →（有问题）定点修订 → 记忆回写（摘要/人物卡/伏笔/时间线）→ 定稿

批量流程 write_range：逐章执行上述流程，天然支持断点续写
（已 final 的章自动跳过），Ctrl+C 中断后再次运行即可从断点继续。
"""

from __future__ import annotations

import json
import time

from . import prompts
from .llm import LLMClient
from .memory import build_chapter_context, update_memory_after_chapter
from .planner import plan_chapter
from .reviewer import review_chapter, revise_chapter
from .store import Store


def ensure_outline(store: Store, llm: LLMClient, cfg, ch_num: int) -> dict:
    """确保第 ch_num 章有细纲；没有则自动生成。"""
    ch = store.get_chapter(ch_num)
    if ch and ch["outline"]:
        try:
            return json.loads(ch["outline"])
        except json.JSONDecodeError:
            pass
    return plan_chapter(store, llm, cfg, ch_num)


def write_chapter(store: Store, llm: LLMClient, cfg, ch_num: int,
                  review: bool | None = None, quiet: bool = False) -> str:
    """写第 ch_num 章，走完整流水线，返回正文。"""
    w = cfg.get_writing
    if review is None:
        review = bool(w("review_before_final", True))

    def log(msg: str):
        if not quiet:
            print(msg, flush=True)

    # 0) 细纲
    log(f"\n{'=' * 60}\n▶ 第 {ch_num} 章 · 生成细纲 ...")
    detail = ensure_outline(store, llm, cfg, ch_num)
    scenes = detail.get("scenes") or []
    title = detail.get("title", f"第{ch_num}章")
    if not scenes:
        raise ValueError(f"第 {ch_num} 章细纲无场景，终止")
    store.update_chapter(ch_num, title=title, outline=json.dumps(detail, ensure_ascii=False),
                         status="outlined")
    log(f"  《{title}》共 {len(scenes)} 个场景")

    # 1) 上下文（人物取本章细纲出场人物 + 主角）
    char_names = []
    for s in scenes:
        char_names.extend(s.get("characters") or [])
    if detail.get("pov"):
        char_names.append(detail["pov"])
    seen: set = set()
    char_names = [n for n in char_names if not (n in seen or seen.add(n))]
    ctx = build_chapter_context(store, cfg, ch_num, character_names=char_names or None)

    # 2) 逐场景生成
    target_scene = max(600, w("chapter_words", 3000) // len(scenes))
    parts: list[str] = []
    for i, scene in enumerate(scenes, 1):
        log(f"▶ 第 {ch_num} 章 · 场景 {i}/{len(scenes)}【{scene.get('location', '')}】流式生成：")
        prev_tail = ""
        if parts:
            prev_tail = f"【本章上一场景结尾】\n…{parts[-1][-400:]}"
        elif ch_num > 1:
            prev_ch = store.get_chapter(ch_num - 1)
            tail_words = w("prev_tail_words", 600)
            if prev_ch and prev_ch["content"]:
                prev_tail = f"【上一章结尾】\n…{prev_ch['content'][-tail_words * 2:]}"
        user = prompts.WRITE_SCENE_USER.format(
            chapter_number=ch_num,
            scene_index=i,
            scene_total=len(scenes),
            world=ctx["world"],
            characters=ctx["characters"],
            recap=prompts.truncate(ctx["recap"], 3000),
            prev_tail=prev_tail or "（本章为开篇第一章，请以有力开场）",
            previous_scenes="；".join(
                f"场景{j}：{scenes[j - 1].get('event', '')}" for j in range(1, i)
            ) or "（无）",
            scene_outline=json.dumps(scene, ensure_ascii=False, indent=1),
            target_words=target_scene,
            style=ctx["style"],
        )
        text, usage = llm.chat("writer", prompts.WRITER_SYSTEM, user, stream=not quiet)
        if usage:
            llm.log_usage(store, "writer", usage)
        if not text:
            raise ValueError(f"第 {ch_num} 章场景 {i} 生成结果为空")
        parts.append(text.strip())

    content = "\n\n".join(parts)
    word_count = len(content.replace(" ", "").replace("\n", ""))
    store.update_chapter(ch_num, content=content, word_count=word_count, status="drafted")
    log(f"  ✔ 初稿完成：{word_count} 字")

    # 3) 一致性审查 + 修订
    if review:
        log("▶ 一致性审查 ...")
        result = review_chapter(store, llm, cfg, ch_num)
        score = result.get("score", "?")
        issues = result.get("issues", [])
        log(f"  审查评分 {score}/10，问题 {len(issues)} 个")
        for it in issues:
            log(f"   - [{it.get('severity')}/{it.get('type')}] {it.get('problem', '')[:60]}")
        if issues and not result.get("pass", True):
            log("▶ 定点修订 ...")
            revised = revise_chapter(store, llm, cfg, ch_num, issues)
            if revised != content:
                content = revised
                word_count = len(content.replace(" ", "").replace("\n", ""))
                log(f"  ✔ 修订完成：{word_count} 字")

    # 4) 记忆回写
    log("▶ 记忆回写（摘要 / 人物卡 / 伏笔 / 时间线）...")
    mem = update_memory_after_chapter(store, llm, cfg, ch_num, content)
    if not quiet and mem.get("summary"):
        log(f"  本章摘要：{mem['summary'][:80]}…")

    # 5) 定稿
    store.update_chapter(ch_num, status="final", word_count=word_count)
    total = store.export_state()["total_words"]
    log(f"✅ 第 {ch_num} 章定稿《{title}》| 本章 {word_count} 字 | 全书累计 {total} 字")
    return content


def write_range(store: Store, llm: LLMClient, cfg, start: int, end: int,
                review: bool | None = None, quiet: bool = False) -> dict:
    """批量写作 [start, end]，断点续写，返回统计。"""
    done, failed = 0, []
    t0 = time.time()
    for n in range(start, end + 1):
        ch = store.get_chapter(n)
        if ch and ch["status"] == "final":
            if not quiet:
                print(f"⏭ 第 {n} 章已定稿，跳过")
            continue
        try:
            write_chapter(store, llm, cfg, n, review=review, quiet=quiet)
            done += 1
        except KeyboardInterrupt:
            print(f"\n⏸ 已中断。断点：第 {n} 章。重新运行 write {n} {end} 可继续。")
            break
        except Exception as e:
            print(f"❌ 第 {n} 章写作失败：{e}")
            failed.append((n, str(e)))
    stats = {
        "written": done,
        "failed": failed,
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "state": store.export_state(),
    }
    return stats
