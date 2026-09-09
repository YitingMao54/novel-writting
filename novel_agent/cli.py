"""CLI 入口 —— python -m novel_agent <command>。

命令一览：
  init <dir>                      初始化小说项目
  bible show / set-*              世界观设定管理
  character add/list/show         人物卡管理
  plan book / volume <n> / chapter <n>   三级规划
  write <n> | <from> <to>         写单章 / 批量连写（断点续写）
  review <n> / revise <n>         手动审查 / 修订
  volume-summary <n>              生成卷级摘要
  status / usage / export         进度 / 用量 / 导出全文
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .llm import LLMClient
from .memory import summarize_volume
from .planner import plan_book, plan_chapter, plan_volume
from .reviewer import review_chapter, revise_chapter
from .store import Store
from .writer import write_chapter, write_range


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _open_project(project_dir: str) -> tuple[Config, Store, LLMClient]:
    p = Path(project_dir).expanduser().resolve()
    if not (p / "config.yaml").exists():
        sys.exit(f"错误：{p} 不是 novel-agent 项目（缺 config.yaml）。先运行 init。")
    cfg = Config(p)
    store = Store(cfg.db_path)
    llm = LLMClient(cfg)
    return cfg, store, llm


def _add_common(parser: argparse.ArgumentParser):
    parser.add_argument("project", help="小说项目目录路径")
    parser.add_argument("--no-review", action="store_true",
                        help="跳过定稿前一致性审查")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="静默模式（不流式打印正文）")


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------

def cmd_init(args):
    p = Path(args.project).expanduser().resolve()
    if (p / "config.yaml").exists():
        sys.exit(f"错误：{p} 已是初始化过的项目")
    p.mkdir(parents=True, exist_ok=True)
    title = args.title or p.name
    cfg = Config(p)
    cfg.save_default_config(title)
    store = Store(cfg.db_path)
    store.set_meta("title", title)
    store.set_setting("genre", args.genre or "未指定")
    store.set_setting("world", "")
    store.set_setting("style", "")
    print(f"✅ 项目已初始化：{p}")
    print(f"   书名：{title}")
    print("下一步：")
    print("  1) 编辑 config.yaml（模型路由 / 每卷章数 / 卷数）")
    print("  2) 设置 API Key 环境变量（如 DEEPSEEK_API_KEY、DASHSCOPE_API_KEY）")
    print(f"  3) novel-agent bible set-world {p} \"世界观文本\"")
    print(f"  4) novel-agent character add {p} 林昭 \"人物卡文本\"")
    print(f"  5) novel-agent plan book {p}")


def cmd_bible(args):
    cfg, store, _ = _open_project(args.project)
    if args.bible_cmd == "show":
        s = store.all_settings()
        print("=" * 50)
        print(f"书名: {store.get_meta('title')}")
        for k in ("genre", "style", "theme", "world", "main_plot"):
            v = s.get(k, "")
            print(f"\n[{k}]\n{v if v else '（未设定）'}")
        if s.get("book_outline"):
            print(f"\n[book_outline]\n{s['book_outline'][:800]}…")
        chars = store.all_characters()
        if chars:
            print(f"\n[人物卡 x{len(chars)}]")
            for n in chars:
                print(f"  - {n}")
    elif args.bible_cmd == "set-title":
        store.set_meta("title", args.value)
    elif args.bible_cmd in ("set-genre", "set-world", "set-style"):
        key = args.bible_cmd.replace("set-", "")
        store.set_setting(key, args.value)
        print(f"✅ 已写入 {key}（{len(args.value)} 字）")
    elif args.bible_cmd == "set-plot":
        store.set_setting("main_plot", args.value)


def cmd_character(args):
    cfg, store, _ = _open_project(args.project)
    if args.char_cmd == "add":
        card = (args.card or "").strip()
        if not card:
            print("请输入人物卡内容（身份/外貌/性格/背景/关系，多行输入，Ctrl+D 结束）：")
            card = sys.stdin.read().strip()
        if not card:
            sys.exit("人物卡为空")
        store.upsert_character(args.name, card)
        print(f"✅ 人物已建档：{args.name}")
    elif args.char_cmd == "list":
        for n, card in store.all_characters().items():
            first = card.strip().splitlines()[0] if card.strip() else ""
            print(f"- {n}  {first[:40]}")
    elif args.char_cmd == "show":
        card = store.get_character(args.name)
        print(card or f"（无 {args.name} 的人物卡）")


def cmd_plan(args):
    cfg, store, llm = _open_project(args.project)
    if args.plan_cmd == "book":
        outline = plan_book(store, llm, cfg)
        print("✅ 全书大纲已生成：")
        print(f"   主题：{outline.get('theme', '')}")
        for v in outline.get("volumes", []):
            print(f"   卷{v.get('number')}《{v.get('title', '')}》 {v.get('summary', '')[:50]}…")
    elif args.plan_cmd == "volume":
        vol = plan_volume(store, llm, cfg, args.number)
        print(f"✅ 卷{args.number}《{vol.get('title', '')}》大纲已生成，"
              f"共 {len(vol.get('chapters', []))} 章梗概")
    elif args.plan_cmd == "chapter":
        detail = plan_chapter(store, llm, cfg, args.number)
        print(f"✅ 第 {args.number} 章细纲《{detail.get('title', '')}》：")
        for i, s in enumerate(detail.get("scenes", []), 1):
            print(f"   场景{i} [{s.get('location')}] {s.get('event', '')}")
    elif args.plan_cmd == "outline-for":
        for n in range(args.number, args.end + 1):
            plan_chapter(store, llm, cfg, n)
            print(f"  ✔ 第 {n} 章细纲完成")
        print(f"✅ 批量细纲完成（{args.number}-{args.end}）")


def cmd_write(args):
    cfg, store, llm = _open_project(args.project)
    review = False if args.no_review else None
    if args.end is None:
        write_chapter(store, llm, cfg, args.start, review=review, quiet=args.quiet)
    else:
        stats = write_range(store, llm, cfg, args.start, args.end,
                            review=review, quiet=args.quiet)
        st = stats["state"]
        print("\n" + "=" * 50)
        print(f"批量写作完成：新写 {stats['written']} 章，用时 {stats['elapsed_min']} 分钟")
        print(f"全书进度：{st['drafted_chapters']} 章 / 累计 {st['total_words']} 字")
        if stats["failed"]:
            print("失败章节：", stats["failed"])


def cmd_review(args):
    cfg, store, llm = _open_project(args.project)
    result = review_chapter(store, llm, cfg, args.number)
    print(f"审查结果：{'通过' if result.get('pass') else '未通过'}，"
          f"评分 {result.get('score')}/10")
    for it in result.get("issues", []):
        print(f"  [{it.get('severity')}/{it.get('type')}] {it.get('problem')}")
        print(f"    原句：{it.get('quote', '')[:50]}")
        print(f"    建议：{it.get('suggestion', '')[:80]}")


def cmd_revise(args):
    cfg, store, llm = _open_project(args.project)
    revised = revise_chapter(store, llm, cfg, args.number)
    print(f"✅ 第 {args.number} 章已修订，当前 {len(revised)} 字")


def cmd_volume_summary(args):
    cfg, store, llm = _open_project(args.project)
    s = summarize_volume(store, llm, cfg, args.number)
    print(f"✅ 卷{args.number}摘要：\n{s}")


def cmd_status(args):
    cfg, store, _ = _open_project(args.project)
    st = store.export_state()
    cpv = store.volume_chapters()
    print("=" * 50)
    print(f"书名: {store.get_meta('title', '未命名')}")
    print(f"进度: 已定稿 {st['latest_final']} 章 / 累计 {st['total_words']} 字")
    print(f"伏笔: {st['open_foreshadows']} 条未回收 | 人物卡 {st['characters']} 张")
    vols = store.all_volumes()
    if vols:
        print(f"卷: {len(vols)} 卷已规划（每卷 {cpv} 章）")
    rows = store.conn.execute(
        "SELECT status, COUNT(*) c FROM chapters GROUP BY status").fetchall()
    print("章节状态: " + ", ".join(f"{r['status']}={r['c']}" for r in rows))
    mock = LLMClient(cfg).mock
    print(f"模型: {'MOCK 模式（未配置 API Key / 设置了 NOVEL_AGENT_MOCK）' if mock else '已配置'}")


def cmd_usage(args):
    cfg, store, _ = _open_project(args.project)
    summary = store.usage_summary()
    if not summary:
        print("（暂无调用记录）")
        return
    total_p = total_c = 0
    for role, u in summary.items():
        p, c = u["prompt_tokens"], u["completion_tokens"]
        total_p += p
        total_c += c
        print(f"{role:<12} 输入 {p:>10,} tok | 输出 {c:>10,} tok")
    print("-" * 46)
    print(f"{'合计':<12} 输入 {total_p:>10,} tok | 输出 {total_c:>10,} tok")


def do_export(store: Store, cfg: Config, fmt: str = "md") -> Path:
    """导出全部已定稿章节，返回文件路径（CLI 与 Web 共用）。"""
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    title = store.get_meta("title", "novel")
    chapters = [c for c in store.chapters_between(1, 10 ** 9) if c["status"] == "final"]
    if not chapters:
        raise ValueError("没有已定稿章节可导出")
    ext = "md" if fmt == "md" else "txt"
    path = out_dir / f"{title}.{ext}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if fmt == "md":
            f.write(f"> {store.get_setting('genre', '')} · "
                    f"{sum(c['word_count'] for c in chapters)} 字\n\n")
        cur_vol = None
        for c in chapters:
            if c["volume"] != cur_vol:
                cur_vol = c["volume"]
                vol = store.get_volume(cur_vol)
                vtitle = vol["title"] if vol else f"第{cur_vol}卷"
                f.write(f"\n{'##' if fmt == 'md' else ''} {vtitle}\n\n")
            heading = "###" if fmt == "md" else ""
            f.write(f"{heading} 第{c['number']}章 {c['title']}\n\n{c['content']}\n\n")
    return path


def cmd_export(args):
    cfg, store, _ = _open_project(args.project)
    try:
        path = do_export(store, cfg, args.format)
    except ValueError as e:
        sys.exit(str(e))
    print(f"✅ 导出完成 → {path}")


def cmd_web(args):
    from .web import WebApp
    WebApp(args.project, host=args.host, port=args.port).serve()


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="novel-agent",
        description="百万字级长篇小说自动写作 Agent（支持 DeepSeek / 千问 / Kimi / GLM 等国产大模型）")
    top.add_argument("--version", action="version", version=f"novel-agent {__version__}")
    sub = top.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="初始化小说项目")
    p.add_argument("project")
    p.add_argument("--title", help="书名（默认取目录名）")
    p.add_argument("--genre", help="题材")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("bible", help="世界观设定管理（bible <set-xxx> <项目> <内容>）")
    p.add_argument("bible_cmd", choices=["show", "set-title", "set-genre",
                                         "set-world", "set-style", "set-plot"])
    p.add_argument("project")
    p.add_argument("value", nargs="?", default="")
    p.set_defaults(func=cmd_bible)

    p = sub.add_parser("character", help="人物卡管理（character add <项目> <名字> [卡片]）")
    p.add_argument("char_cmd", choices=["add", "list", "show"])
    p.add_argument("project")
    p.add_argument("name", nargs="?", default="")
    p.add_argument("card", nargs="?", default="")
    p.set_defaults(func=cmd_character)

    p = sub.add_parser("plan", help="三级规划（plan book <项目> / plan volume <项目> <卷号>）")
    p.add_argument("plan_cmd", choices=["book", "volume", "chapter", "outline-for"])
    p.add_argument("project")
    p.add_argument("number", nargs="?", type=int)
    p.add_argument("end", nargs="?", type=int)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("write", help="写作：write <章号> 或 write <起> <止>")
    _add_common(p)
    p.add_argument("start", type=int)
    p.add_argument("end", nargs="?", type=int)
    p.set_defaults(func=cmd_write)

    p = sub.add_parser("review", help="一致性审查")
    p.add_argument("project")
    p.add_argument("number", type=int)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("revise", help="按审查意见修订")
    p.add_argument("project")
    p.add_argument("number", type=int)
    p.set_defaults(func=cmd_revise)

    p = sub.add_parser("volume-summary", help="卷级摘要聚合")
    p.add_argument("project")
    p.add_argument("number", type=int)
    p.set_defaults(func=cmd_volume_summary)

    p = sub.add_parser("status", help="项目进度")
    p.add_argument("project")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("usage", help="Token 用量统计")
    p.add_argument("project")
    p.set_defaults(func=cmd_usage)

    p = sub.add_parser("export", help="导出全文（md/txt）")
    p.add_argument("project")
    p.add_argument("--format", choices=["md", "txt"], default="md")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("web", help="启动本地 Web 工作台")
    p.add_argument("project")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.set_defaults(func=cmd_web)

    return top


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
