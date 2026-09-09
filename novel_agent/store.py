"""SQLite 存储层 —— 项目全部状态：卷/章/人物/设定/伏笔/时间线/用量。

选择 SQLite 的原因：百万字正文 + 数百章的元数据需要可靠的断点续写，
单文件数据库便于备份、随项目目录迁移。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS volumes (
    number INTEGER PRIMARY KEY,
    title TEXT DEFAULT '',
    outline TEXT DEFAULT '',
    summary TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chapters (
    number INTEGER PRIMARY KEY,
    volume INTEGER NOT NULL DEFAULT 1,
    title TEXT DEFAULT '',
    outline TEXT DEFAULT '',
    content TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'empty',
    word_count INTEGER NOT NULL DEFAULT 0,
    review_json TEXT DEFAULT '',
    updated_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS characters (
    name TEXT PRIMARY KEY,
    card TEXT NOT NULL,
    updated_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS foreshadows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    planted_ch INTEGER,
    resolved_ch INTEGER
);
CREATE TABLE IF NOT EXISTS timeline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter INTEGER,
    event TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT, model TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_chapters_volume ON chapters(volume);
"""

# 世界观设定的固定键
SETTING_KEYS = ("genre", "world", "style", "theme", "main_plot", "book_outline")


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # -- meta ---------------------------------------------------------------

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row and row["value"] is not None else default

    def set_meta(self, key: str, value: str):
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        self.conn.commit()

    # -- settings（世界观设定） ------------------------------------------------

    def get_setting(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str):
        self.conn.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value.strip()))
        self.conn.commit()

    def all_settings(self) -> dict:
        return {r["key"]: r["value"] for r in
                self.conn.execute("SELECT key,value FROM settings")}

    # -- 人物 -----------------------------------------------------------------

    def upsert_character(self, name: str, card: str):
        self.conn.execute(
            "INSERT INTO characters(name,card,updated_at) VALUES(?,?,datetime('now','localtime')) "
            "ON CONFLICT(name) DO UPDATE SET card=excluded.card,"
            "updated_at=excluded.updated_at", (name, card.strip()))
        self.conn.commit()

    def get_character(self, name: str) -> str | None:
        row = self.conn.execute("SELECT card FROM characters WHERE name=?", (name,)).fetchone()
        return row["card"] if row else None

    def all_characters(self) -> dict[str, str]:
        return {r["name"]: r["card"] for r in
                self.conn.execute("SELECT name,card FROM characters ORDER BY name")}

    # -- 卷 -------------------------------------------------------------------

    def upsert_volume(self, number: int, title: str = "", outline: str = "",
                      summary: str | None = None):
        self.conn.execute(
            "INSERT INTO volumes(number,title,outline,summary) VALUES(?,?,?,?) "
            "ON CONFLICT(number) DO UPDATE SET title=excluded.title, outline=excluded.outline"
            + (", summary=excluded.summary" if summary is not None else ""),
            (number, title, outline, summary if summary is not None else ""))
        self.conn.commit()

    def get_volume(self, number: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM volumes WHERE number=?", (number,)).fetchone()

    def all_volumes(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM volumes ORDER BY number").fetchall()

    # -- 章 -------------------------------------------------------------------

    def ensure_chapter(self, number: int, volume: int = 1):
        self.conn.execute(
            "INSERT OR IGNORE INTO chapters(number,volume) VALUES(?,?)", (number, volume))
        self.conn.commit()

    def update_chapter(self, number: int, **fields):
        if not fields:
            return
        allowed = ("volume", "title", "outline", "content", "summary", "status",
                   "word_count", "review_json", "updated_at")
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise ValueError(f"非法章节字段: {k}")
            sets.append(f"{k}=?")
            vals.append(v)
        sets.append("updated_at=datetime('now','localtime')")
        vals.append(number)
        self.conn.execute(f"UPDATE chapters SET {', '.join(sets)} WHERE number=?", vals)
        self.conn.commit()

    def get_chapter(self, number: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM chapters WHERE number=?", (number,)).fetchone()

    def chapters_between(self, a: int, b: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chapters WHERE number BETWEEN ? AND ? ORDER BY number", (a, b)
        ).fetchall()

    def latest_final_chapter(self) -> int:
        row = self.conn.execute(
            "SELECT MAX(number) AS n FROM chapters WHERE status='final'").fetchone()
        return row["n"] or 0

    def volume_of(self, number: int) -> int:
        """按每卷章数推算章节所属卷号。"""
        cpv = self.volume_chapters()
        return (number - 1) // cpv + 1 if cpv > 0 else 1

    def volume_chapters(self) -> int:
        try:
            return int(self.get_meta("chapters_per_volume", "40"))
        except ValueError:
            return 40

    # -- 伏笔 -------------------------------------------------------------------

    def add_foreshadow(self, content: str, planted_ch: int | None):
        self.conn.execute("INSERT INTO foreshadows(content,planted_ch) VALUES(?,?)",
                          (content, planted_ch))
        self.conn.commit()

    def resolve_foreshadow(self, content_prefix: str, resolved_ch: int) -> bool:
        """按内容前缀匹配并标记回收。"""
        row = self.conn.execute(
            "SELECT id FROM foreshadows WHERE resolved_ch IS NULL AND ? LIKE content || '%'",
            (content_prefix,)).fetchone()
        if row:
            self.conn.execute("UPDATE foreshadows SET resolved_ch=? WHERE id=?",
                              (resolved_ch, row["id"]))
            self.conn.commit()
            return True
        return False

    def open_foreshadows(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM foreshadows WHERE resolved_ch IS NULL ORDER BY planted_ch"
        ).fetchall()

    def all_foreshadows(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM foreshadows ORDER BY planted_ch, id").fetchall()

    # -- 时间线 ---------------------------------------------------------------

    def add_timeline(self, chapter: int, event: str):
        self.conn.execute("INSERT INTO timeline(chapter,event) VALUES(?,?)",
                          (chapter, event))
        self.conn.commit()

    # -- 用量统计 ---------------------------------------------------------------

    def log_usage(self, role: str, model: str, usage: dict):
        if not usage:
            return
        self.conn.execute(
            "INSERT INTO usage_log(role,model,prompt_tokens,completion_tokens) "
            "VALUES(?,?,?,?)",
            (role, model, usage.get("prompt_tokens") or 0,
             usage.get("completion_tokens") or 0))
        self.conn.commit()

    def usage_summary(self) -> dict:
        rows = self.conn.execute(
            "SELECT role, SUM(prompt_tokens) p, SUM(completion_tokens) c "
            "FROM usage_log GROUP BY role").fetchall()
        return {r["role"]: {"prompt_tokens": r["p"] or 0,
                            "completion_tokens": r["c"] or 0} for r in rows}

    # -- 导出 -------------------------------------------------------------------

    def export_state(self) -> dict:
        """把轻量状态导出为 dict（用于 status 命令展示）。"""
        total_words = self.conn.execute(
            "SELECT SUM(word_count) w FROM chapters WHERE status='final'").fetchone()["w"] or 0
        drafted = self.conn.execute(
            "SELECT COUNT(*) c FROM chapters WHERE status IN ('drafted','final')").fetchone()["c"]
        return {
            "total_words": total_words,
            "drafted_chapters": drafted,
            "latest_final": self.latest_final_chapter(),
            "open_foreshadows": len(self.open_foreshadows()),
            "characters": len(self.all_characters()),
        }


def dump_json_safe(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
