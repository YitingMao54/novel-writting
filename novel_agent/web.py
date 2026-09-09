"""Web 工作台 —— novel-agent 的本地网页界面。

启动：novel-agent web <项目目录> [--port 8765]
功能：设定管理 / 人物卡 / 三级规划 / 流式写作 / 在线阅读 / 伏笔台账 / 导出。
仅用 Python 标准库 http.server，前端为内嵌单页应用，无额外依赖。
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from . import planner, reviewer, writer
from .config import Config
from .llm import LLMClient
from .store import Store


# ---------------------------------------------------------------------------
# 状态快照（前端全量数据）
# ---------------------------------------------------------------------------

def snapshot(store: Store, cfg: Config, project_dir: Path) -> dict:
    settings = store.all_settings()
    vols = []
    for v in store.all_volumes():
        cpv = store.volume_chapters()
        chs = store.chapters_between((v["number"] - 1) * cpv + 1, v["number"] * cpv)
        vols.append({
            "number": v["number"], "title": v["title"], "outline": v["outline"],
            "summary": v["summary"],
            "chapters": [{"number": c["number"], "title": c["title"],
                          "status": c["status"], "word_count": c["word_count"],
                          "brief": (c["outline"] or "")[:120]}
                         for c in chs if c["status"] != "empty" or c["outline"]],
    })
    # 未归入任何已规划卷的散章（自动细纲链产生的）
    known = {c["number"] for v in vols for c in v["chapters"]}
    loose = [c for c in store.chapters_between(1, 10 ** 6) if c["number"] not in known]
    if loose:
        vols.append({"number": None, "title": "未分卷章节", "outline": "", "summary": "",
                     "chapters": [{"number": c["number"], "title": c["title"],
                                   "status": c["status"], "word_count": c["word_count"],
                                   "brief": (c["outline"] or "")[:120]} for c in loose]})
    st = store.export_state()
    return {
        "project": str(project_dir),
        "title": store.get_meta("title", "未命名"),
        "settings": settings,
        "characters": [{"name": n, "card": c} for n, c in store.all_characters().items()],
        "volumes": vols,
        "foreshadows": [{"id": r["id"], "content": r["content"],
                         "planted_ch": r["planted_ch"], "resolved_ch": r["resolved_ch"]}
                        for r in store.all_foreshadows()],
        "usage": store.usage_summary(),
        "stats": {**st,
                  "target_words": cfg.get_writing("chapters_per_volume", 40)
                                  * cfg.get_writing("volume_count", 10)
                                  * cfg.get_writing("chapter_words", 3000)},
        "writing": {k: cfg.get_writing(k) for k in
                    ("chapter_words", "chapters_per_volume", "volume_count")},
        "mock": LLMClient(cfg).mock,
    }


# ---------------------------------------------------------------------------
# HTTP 服务
# ---------------------------------------------------------------------------

class WebApp:
    """轻量 Web 服务：每请求独立 Store 连接（SQLite WAL 支持并发读写）。"""

    def __init__(self, project_dir: str, host: str = "127.0.0.1", port: int = 8765):
        self.project_dir = Path(project_dir).expanduser().resolve()
        if not (self.project_dir / "config.yaml").exists():
            raise SystemExit(f"错误：{self.project_dir} 不是 novel-agent 项目（缺 config.yaml）")
        self.cfg = Config(self.project_dir)
        self.llm = LLMClient(self.cfg, verbose=False)
        self.host, self.port = host, port
        self.busy = threading.Lock()  # 写作/规划互斥

    def serve(self):
        print(f"✅ 小说工作台已启动: http://{self.host}:{self.port}")
        print(f"   项目：{self.project_dir}（书名：{self.store_title()}）")
        if self.llm.mock:
            print("   ⚠️ 当前为 MOCK 模式（未检测到 API Key / 设置了 NOVEL_AGENT_MOCK），"
                  "生成内容为演示数据")
        server = ThreadingHTTPServer((self.host, self.port), make_handler(self))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")

    def store_title(self) -> str:
        store = Store(self.cfg.db_path)
        try:
            return store.get_meta("title", "未命名")
        finally:
            store.close()


def make_handler(app: WebApp):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # -- plumbing -------------------------------------------------------

        def _store(self) -> Store:
            return Store(app.cfg.db_path)

        def send_json(self, obj, code: int = 200):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                return json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return {}

        def log_message(self, fmt, *args):  # 静默访问日志
            pass

        # -- routing --------------------------------------------------------

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = PAGE_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            qs = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/state":
                    store = self._store()
                    try:
                        self.send_json(snapshot(store, app.cfg, app.project_dir))
                    finally:
                        store.close()
                elif parsed.path == "/api/chapter":
                    self.send_json(self.api_chapter(qs))
                else:
                    self.send_json({"error": f"未知路由 {parsed.path}"}, 404)
            except (BrokenPipeError, ConnectionAbortedError):
                pass
            except Exception as e:  # noqa: BLE001
                self._safe_error(e)

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            body = self.read_json()
            handlers = {
                "/api/settings": self.h_settings,
                "/api/character": self.h_character,
                "/api/plan/book": self.h_plan_book,
                "/api/plan/volume": self.h_plan_volume,
                "/api/plan/chapter": self.h_plan_chapter,
                "/api/review": self.h_review,
                "/api/revise": self.h_revise,
                "/api/export": self.h_export,
                "/api/write": self.h_write,
            }
            fn = handlers.get(parsed.path)
            if not fn:
                self.send_json({"error": f"未知路由 {parsed.path}"}, 404)
                return
            try:
                fn(body)
            except (BrokenPipeError, ConnectionAbortedError):
                pass
            except Exception as e:  # noqa: BLE001
                self._safe_error(e)

        def _safe_error(self, e: Exception):
            try:
                self.send_json({"error": str(e)}, 500)
            except OSError:
                pass

        def _acquire(self) -> bool:
            """抢占全局任务锁；失败则返回 False（由调用方回 409）。"""
            return app.busy.acquire(blocking=False)

        # -- GET handlers ---------------------------------------------------

        def api_chapter(self, qs) -> dict:
            n = int(qs.get("n", ["0"])[0])
            store = self._store()
            try:
                ch = store.get_chapter(n)
                if not ch:
                    return {"error": f"第 {n} 章不存在"}, 404
                detail = {}
                try:
                    detail = json.loads(ch["outline"]) if ch["outline"] else {}
                except json.JSONDecodeError:
                    detail = {"brief": ch["outline"]}
                review = {}
                try:
                    review = json.loads(ch["review_json"]) if ch["review_json"] else {}
                except json.JSONDecodeError:
                    pass
                return {"number": n, "volume": ch["volume"], "title": ch["title"],
                        "status": ch["status"], "word_count": ch["word_count"],
                        "outline": detail, "content": ch["content"],
                        "summary": ch["summary"], "review": review}
            finally:
                store.close()

        # -- POST handlers ----------------------------------------------------

        def h_settings(self, body):
            key, value = body.get("key", ""), body.get("value", "")
            store = self._store()
            try:
                if key == "title":
                    store.set_meta("title", value)
                elif key in ("genre", "world", "style", "main_plot"):
                    store.set_setting(key, value)
                else:
                    self.send_json({"error": f"不支持的设定键: {key}"}, 400)
                    return
            finally:
                store.close()
            self.send_json({"ok": True})

        def h_character(self, body):
            name, card = (body.get("name") or "").strip(), (body.get("card") or "").strip()
            if not name or not card:
                self.send_json({"error": "需要 name 与 card"}, 400)
                return
            store = self._store()
            try:
                store.upsert_character(name, card)
            finally:
                store.close()
            self.send_json({"ok": True})

        def h_plan_book(self, body):
            if not self._acquire():
                self.send_json({"error": "有任务正在进行中，请稍候"}, 409)
                return
            store = self._store()
            try:
                out = planner.plan_book(store, app.llm, app.cfg)
                self.send_json({"ok": True, "outline": out})
            finally:
                store.close()
                app.busy.release()

        def h_plan_volume(self, body):
            try:
                n = int(body.get("number"))
            except (TypeError, ValueError):
                self.send_json({"error": "需要卷号 number"}, 400)
                return
            if not self._acquire():
                self.send_json({"error": "有任务正在进行中，请稍候"}, 409)
                return
            store = self._store()
            try:
                out = planner.plan_volume(store, app.llm, app.cfg, n)
                self.send_json({"ok": True, "volume": out})
            finally:
                store.close()
                app.busy.release()

        def h_plan_chapter(self, body):
            try:
                n = int(body.get("number"))
            except (TypeError, ValueError):
                self.send_json({"error": "需要章号 number"}, 400)
                return
            if not self._acquire():
                self.send_json({"error": "有任务正在进行中，请稍候"}, 409)
                return
            store = self._store()
            try:
                out = planner.plan_chapter(store, app.llm, app.cfg, n)
                self.send_json({"ok": True, "detail": out})
            finally:
                store.close()
                app.busy.release()

        def h_review(self, body):
            n = int(body.get("number", 0))
            store = self._store()
            try:
                result = reviewer.review_chapter(store, app.llm, app.cfg, n)
            finally:
                store.close()
            self.send_json({"ok": True, "review": result})

        def h_revise(self, body):
            n = int(body.get("number", 0))
            store = self._store()
            try:
                revised = reviewer.revise_chapter(store, app.llm, app.cfg, n)
            finally:
                store.close()
            self.send_json({"ok": True, "word_count": len(revised)})

        def h_export(self, body):
            from .cli import do_export
            store = self._store()
            try:
                path = do_export(store, app.cfg, body.get("format", "md"))
            finally:
                store.close()
            self.send_json({"ok": True, "path": str(path)})

        def h_write(self, body):
            """流式写作：text/event-stream，逐条推送 log/chunk/done/error 事件。"""
            try:
                start = int(body.get("start"))
                end = int(body.get("end", start))
            except (TypeError, ValueError):
                self.send_json({"error": "需要 start（章号）"}, 400)
                return
            review = None if body.get("review", True) else False

            if not self._acquire():
                self.send_json({"error": "有写作任务正在进行中"}, 409)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(obj: dict):
                self.wfile.write(
                    f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()

            def on_log(msg: str):
                emit({"type": "log", "text": msg})

            def on_chunk(piece: str):
                emit({"type": "chunk", "text": piece})

            store = self._store()
            try:
                stats = writer.write_range(store, app.llm, app.cfg, start, end,
                                           review=review, quiet=True,
                                           log_callback=on_log, chunk_callback=on_chunk)
                emit({"type": "done", "stats": stats})
            except ConnectionAbortedError:
                pass  # 前端关闭页面，写作已到中断点为止
            except Exception as e:  # noqa: BLE001
                try:
                    emit({"type": "error", "text": str(e)})
                except OSError:
                    pass
            finally:
                store.close()
                app.busy.release()

    return Handler


# ---------------------------------------------------------------------------
# 前端单页应用
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>novel-agent · 小说工作台</title>
<style>
:root{
  --bg:#f6f4ef; --panel:#fffdf8; --ink:#2c2822; --sub:#8d8579;
  --accent:#b0483c; --accent-soft:#f3e3e0; --blue:#3d5a80; --blue-soft:#e4ebf3;
  --line:#e7e1d4; --ok:#4a7c59; --ok-soft:#e2eee5; --warn:#b07f28;
  --serif:"Songti SC","STSong","Noto Serif SC",serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  background:var(--bg);color:var(--ink);height:100vh;display:flex;flex-direction:column;overflow:hidden}
button{font:inherit;cursor:pointer;border:1px solid var(--line);background:var(--panel);
  color:var(--ink);border-radius:8px;padding:6px 14px;transition:.15s}
button:hover{border-color:var(--accent);color:var(--accent)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.primary:hover{opacity:.88;color:#fff}
button.blue{background:var(--blue);border-color:var(--blue);color:#fff}
button:disabled{opacity:.45;cursor:not-allowed}
input,textarea,select{font:inherit;color:var(--ink);border:1px solid var(--line);
  border-radius:8px;padding:8px 10px;background:#fff;width:100%}
textarea{resize:vertical;line-height:1.7}
header{display:flex;align-items:center;gap:16px;padding:12px 22px;
  background:var(--panel);border-bottom:1px solid var(--line);flex-shrink:0}
.brand{font-family:var(--serif);font-size:20px;font-weight:700;letter-spacing:2px}
.brand em{font-style:normal;color:var(--accent)}
.brand small{font-size:12px;color:var(--sub);font-weight:400;margin-left:8px;letter-spacing:0}
#bookTitle{font-family:var(--serif);font-size:17px;color:var(--ink)}
.hspace{flex:1}
.badge{font-size:11px;padding:3px 9px;border-radius:99px;background:var(--ok-soft);color:var(--ok)}
.badge.mock{background:var(--accent-soft);color:var(--accent)}
main{flex:1;display:flex;min-height:0}
aside{width:292px;flex-shrink:0;background:var(--panel);border-right:1px solid var(--line);
  display:flex;flex-direction:column;min-height:0}
.stats{padding:14px 16px;border-bottom:1px solid var(--line)}
.stats .row{display:flex;justify-content:space-between;font-size:12px;color:var(--sub);margin-bottom:6px}
.stats .row b{color:var(--ink);font-weight:600}
.bar{height:7px;background:var(--line);border-radius:99px;overflow:hidden;margin:4px 0 8px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),#d98a68);border-radius:99px}
.chapters{flex:1;overflow-y:auto;padding:10px 12px 20px}
.vol{margin-top:10px}
.vol-h{display:flex;align-items:center;justify-content:space-between;
  font-family:var(--serif);font-weight:700;font-size:14px;padding:4px 6px;cursor:pointer}
.vol-h .mini{font-size:11px;color:var(--accent);font-weight:400}
.ch{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:8px;
  font-size:13px;cursor:pointer;color:var(--ink)}
.ch:hover{background:var(--bg)}
.ch .dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.dot.empty{background:#d8d2c6}.dot.outlined{background:var(--blue)}
.dot.drafted{background:var(--warn)}.dot.final{background:var(--ok)}
.ch .wn{margin-left:auto;font-size:11px;color:var(--sub)}
section{flex:1;display:flex;flex-direction:column;min-width:0;min-height:0}
nav.tabs{display:flex;gap:2px;padding:10px 20px 0;background:var(--panel);
  border-bottom:1px solid var(--line);flex-shrink:0}
nav.tabs button{border:none;background:none;border-radius:8px 8px 0 0;padding:9px 18px;
  color:var(--sub);border-bottom:2px solid transparent}
nav.tabs button.on{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
.pane{flex:1;overflow-y:auto;padding:22px 26px;display:none;min-height:0}
.pane.on{display:block}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:18px 20px;margin-bottom:16px}
.card h3{font-family:var(--serif);font-size:15px;margin-bottom:10px;display:flex;
  align-items:center;gap:10px}
.card h3 .act{margin-left:auto;display:flex;gap:8px}
label.f{display:block;font-size:12px;color:var(--sub);margin:10px 0 5px}
.char-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.char-card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:#fff;
  font-size:13px;white-space:pre-wrap;line-height:1.7;max-height:220px;overflow-y:auto;position:relative}
.char-card b{font-family:var(--serif);font-size:15px}
.char-card .edit{position:absolute;top:8px;right:8px;font-size:11px;color:var(--blue);
  cursor:pointer;background:none;border:none;padding:2px 6px}
.ol-row{padding:10px 4px;border-bottom:1px dashed var(--line);font-size:13px;line-height:1.65}
.ol-row b{font-family:var(--serif)}
.ol-row .brief{color:var(--sub)}
.write-grid{display:grid;grid-template-columns:300px 1fr;gap:18px;height:100%}
.wleft{display:flex;flex-direction:column;gap:12px}
.wright{display:flex;flex-direction:column;min-height:0;border:1px solid var(--line);
  border-radius:12px;background:var(--panel);overflow:hidden}
.wright .whd{padding:10px 16px;border-bottom:1px solid var(--line);font-size:13px;color:var(--sub);
  display:flex;gap:10px;align-items:center}
#streamArea{flex:1;overflow-y:auto;padding:16px 22px;font-family:var(--serif);
  font-size:15px;line-height:2;white-space:pre-wrap}
#logArea{max-height:130px;overflow-y:auto;padding:8px 16px;border-top:1px solid var(--line);
  font-size:12px;color:var(--sub);font-family:ui-monospace,Menlo,monospace;line-height:1.7}
.reader{max-width:760px;margin:0 auto;background:var(--panel);border:1px solid var(--line);
  border-radius:14px;padding:40px 52px;font-family:var(--serif);font-size:16px;line-height:2.1;
  white-space:pre-wrap}
.reader h2{font-size:20px;text-align:center;margin-bottom:6px}
.reader .meta{text-align:center;color:var(--sub);font-size:12px;font-family:sans-serif;margin-bottom:22px}
.score{display:inline-block;font-size:12px;padding:2px 10px;border-radius:99px;margin-left:8px}
.score.good{background:var(--ok-soft);color:var(--ok)}
.score.bad{background:var(--accent-soft);color:var(--accent)}
.issue{border-left:3px solid var(--warn);background:#fdf8ee;border-radius:0 8px 8px 0;
  padding:10px 14px;margin:10px 0;font-size:13px;line-height:1.7}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--sub);font-weight:500;font-size:12px}
.tag{font-size:11px;padding:2px 9px;border-radius:99px}
.tag.open{background:var(--accent-soft);color:var(--accent)}
.tag.resolved{background:var(--ok-soft);color:var(--ok)}
#toast{position:fixed;bottom:26px;left:50%;transform:translateX(-50%) translateY(80px);
  background:var(--ink);color:#fff;padding:10px 22px;border-radius:10px;font-size:13px;
  opacity:0;transition:.3s;z-index:99;max-width:70vw}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.hint{font-size:12px;color:var(--sub);line-height:1.7}
.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--accent);
  border-top-color:transparent;border-radius:50%;animation:r 0.8s linear infinite;vertical-align:-2px}
@keyframes r{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header>
  <div class="brand">墨<em>阁</em><small>novel-agent 长篇写作工作台</small></div>
  <div id="bookTitle">—</div>
  <div class="hspace"></div>
  <span id="modelBadge" class="badge">加载中</span>
  <button onclick="doExport()">导出全文</button>
</header>
<main>
<aside>
  <div class="stats">
    <div class="row"><span>已定稿章节</span><b id="stChapters">—</b></div>
    <div class="row"><span>累计字数</span><b id="stWords">—</b></div>
    <div class="bar"><i id="stBar" style="width:0%"></i></div>
    <div class="row"><span>目标规模</span><b id="stTarget" style="font-size:11px">—</b></div>
  </div>
  <div class="chapters" id="chapterTree">加载中…</div>
</aside>
<section>
<nav class="tabs">
  <button data-t="setting" class="on">设定</button>
  <button data-t="outline">大纲</button>
  <button data-t="write">写作台</button>
  <button data-t="read">阅读</button>
  <button data-t="foreshadow">伏笔 · 用量</button>
</nav>

<div class="pane on" id="pane-setting">
  <div class="card">
    <h3>世界设定
      <span class="act"><button class="primary" onclick="saveSetting('world')">保存世界设定</button></span></h3>
    <textarea id="setWorld" rows="7" placeholder="世界观、力量体系、地理势力、核心规则……"></textarea>
  </div>
  <div class="card">
    <h3>文风与题材
      <span class="act"><button class="primary" onclick="saveSetting('style')">保存文风</button>
      <button class="primary" onclick="saveSetting('genre')">保存题材</button></span></h3>
    <textarea id="setStyle" rows="3" placeholder="叙事视角、语言风格、节奏偏好……"></textarea>
    <label class="f">题材 / 类型</label>
    <input id="setGenre" placeholder="东方玄幻 / 都市脑洞 / 星际科幻 ……">
  </div>
  <div class="card">
    <h3>人物卡 <span id="charCount" class="hint"></span>
      <span class="act"><button class="blue" onclick="showCharEditor()">＋ 新建人物</button></span></h3>
    <div id="charEditor" style="display:none;margin-bottom:14px">
      <label class="f">姓名</label><input id="charName" placeholder="如：林昭">
      <label class="f">人物卡（身份 / 外貌 / 性格 / 背景 / 关系 / 当前状态）</label>
      <textarea id="charCard" rows="5"></textarea>
      <div style="margin-top:10px;display:flex;gap:8px">
        <button class="primary" onclick="saveCharacter()">保存人物</button>
        <button onclick="hideCharEditor()">取消</button>
      </div>
    </div>
    <div class="char-grid" id="charGrid"></div>
  </div>
</div>

<div class="pane" id="pane-outline">
  <div class="card">
    <h3>全书大纲
      <span class="act"><button class="primary" id="btnPlanBook" onclick="planBook()">生成全书大纲</button></span></h3>
    <div id="bookOutline" class="hint">尚无全书大纲。设定与人物就绪后，点击右上角生成。</div>
  </div>
  <div class="card" id="volCard">
    <h3>分卷规划
      <span class="act">
        <input id="volNum" type="number" min="1" value="1" style="width:64px">
        <button class="primary" id="btnPlanVol" onclick="planVolume()">生成该卷大纲</button>
      </span></h3>
    <div id="volList" class="hint">尚无卷大纲。</div>
  </div>
</div>

<div class="pane" id="pane-write">
  <div class="write-grid">
    <div class="wleft">
      <div class="card" style="margin:0">
        <h3>写作任务</h3>
        <label class="f">起始章</label><input id="wStart" type="number" min="1" value="1">
        <label class="f">结束章（批量连写，相同则为单章）</label>
        <input id="wEnd" type="number" min="1" value="1">
        <label class="f" style="display:flex;align-items:center;gap:6px;margin-top:12px">
          <input type="checkbox" id="wReview" checked style="width:auto"> 定稿前一致性审查</label>
        <div style="margin-top:14px;display:flex;gap:8px">
          <button class="primary" id="btnWrite" onclick="startWrite()">▶ 开始写作</button>
        </div>
        <p class="hint" style="margin-top:10px">
          已定稿章节自动跳过；中途关闭页面写作会继续在后台完成并写入数据库。</p>
      </div>
      <div class="card" style="margin:0">
        <h3>手动质检</h3>
        <label class="f">章号</label><input id="rvNum" type="number" min="1" value="1">
        <div style="margin-top:10px;display:flex;gap:8px">
          <button onclick="doReview()">审查</button>
          <button onclick="doRevise()">按意见修订</button>
        </div>
        <div id="rvResult" style="margin-top:10px"></div>
      </div>
    </div>
    <div class="wright">
      <div class="whd"><span id="writeStatus">待命</span><span class="hspace"></span>
        <button onclick="clearStream()">清屏</button></div>
      <div id="streamArea"></div>
      <div id="logArea"></div>
    </div>
  </div>
</div>

<div class="pane" id="pane-read">
  <div class="card" style="display:flex;gap:10px;align-items:center">
    <select id="readSel" style="max-width:340px" onchange="loadReader()"></select>
    <span id="readMeta" class="hint"></span>
  </div>
  <div id="readerWrap"><div class="reader"><h2>从左侧或下拉选择章节</h2></div></div>
</div>

<div class="pane" id="pane-foreshadow">
  <div class="card">
    <h3>伏笔台账</h3>
    <table><thead><tr><th style="width:90px">埋设</th><th>内容</th><th style="width:90px">状态</th></tr></thead>
    <tbody id="fTable"></tbody></table>
  </div>
  <div class="card">
    <h3>Token 用量（按角色）</h3>
    <table><thead><tr><th>角色</th><th>输入 tokens</th><th>输出 tokens</th></tr></thead>
    <tbody id="uTable"></tbody></table>
  </div>
</div>
</section>
</main>
<div id="toast"></div>

<script>
let S=null, curTab='setting';
const $=s=>document.querySelector(s);
const esc=t=>(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');
const fmtN=n=>(n||0).toLocaleString('zh-CN');

function toast(t){const el=$('#toast');el.textContent=t;el.classList.add('show');
  clearTimeout(el._t);el._t=setTimeout(()=>el.classList.remove('show'),2600);}

async function api(path,body,method){
  const opt=method==='GET'?{}:{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body||{})};
  const r=await fetch(path+(method==='GET'&&path.includes('?')?'':''),opt);
  const j=await r.json().catch(()=>({error:'响应解析失败'}));
  if(j.error)toast('⚠️ '+j.error);
  return j;
}

async function refresh(){
  S=await api('/api/state',null,'GET');
  $('#bookTitle').textContent='《'+S.title+'》';
  const mb=$('#modelBadge');
  if(S.mock){mb.textContent='MOCK 演示模式';mb.classList.add('mock');}
  else mb.textContent='模型已连接';
  $('#stChapters').textContent=S.stats.latest_final+' 章 / '+S.stats.total_words+' 字';
  $('#stWords').textContent=fmtN(S.stats.total_words)+' 字';
  $('#stTarget').textContent='约 '+(S.stats.target_words/10000).toFixed(0)+' 万字 · '
    +S.writing.volume_count+'卷 × '+S.writing.chapters_per_volume+'章';
  $('#stBar').style.width=Math.min(100,S.stats.total_words/Math.max(1,S.stats.target_words)*100)+'%';
  renderTree();renderChars();renderOutline();renderForeshadow();renderUsage();renderReadSel();
}

/* ---------- 左侧章节树 ---------- */
function renderTree(){
  const box=$('#chapterTree');box.innerHTML='';
  const q=(window._q||'').trim();
  S.volumes.forEach(v=>{
    const chs=v.chapters.filter(c=>!q||(c.title||'').includes(q)||String(c.number).includes(q));
    if(!chs.length)return;
    const vd=document.createElement('div');vd.className='vol';
    const name=v.number?('第'+v.number+'卷 · '+(v.title||'未命名')):(v.title||'散章');
    vd.innerHTML='<div class="vol-h"><span>'+esc(name)+'</span><span class="mini">'
      +chs.filter(c=>c.status==='final').length+'/'+chs.length+' 章</span></div>';
    chs.forEach(c=>{
      const d=document.createElement('div');d.className='ch';
      d.innerHTML='<span class="dot '+c.status+'"></span><span>第'+c.number+'章 '
        +esc(c.title||'（未命名）')+'</span><span class="wn">'+(c.word_count?c.word_count+'字':'')+'</span>';
      d.onclick=()=>{switchTab('read');$('#readSel').value=c.number;loadReader();};
      vd.appendChild(d);
    });
    box.appendChild(vd);
  });
  if(!box.children.length)box.innerHTML='<p class="hint" style="padding:10px">没有章节。去「大纲」生成卷大纲，或到「写作台」直接开写（细纲会自动补齐）。</p>';
}
document.addEventListener('input',e=>{if(e.target.id==='chapterTree')return;});

/* ---------- 设定 ---------- */
function renderChars(){
  $('#setWorld').value=S.settings.world||'';
  $('#setStyle').value=S.settings.style||'';
  $('#setGenre').value=S.settings.genre||'';
  const g=$('#charGrid');g.innerHTML='';
  $('#charCount').textContent='共 '+S.characters.length+' 位';
  S.characters.forEach(c=>{
    const d=document.createElement('div');d.className='char-card';
    d.innerHTML='<button class="edit" data-n="'+esc(c.name)+'">编辑</button><b>'+esc(c.name)
      +'</b>\n'+esc(c.card);
    d.querySelector('.edit').onclick=()=>showCharEditor(c.name,c.card);
    g.appendChild(d);
  });
}
async function saveSetting(key){
  const val=key==='world'?$('#setWorld').value:key==='style'?$('#setStyle').value:$('#setGenre').value;
  await api('/api/settings',{key,value});
  toast('✅ 已保存');refresh();
}
function showCharEditor(name,card){
  $('#charEditor').style.display='block';
  $('#charName').value=name||'';$('#charCard').value=card||'';
  $('#charName').disabled=!!name;
  $('#charEditor').scrollIntoView({behavior:'smooth'});
}
function hideCharEditor(){$('#charEditor').style.display='none';}
async function saveCharacter(){
  const r=await api('/api/character',{name:$('#charName').value,card:$('#charCard').value});
  if(r.ok){toast('✅ 人物已保存');hideCharEditor();refresh();}
}

/* ---------- 大纲 ---------- */
function renderOutline(){
  const bo=S.settings.book_outline;
  const el=$('#bookOutline');
  if(!bo){el.innerHTML='尚无全书大纲。设定与人物就绪后，点击右上角生成。';}
  else{try{const o=JSON.parse(bo);
    el.innerHTML='<p style="line-height:1.9;margin-bottom:8px"><b>主题：</b>'+esc(o.theme)+'</p>'
      +'<p style="line-height:1.9;margin-bottom:8px"><b>主线：</b>'+esc(o.main_plot)+'</p>'
      +'<p style="line-height:1.9"><b>结局：</b>'+esc(o.ending)+'</p>';
  }catch(e){el.textContent=bo.slice(0,600);}}
  const vl=$('#volList');vl.innerHTML='';
  if(!S.volumes.length){vl.innerHTML='尚无卷大纲。输入卷号点击「生成该卷大纲」。';return;}
  S.volumes.forEach(v=>{
    if(v.number==null)return;
    const d=document.createElement('div');d.className='ol-row';
    d.innerHTML='<b>第'+v.number+'卷 · '+esc(v.title||'')+'</b>'
      +'<div class="brief">'+esc(v.outline||'')+'</div>'
      +(v.chapters.length?'<details style="margin-top:6px"><summary style="cursor:pointer;color:var(--blue);font-size:12px">'
        +v.chapters.length+' 章梗概</summary>'
        +v.chapters.map(c=>'<div style="padding:4px 0;color:var(--sub)"><b style="color:var(--ink)">第'
          +c.number+'章 '+esc(c.title||'')+'</b> · '+esc(c.brief||'')+'</div>').join('')+'</details>':'');
    vl.appendChild(d);
  });
}
async function planBook(){
  const b=$('#btnPlanBook');b.disabled=true;b.innerHTML='<span class="spin"></span> 规划中…';
  const r=await api('/api/plan/book');b.disabled=false;b.textContent='生成全书大纲';
  if(r.ok){toast('✅ 全书大纲已生成');refresh();}
}
async function planVolume(){
  const b=$('#btnPlanVol');b.disabled=true;b.innerHTML='<span class="spin"></span> 规划中…';
  const r=await api('/api/plan/volume',{number:parseInt($('#volNum').value)||1});
  b.disabled=false;b.textContent='生成该卷大纲';
  if(r.ok){toast('✅ 卷大纲已生成');refresh();}
}

/* ---------- 写作台 ---------- */
function switchTab(t){
  curTab=t;
  document.querySelectorAll('nav.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));
  document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('on',p.id==='pane-'+t));
}
document.querySelectorAll('nav.tabs button').forEach(b=>b.onclick=()=>switchTab(b.dataset.t));

async function startWrite(){
  const start=parseInt($('#wStart').value),end=parseInt($('#wEnd').value)||start;
  if(!start||start<1)return toast('请输入章号');
  if(end-start>=10&&!confirm('即将批量写作第 '+start+'～'+end+' 章（共 '+(end-start+1)
    +' 章，真实调用会产生 API 费用）。继续？'))return;
  const btn=$('#btnWrite');btn.disabled=true;
  $('#writeStatus').innerHTML='<span class="spin"></span> 写作中…';
  const area=$('#streamArea'),log=$('#logArea');
  area.textContent='';log.textContent='';
  try{
    const resp=await fetch('/api/write',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({start,end,review:$('#wReview').checked})});
    const reader=resp.body.getReader(),dec=new TextDecoder();let buf='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=dec.decode(value,{stream:true});let i;
      while((i=buf.indexOf('\n\n'))>=0){
        const line=buf.slice(0,i).trim();buf=buf.slice(i+2);
        if(!line.startsWith('data: '))continue;
        const ev=JSON.parse(line.slice(6));
        if(ev.type==='log'){const d=document.createElement('div');d.textContent=ev.text;log.appendChild(d);log.scrollTop=1e9;}
        else if(ev.type==='chunk'){area.textContent+=ev.text;area.scrollTop=1e9;}
        else if(ev.type==='done'){$('#writeStatus').textContent='✅ 完成：新写 '
          +ev.stats.written+' 章 / 用时 '+ev.stats.elapsed_min+' 分钟';}
        else if(ev.type==='error'){$('#writeStatus').textContent='❌ 出错：'+ev.text;}
      }
    }
  }catch(e){$('#writeStatus').textContent='❌ '+e;}
  btn.disabled=false;refresh();
}
function clearStream(){$('#streamArea').textContent='';$('#logArea').textContent='';}

async function doReview(){
  const r=await api('/api/review',{number:parseInt($('#rvNum').value)});
  if(!r.ok)return;
  const v=r.review;
  $('#rvResult').innerHTML='<b>'+(v.pass?'✅ 通过':'⚠️ 未通过')+'</b> 评分 '+v.score
    +'/10'+(v.issues||[]).map(i=>'<div class="issue"><b>['+i.severity+'/'+i.type+']</b> '
    +esc(i.problem)+'<br><span style="color:var(--sub)">建议：'+esc(i.suggestion||'')+'</span></div>').join('');
  toast('✅ 审查完成');
}
async function doRevise(){
  const r=await api('/api/revise',{number:parseInt($('#rvNum').value)});
  if(r.ok){toast('✅ 修订完成，当前 '+r.word_count+' 字');refresh();}
}

/* ---------- 阅读 ---------- */
function renderReadSel(){
  const sel=$('#readSel');const cur=sel.value;sel.innerHTML='';
  S.volumes.forEach(v=>v.chapters.forEach(c=>{
    if(c.status!=='final')return;
    const o=document.createElement('option');o.value=c.number;
    o.textContent='第'+c.number+'章 '+(c.title||'');sel.appendChild(o);
  }));
  if(cur)sel.value=cur;
}
async function loadReader(){
  const n=parseInt($('#readSel').value);if(!n)return;
  const ch=await api('/api/chapter?n='+n,null,'GET');
  if(ch.error)return;
  $('#readMeta').textContent=ch.word_count+' 字 · '+ch.status;
  const rv=ch.review&&ch.review.issues?ch.review:null;
  const score=rv?'<span class="score '+(rv.pass?'good':'bad')+'">审查 '+rv.score+'/10</span>':'';
  let issues='';
  if(rv&&rv.issues&&rv.issues.length){
    issues='<div style="font-family:sans-serif;text-align:left;margin-top:16px">'
      +rv.issues.map(i=>'<div class="issue"><b>['+i.severity+'/'+i.type+']</b> '+esc(i.problem)+'</div>').join('')+'</div>';
  }
  $('#readerWrap').innerHTML='<div class="reader"><h2>第'+ch.number+'章 · '+esc(ch.title||'')
    +'</h2><div class="meta">'+score+(ch.summary?' · '+esc(ch.summary.slice(0,50))+'…':'')+'</div>'
    +esc(ch.content||'（正文未生成，可到写作台生成）')+issues+'</div>';
}

/* ---------- 伏笔 & 用量 ---------- */
function renderForeshadow(){
  const tb=$('#fTable');tb.innerHTML='';
  S.foreshadows.forEach(f=>{
    tb.innerHTML+='<tr><td>'+(f.planted_ch?'第'+f.planted_ch+'章':'—')+'</td><td>'
      +esc(f.content)+'</td><td><span class="tag '+(f.resolved_ch?'resolved':'open')+'">'
      +(f.resolved_ch?'第'+f.resolved_ch+'章已回收':'未回收')+'</span></td></tr>';
  });
  if(!S.foreshadows.length)tb.innerHTML='<tr><td colspan="3" class="hint">暂无伏笔记录，写作后自动登记。</td></tr>';
}
function renderUsage(){
  const tb=$('#uTable');tb.innerHTML='';let tp=0,tc=0;
  Object.entries(S.usage).forEach(([role,u])=>{tp+=u.prompt_tokens;tc+=u.completion_tokens;
    tb.innerHTML+='<tr><td>'+role+'</td><td>'+fmtN(u.prompt_tokens)+'</td><td>'+fmtN(u.completion_tokens)+'</td></tr>';});
  if(!Object.keys(S.usage).length)tb.innerHTML='<tr><td colspan="3" class="hint">暂无调用记录。</td></tr>';
  else tb.innerHTML+='<tr><td><b>合计</b></td><td><b>'+fmtN(tp)+'</b></td><td><b>'+fmtN(tc)+'</b></td></tr>';
}

/* ---------- 导出 ---------- */
async function doExport(){
  const r=await api('/api/export',{format:'md'});
  if(r.ok)toast('✅ 已导出：'+r.path);
}

refresh().then(()=>loadReader());
</script>
</body>
</html>"""
