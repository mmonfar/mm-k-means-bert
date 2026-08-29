"""
serve.py
mmonfar. // Semantic M&M Failure Navigation Engine — local workbench

Runs the canvas as a local app so a non-technical user can drop their own M&M case
register onto the page and watch the galaxy rebuild, without touching a terminal after
the first launch.

    python serve.py
    -> http://127.0.0.1:8000

Nothing leaves the machine. The upload endpoint writes to a local scratch folder, runs
exactly the same `engine.py` pipeline the CLI runs, and returns the payload. There is no
outbound request at any point, and the file never goes anywhere near a network.

Deliberately stdlib-only — no Flask, no FastAPI, nothing to install beyond what the
pipeline already needs. A hospital IG review should be able to read this file in full in
about five minutes.

Security posture: binds to 127.0.0.1 by default, caps upload size, allows only the
extensions the engine can read, and never uses a client-supplied path component.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import engine

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
UPLOAD_DIR = ROOT / ".uploads"

MAX_UPLOAD_BYTES = 32 * 1024 * 1024  # 32 MB — far above any realistic M&M register
ALLOWED_SUFFIXES = engine.SPREADSHEET_SUFFIXES | engine.DELIMITED_SUFFIXES

# One pipeline run at a time. The model is not cheap to hold twice, and two concurrent
# runs would race on app/data.json.
_pipeline_lock = threading.Lock()


def _core_members(members, centroid, keep=0.7):
    """The cases nearest the middle of their group.

    Ranked in projection space, which is cheap and already in the payload. That
    is only used to CHOOSE which cases to learn from — the learning itself
    embeds them properly in 384-d. The outliers are exactly the cases a mixed
    group has wrongly swept up, and they are the ones that would teach a
    taxonomy something false.
    """
    if len(members) <= 3:
        return list(members)
    mid = [
        sum(p["x"] for p in members) / len(members),
        sum(p["y"] for p in members) / len(members),
        sum(p["z"] for p in members) / len(members),
    ]
    ranked = sorted(
        members,
        key=lambda p: (p["x"] - mid[0]) ** 2 + (p["y"] - mid[1]) ** 2
        + (p["z"] - mid[2]) ** 2,
    )
    return ranked[: max(3, int(len(ranked) * keep))]


class Handler(SimpleHTTPRequestHandler):
    """Static file server for app/, plus a single ingest endpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    # -- logging ---------------------------------------------------------------

    def log_message(self, fmt, *args):  # quieter than the default one-line-per-asset
        if self.path.startswith("/api/"):
            print(f"      {self.command} {self.path}")

    # -- helpers ---------------------------------------------------------------

    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def end_headers(self):
        # The payload is regenerated on upload; a cached data.json would show the
        # previous register after a successful ingest.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # -- routes ----------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.split("?")[0] != "/api/ingest":
            self._json(404, {"error": "no such endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return

        if length <= 0:
            self._json(400, {"error": "empty upload"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._json(
                413,
                {"error": f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"},
            )
            return

        # Only the suffix is taken from the client. The stem is fixed, so a crafted
        # filename cannot escape the upload directory or overwrite project files.
        raw_name = self.headers.get("X-Filename", "upload")
        suffix = Path(raw_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            self._json(415, {
                "error": f"{suffix or 'that file type'} is not supported",
                "detail": "Use " + ", ".join(sorted(ALLOWED_SUFFIXES)),
            })
            return

        UPLOAD_DIR.mkdir(exist_ok=True)
        target = UPLOAD_DIR / f"register{suffix}"
        with target.open("wb") as fh:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)

        overrides = {}
        for header, canonical in (
            ("X-Text-Col", "Case_Summary"),
            ("X-Id-Col", "Case_ID"),
            ("X-Date-Col", "Date"),
            ("X-Dept-Col", "Department"),
            ("X-Severity-Col", "Severity_Score"),
        ):
            value = (self.headers.get(header) or "").strip()
            if value:
                overrides[canonical] = value

        try:
            k = int(self.headers.get("X-Clusters") or 5)
        except ValueError:
            k = 5
        k = max(2, min(12, k))

        if not _pipeline_lock.acquire(blocking=False):
            self._json(409, {"error": "a rebuild is already running"})
            return

        try:
            print(f"\n[mmonfar.] rebuilding from {raw_name!r} (k={k})")
            payload = engine.run(target, k=k, overrides=overrides or None)
            self._json(200, {"ok": True, "meta": payload["meta"]})
        except (ValueError, FileNotFoundError) as exc:
            # Expected, user-fixable problems: bad columns, too few rows, wrong format.
            print(f"[mmonfar.] ingest rejected: {exc}")
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - unexpected
            traceback.print_exc()
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            _pipeline_lock.release()

    def _current_input(self) -> Path | None:
        """The register currently loaded: an uploaded one if present, else the demo."""
        if UPLOAD_DIR.exists():
            for candidate in sorted(UPLOAD_DIR.glob("register.*")):
                if candidate.suffix.lower() in ALLOWED_SUFFIXES:
                    return candidate
        return engine.DEFAULT_INPUT if engine.DEFAULT_INPUT.exists() else None

    def do_PUT(self) -> None:  # noqa: N802
        """Re-cluster the register already loaded, at a different k.

        Changing the number of galaxies is a clustering decision, not a filter — it
        cannot be done in the browser, because it needs the embeddings. Re-running
        is cheap on an already-cached model.
        """
        if self.path.split("?")[0] != "/api/clusters":
            self._json(404, {"error": "no such endpoint"})
            return

        try:
            k = int(self.headers.get("X-Clusters") or 5)
        except ValueError:
            self._json(400, {"error": "X-Clusters must be a number"})
            return
        k = max(2, min(12, k))

        source = self._current_input()
        if source is None:
            self._json(400, {
                "error": "no register loaded",
                "detail": "Run `python data_generator.py` or upload a file.",
            })
            return

        if not _pipeline_lock.acquire(blocking=False):
            self._json(409, {"error": "a rebuild is already running"})
            return
        try:
            print(f"\n[mmonfar.] re-clustering {source.name} at k={k}")
            payload = engine.run(source, k=k)
            self._json(200, {"ok": True, "meta": payload["meta"]})
        except (ValueError, FileNotFoundError) as exc:
            self._json(400, {"error": str(exc)})
        finally:
            _pipeline_lock.release()

    def do_PATCH(self) -> None:  # noqa: N802
        """Record the name a human gave a group, and rebuild so it takes effect.

        This is the only write in the whole tool that is not derived from the
        register: it is a person's judgement, and it outranks every generated
        name. Stored against a fingerprint of the group's members plus its
        centroid, so it survives into next quarter's export even though cluster
        ids will not.
        """
        route = self.path.split("?")[0]
        if route not in ("/api/name", "/api/case", "/api/cases"):
            self._json(404, {"error": "no such endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError) as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return

        if route == "/api/case":
            self._file_case(body)
            return

        if route == "/api/cases":
            self._file_cases(body)
            return

        try:
            cluster_id = int(body["cluster"])
            name = str(body["name"]).strip()
        except (ValueError, KeyError, TypeError) as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return

        if not 1 <= len(name) <= 60:
            self._json(400, {"error": "a name must be 1-60 characters"})
            return

        payload_path = APP_DIR / "data.json"
        if not payload_path.exists():
            self._json(400, {"error": "no galaxy built yet"})
            return

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        group = next((c for c in payload["clusters"] if c["id"] == cluster_id), None)
        if group is None:
            self._json(404, {"error": f"no group {cluster_id}"})
            return

        members = [p for p in payload["points"] if p["cluster"] == cluster_id]
        if not members:
            self._json(400, {"error": "that group is empty"})
            return

        import feedback

        author = self.headers.get("X-Author") or None
        conn = feedback.connect()
        try:
            fp = feedback.save_name(
                conn,
                [feedback.case_key(p["summary"]) for p in members],
                name,
                group.get("centroid") or [],
                author=author,
            )

            # Naming a group is also the cheapest bulk labelling there is: the
            # person has just told us what these cases are. Only the core of the
            # group is learned, though — the cases nearest its centre. A mixed
            # group's stragglers would teach the taxonomy the wrong thing, and a
            # taxonomy is much harder to unlearn than it is to poison.
            learned = 0
            centroid = group.get("centroid")
            if centroid:
                core = _core_members(members, centroid)
                for p_ in core:
                    feedback.label_case(conn, p_["summary"], name, author=author)
                # Embedded properly, in the same 384-d space the classifier
                # uses. The projection coordinates in the payload are 3-d and
                # would have taught the taxonomy in a space it never reads
                # from — a silent dimension mismatch.
                learned = feedback.learn(
                    conn, name,
                    engine.embed([p_["summary"] for p_ in core]),
                    author=author,
                )
            state = feedback.summary(conn)
        finally:
            conn.close()

        print(f"[mmonfar.] group {cluster_id} named {name!r} by hand "
              f"({learned} cases now under that label)")

        if not _pipeline_lock.acquire(blocking=False):
            self._json(409, {"error": "a rebuild is already running"})
            return
        try:
            source = self._current_input()
            engine.run(source, k=payload["meta"]["n_clusters"])
        finally:
            _pipeline_lock.release()

        self._json(200, {"ok": True, "fingerprint": fp, "store": state})

    def _file_case(self, body: dict) -> None:
        """File one case under a label, by hand.

        The counterpart to renaming a group. Clustering gets roughly two thirds
        of cases into the group a clinician would choose; this is how the other
        third gets corrected, and every correction also teaches the taxonomy, so
        the same mistake is less likely next quarter.
        """
        case_id = str(body.get("case", "")).strip()
        name = str(body.get("label", "")).strip()
        if not case_id or not 1 <= len(name) <= 60:
            self._json(400, {"error": "need a case id and a 1-60 character label"})
            return

        payload_path = APP_DIR / "data.json"
        if not payload_path.exists():
            self._json(400, {"error": "no galaxy built yet"})
            return

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        point = next((p for p in payload["points"] if p["id"] == case_id), None)
        if point is None:
            self._json(404, {"error": f"no case {case_id}"})
            return

        import feedback

        author = self.headers.get("X-Author") or None
        conn = feedback.connect()
        try:
            feedback.label_case(conn, point["summary"], name, author=author)
            total = feedback.learn(
                conn, name, engine.embed([point["summary"]]), author=author)
            state = feedback.summary(conn)
        finally:
            conn.close()

        print(f"[mmonfar.] {case_id} filed under {name!r} "
              f"({total} cases now under that label)")

        if not _pipeline_lock.acquire(blocking=False):
            self._json(409, {"error": "a rebuild is already running"})
            return
        try:
            engine.run(self._current_input(), k=payload["meta"]["n_clusters"])
        finally:
            _pipeline_lock.release()

        self._json(200, {"ok": True, "label": name, "store": state})

    def _file_cases(self, body: dict) -> None:
        """File every case in a filtered view under one label, in a single action.

        Same semantics as `_file_case` for each member — the point is that a
        clinician reviewing "all severity-5 cases in ED" should be able to file
        the lot at once rather than clicking through them one at a time. Every
        case filed this way is a human filing exactly like the single-case
        route, so it is never overruled by the classifier on a later run (see
        `filed` handling in engine.apply_house_taxonomy). The pipeline rebuild
        runs once at the end, not once per case.
        """
        case_ids = body.get("cases")
        name = str(body.get("label", "")).strip()
        if not isinstance(case_ids, list) or not case_ids:
            self._json(400, {"error": "need a non-empty list of case ids"})
            return
        if not 1 <= len(name) <= 60:
            self._json(400, {"error": "a label must be 1-60 characters"})
            return

        payload_path = APP_DIR / "data.json"
        if not payload_path.exists():
            self._json(400, {"error": "no galaxy built yet"})
            return

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        by_id = {p["id"]: p for p in payload["points"]}
        wanted = {str(c) for c in case_ids}
        points = [by_id[c] for c in wanted if c in by_id]
        if not points:
            self._json(404, {"error": "none of the given case ids were found"})
            return

        import feedback

        author = self.headers.get("X-Author") or None
        conn = feedback.connect()
        try:
            for point in points:
                feedback.label_case(conn, point["summary"], name, author=author)
            total = feedback.learn(
                conn, name, engine.embed([p["summary"] for p in points]), author=author)
            state = feedback.summary(conn)
        finally:
            conn.close()

        print(f"[mmonfar.] {len(points)} case(s) filed under {name!r} in one action "
              f"({total} cases now under that label)")

        if not _pipeline_lock.acquire(blocking=False):
            self._json(409, {"error": "a rebuild is already running"})
            return
        try:
            engine.run(self._current_input(), k=payload["meta"]["n_clusters"])
        finally:
            _pipeline_lock.release()

        self._json(200, {"ok": True, "label": name, "filed": len(points), "store": state})

    def do_DELETE(self) -> None:  # noqa: N802
        """Restore the shipped demo dataset and forget the uploaded register."""
        if self.path.split("?")[0] != "/api/uploads":
            self._json(404, {"error": "no such endpoint"})
            return

        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        default = engine.DEFAULT_INPUT
        if not default.exists():
            self._json(400, {
                "error": "mock_mm_minutes.xlsx is missing",
                "detail": "Run `python data_generator.py` to regenerate the demo data.",
            })
            return

        if not _pipeline_lock.acquire(blocking=False):
            self._json(409, {"error": "a rebuild is already running"})
            return
        try:
            payload = engine.run(default)
            self._json(200, {"ok": True, "meta": payload["meta"]})
        finally:
            _pipeline_lock.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the galaxy as a local app.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="default 127.0.0.1 — this machine only")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not (APP_DIR / "data.json").exists():
        print("[mmonfar.] no payload yet — building from the demo dataset first")
        if not engine.DEFAULT_INPUT.exists():
            import data_generator

            data_generator.main()
        engine.run(engine.DEFAULT_INPUT)

    url = f"http://{args.host}:{args.port}"
    print(f"\n[mmonfar.] workbench running at {url}")
    print("[mmonfar.] drop your own register onto the page to rebuild the galaxy")
    if args.host not in ("127.0.0.1", "localhost"):
        print("[mmonfar.] WARNING: bound beyond localhost — this machine is now "
              "reachable on the network. Do not do this with real data.")
    print("[mmonfar.] ctrl-c to stop\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mmonfar.] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
