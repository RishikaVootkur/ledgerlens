"""The ingestion pipeline: select, download, parse, chunk, check quality.

Idempotent and resumable through `DATA_DIR/manifest.json`, which records each step
done per filing with content hashes:

- `download`: source URL used, sha256 and size of the raw file;
- `parse`: sha256 of the raw input, parser version, sha256 of `parsed/{id}.json`;
- `chunk:{strategy}`: sha256 of the parsed input, chunker version, sha256 of
  `chunks/{strategy}/{id}.jsonl`.

`filings.json` lists the FilingRefs ingested successfully by the latest run.

A step is skipped when its recorded input hash, version and output hash all still
match what is on disk, so a second run makes no requests and parses nothing. The
manifest is saved after every filing, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ledgerlens.config import ROOT, get_settings
from ledgerlens.ingest import quality
from ledgerlens.ingest.chunk import CHUNKER_VERSION, CHUNKERS
from ledgerlens.ingest.edgar import (
    FetchError,
    HttpCache,
    fetch_companyfacts,
    fetch_submissions,
    select_from_submissions,
)
from ledgerlens.ingest.financebench import financebench_filings
from ledgerlens.ingest.parse_html import parse_html
from ledgerlens.ingest.parse_pdf import parse_pdf
from ledgerlens.logging import configure_logging, get_logger
from ledgerlens.schemas import Chunk, FilingRef, ParsedFiling
from ledgerlens.universe import BY_TICKER, UNIVERSE

log = get_logger(__name__)

MANIFEST_VERSION = 1


def _code_version(*modules: str) -> str:
    """Hash of the source files that shape an output, so code changes redo the step."""
    here = Path(__file__).parent
    h = hashlib.sha256()
    for name in modules:
        h.update((here / f"{name}.py").read_bytes())
    return h.hexdigest()[:12]


PARSER_VERSION = _code_version("parse_html", "parse_pdf", "sections", "tables")
CHUNK_VERSION = f"{CHUNKER_VERSION}-{_code_version('chunk')}"
DOCS_REPORT = ROOT / "docs" / "data-quality.md"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def filings(self) -> Path:
        return self.root / "filings.json"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def parsed(self, filing_id: str) -> Path:
        return self.root / "parsed" / f"{filing_id}.json"

    def chunks(self, strategy: str, filing_id: str) -> Path:
        return self.root / "chunks" / strategy / f"{filing_id}.jsonl"


class Manifest:
    """`manifest.json`: per-filing steps with hashes. Saved atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"version": MANIFEST_VERSION, "filings": {}}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def entry(self, ref: FilingRef) -> dict[str, Any]:
        filings: dict[str, Any] = self.data.setdefault("filings", {})
        e: dict[str, Any] = filings.setdefault(ref.filing_id, {"steps": {}})
        e["ref"] = ref.model_dump(mode="json")
        return e

    def step(self, filing_id: str, name: str) -> dict[str, Any] | None:
        e = self.data.get("filings", {}).get(filing_id)
        if not e:
            return None
        s: dict[str, Any] | None = e.get("steps", {}).get(name)
        return s

    def set_step(self, ref: FilingRef, name: str, info: dict[str, Any]) -> None:
        e = self.entry(ref)
        e["steps"][name] = info
        e.pop("error", None)

    def set_error(self, ref: FilingRef, error: str, permanent: bool) -> None:
        self.entry(ref)["error"] = {"message": error, "permanent": permanent}

    def error(self, filing_id: str) -> dict[str, Any] | None:
        e = self.data.get("filings", {}).get(filing_id)
        return e.get("error") if e else None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)


@dataclass
class RunSummary:
    filings: list[FilingRef] = field(default_factory=list)
    downloaded: int = 0
    parsed: int = 0
    chunked: int = 0
    skipped: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    requests: int = 0
    report: dict[str, Any] | None = None


# ---------------------------------------------------------------- selection


def select_all(
    http: HttpCache,
    tickers: Iterable[str] | None,
    include_financebench: bool,
    summary: RunSummary,
    *,
    refresh: bool = False,
) -> tuple[list[FilingRef], dict[str, str]]:
    refs: list[FilingRef] = []
    companies = [BY_TICKER[t] for t in tickers] if tickers is not None else list(UNIVERSE)
    for c in companies:
        try:
            sub = fetch_submissions(c.cik, http, refresh=refresh)
            facts = fetch_companyfacts(c.cik, http, refresh=refresh)
            refs.extend(select_from_submissions(sub, facts, ticker=c.ticker, company=c.name))
        except FetchError as exc:
            log.error("selection_failed", ticker=c.ticker, error=str(exc))
            summary.failed[f"{c.ticker}-submissions"] = str(exc)
    links: dict[str, str] = {}
    if include_financebench:
        fb_refs, links = financebench_filings(http.raw_dir)
        refs.extend(fb_refs)
    return refs, links


# ---------------------------------------------------------------- download


def _valid(ref: FilingRef, data: bytes) -> bool:
    if ref.source_format == "pdf":
        return data[:5] == b"%PDF-"
    return b"<" in data[:2048]


def download(
    ref: FilingRef, http: HttpCache, manifest: Manifest, fallback: str | None
) -> Path | None:
    """Ensure the raw document is on disk. Returns its path, or None on failure."""
    done = manifest.step(ref.filing_id, "download")
    if done:
        path = Path(done["path"])
        if not path.is_absolute():
            path = http.raw_dir.parent / path
        if path.exists():
            return path
    urls = [ref.source_url] + ([fallback] if fallback else [])
    last_error = ""
    permanent = True
    for url in urls:
        try:
            data = http.fetch(url)
        except FetchError as exc:
            last_error = str(exc)
            permanent = permanent and exc.status in (403, 404, 410)
            continue
        if not _valid(ref, data):
            last_error = f"not a {ref.source_format} document: {url}"
            continue
        path = http.cache_path(url)
        manifest.set_step(
            ref,
            "download",
            {
                "url": url,
                "path": str(path.relative_to(http.raw_dir.parent)),
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            },
        )
        return path
    manifest.set_error(ref, last_error, permanent)
    log.error("download_failed", filing_id=ref.filing_id, error=last_error)
    return None


# ---------------------------------------------------------------- parse + chunk


def _write_atomic(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return sha256_bytes(text.encode("utf-8"))


def process_filing(
    ref_json: dict[str, Any],
    raw_path: str,
    data_root: str,
    do_parse: bool,
    strategies: list[str],
) -> dict[str, Any]:
    """Parse (if asked) and chunk one filing. Runs in a worker process."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    ref = FilingRef.model_validate(ref_json)
    paths = DataPaths(Path(data_root))
    out: dict[str, Any] = {"filing_id": ref.filing_id, "chunks": {}}
    raw = Path(raw_path).read_bytes()
    parsed_path = paths.parsed(ref.filing_id)
    if do_parse:
        parser = parse_pdf if ref.source_format == "pdf" else parse_html
        parsed = parser(raw, ref)
        out["parse"] = {
            "input_sha256": sha256_bytes(raw),
            "parser_version": PARSER_VERSION,
            "output_sha256": _write_atomic(parsed_path, parsed.model_dump_json()),
            "blocks": len(parsed.blocks),
        }
    else:
        parsed = ParsedFiling.model_validate_json(parsed_path.read_text(encoding="utf-8"))
    parsed_sha = sha256_file(parsed_path)
    for strategy in strategies:
        chunks = CHUNKERS[strategy](parsed)
        text = "".join(c.model_dump_json() + "\n" for c in chunks)
        out["chunks"][strategy] = {
            "input_sha256": parsed_sha,
            "chunker_version": CHUNK_VERSION,
            "output_sha256": _write_atomic(paths.chunks(strategy, ref.filing_id), text),
            "chunks": len(chunks),
        }
    return out


def _step_ok(
    step: dict[str, Any] | None, version_key: str, version: str, inp: str, out: Path
) -> bool:
    if not step or step.get(version_key) != version or step.get("input_sha256") != inp:
        return False
    return out.exists() and sha256_file(out) == step.get("output_sha256")


def plan_work(
    ref: FilingRef, raw_path: Path, manifest: Manifest, paths: DataPaths, strategies: list[str]
) -> tuple[bool, list[str]]:
    """Which steps still need doing: (parse?, strategies to chunk)."""
    raw_sha = sha256_file(raw_path)
    parsed_path = paths.parsed(ref.filing_id)
    parse_step = manifest.step(ref.filing_id, "parse")
    if not _step_ok(parse_step, "parser_version", PARSER_VERSION, raw_sha, parsed_path):
        return True, list(strategies)
    parsed_sha = sha256_file(parsed_path)
    todo = [
        s
        for s in strategies
        if not _step_ok(
            manifest.step(ref.filing_id, f"chunk:{s}"),
            "chunker_version",
            CHUNK_VERSION,
            parsed_sha,
            paths.chunks(s, ref.filing_id),
        )
    ]
    return False, todo


# ---------------------------------------------------------------- run


def default_workers() -> int:
    return max(1, min(6, (os.cpu_count() or 2) - 1))


def run(
    data_dir: Path,
    *,
    tickers: list[str] | None = None,
    include_financebench: bool = True,
    strategies: list[str] | None = None,
    workers: int | None = None,
    refresh_index: bool = False,
    retry_failed: bool = False,
    docs_report: Path | None = None,
    http: HttpCache | None = None,
) -> RunSummary:
    """Run the whole pipeline. Safe to rerun; finished steps are skipped."""
    strategies = strategies or list(CHUNKERS)
    for s in strategies:
        if s not in CHUNKERS:
            raise ValueError(f"unknown chunk strategy {s!r}; choose from {sorted(CHUNKERS)}")
    if tickers is not None:
        unknown = [t for t in tickers if t not in BY_TICKER]
        if unknown:
            raise ValueError(f"tickers not in the universe: {unknown}")
    paths = DataPaths(data_dir)
    own_http = http is None
    http = http or HttpCache(paths.raw)
    manifest = Manifest(paths.manifest)
    summary = RunSummary()
    try:
        refs, links = select_all(
            http, tickers, include_financebench, summary, refresh=refresh_index
        )
        summary.filings = refs
        jobs: list[tuple[FilingRef, Path, bool, list[str]]] = []
        for ref in refs:
            err = manifest.error(ref.filing_id)
            if err and err.get("permanent") and not retry_failed:
                summary.failed[ref.filing_id] = err["message"]
                continue
            had_download = manifest.step(ref.filing_id, "download") is not None
            raw_path = download(ref, http, manifest, links.get(ref.doc_name or ""))
            manifest.save()
            if raw_path is None:
                summary.failed[ref.filing_id] = (manifest.error(ref.filing_id) or {}).get(
                    "message", "download failed"
                )
                continue
            summary.downloaded += not had_download
            do_parse, todo = plan_work(ref, raw_path, manifest, paths, strategies)
            if do_parse or todo:
                jobs.append((ref, raw_path, do_parse, todo))
            else:
                summary.skipped += 1
        _run_jobs(jobs, paths, manifest, summary, workers or default_workers())
    finally:
        summary.requests = http.requests_made
        if own_http:
            http.close()

    ok = [r for r in summary.filings if r.filing_id not in summary.failed]
    _write_atomic(
        paths.filings,
        json.dumps([r.model_dump(mode="json") for r in ok], indent=2) + "\n",
    )
    summary.report = build_report(ok, paths, strategies)
    _, md_path = quality.write_reports(summary.report, paths.reports)
    if docs_report is not None:
        docs_report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(md_path, docs_report)
    log.info(
        "ingest_done",
        filings=len(summary.filings),
        downloaded=summary.downloaded,
        parsed=summary.parsed,
        chunked=summary.chunked,
        skipped=summary.skipped,
        failed=len(summary.failed),
        requests=summary.requests,
    )
    return summary


def _record(
    result: dict[str, Any], ref: FilingRef, manifest: Manifest, summary: RunSummary
) -> None:
    if "parse" in result:
        manifest.set_step(ref, "parse", result["parse"])
        summary.parsed += 1
    for strategy, info in result["chunks"].items():
        manifest.set_step(ref, f"chunk:{strategy}", info)
    if result["chunks"]:
        summary.chunked += 1
    manifest.save()


def _run_jobs(
    jobs: list[tuple[FilingRef, Path, bool, list[str]]],
    paths: DataPaths,
    manifest: Manifest,
    summary: RunSummary,
    workers: int,
) -> None:
    if not jobs:
        return
    log.info("processing", filings=len(jobs), workers=workers)
    if workers <= 1:
        for ref, raw, do_parse, todo in jobs:
            try:
                result = process_filing(
                    ref.model_dump(mode="json"), str(raw), str(paths.root), do_parse, todo
                )
            except Exception as exc:
                _fail(ref, exc, manifest, summary)
                continue
            _record(result, ref, manifest, summary)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[dict[str, Any]], FilingRef] = {
            pool.submit(
                process_filing,
                ref.model_dump(mode="json"),
                str(raw),
                str(paths.root),
                do_parse,
                todo,
            ): ref
            for ref, raw, do_parse, todo in jobs
        }
        for fut, ref in futures.items():
            try:
                result = fut.result()
            except Exception as exc:
                _fail(ref, exc, manifest, summary)
                continue
            _record(result, ref, manifest, summary)
            log.info("processed", filing_id=ref.filing_id)


def _fail(ref: FilingRef, exc: Exception, manifest: Manifest, summary: RunSummary) -> None:
    msg = f"{type(exc).__name__}: {exc}"
    log.error("process_failed", filing_id=ref.filing_id, error=msg)
    manifest.set_error(ref, msg, permanent=False)
    manifest.save()
    summary.failed[ref.filing_id] = msg


def load_parsed(paths: DataPaths, filing_id: str) -> ParsedFiling:
    return ParsedFiling.model_validate_json(paths.parsed(filing_id).read_text(encoding="utf-8"))


def load_chunks(paths: DataPaths, strategy: str, filing_id: str) -> list[Chunk]:
    path = paths.chunks(strategy, filing_id)
    with path.open(encoding="utf-8") as fh:
        return [Chunk.model_validate_json(line) for line in fh if line.strip()]


def build_report(refs: list[FilingRef], paths: DataPaths, strategies: list[str]) -> dict[str, Any]:
    parsed = [load_parsed(paths, r.filing_id) for r in refs if paths.parsed(r.filing_id).exists()]
    chunks: dict[str, list[Chunk]] = {}
    for s in strategies:
        chunks[s] = []
        for r in refs:
            if paths.chunks(s, r.filing_id).exists():
                chunks[s].extend(load_chunks(paths, s, r.filing_id))
    return quality.check(parsed, chunks)
