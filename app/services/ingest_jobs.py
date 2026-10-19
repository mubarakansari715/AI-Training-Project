"""Background document indexing.

An upload is queued and indexed by one worker thread, so the page stays
usable while a long PDF is read. Each file goes through two stages
(RAGPipeline.index_text, then index_visuals): its text becomes searchable
first, then the charts and tables read from its page images (a minute or
more on a long report) are added.

One worker, so the embedding model and the vector store are never written
by two jobs at once. The queue lives in memory, shared by every browser
session of the app process: a page refresh keeps a running job, an app
restart loses it. The worker never touches Streamlit; the page reads each
job's status (IngestQueue.jobs) and shows it.

The reading itself (PDF text, page rendering, chart detection) runs in a
worker process (ProcessRunner): PyMuPDF and pypdf hold Python's GIL through
long page operations, which on a thread would leave the page unresponsive
for most of a long PDF. Embedding and storing stay in this process, next to
the vector store.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace

from app.rag.progress import Progress, report
from app.services.document_service import UploadedDocument
from app.services.rag_service import run_inline

logger = logging.getLogger(__name__)

QUEUED = "queued"
READING = "reading"  # the text
SEARCHABLE = "searchable"  # text indexed, page images still being read
READY = "ready"
FAILED = "failed"


@dataclass
class IngestJob:
    id: str
    name: str
    status: str = QUEUED
    label: str = "Waiting to start"
    fraction: float = 0.0
    # {"pages", "passages", "charts"} once the text is indexed (IndexResult.stats).
    stats: dict[str, int] = field(default_factory=dict)
    # Chunks added; 0 for a re-upload of an unchanged document.
    added: int = 0
    visual_passages: int = 0
    # The text is indexed and its page images are being read.
    charts_pending: bool = False
    error: str | None = None
    # Page images couldn't be read; the text is still searchable.
    warning: str | None = None

    @property
    def active(self) -> bool:
        return self.status in (QUEUED, READING, SEARCHABLE)


class ProcessRunner:
    """Runs a reading step (run_inline's signature) in a worker process,
    relaying its progress reports; one step at a time. The process starts on
    first use and is replaced if it dies."""

    def __init__(self):
        self._context = multiprocessing.get_context("spawn")
        self._progress = self._context.Queue()
        self._pool: ProcessPoolExecutor | None = None

    def __call__(self, fn, *args, progress: Progress | None = None, **kwargs):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=self._context,
                                             initializer=_start_worker, initargs=(self._progress,))
        future = self._pool.submit(_run_in_worker, fn, args, kwargs)
        while True:
            try:
                report(progress, *self._progress.get(timeout=0.25))
            except queue.Empty:
                if future.done():
                    break
        try:
            return future.result()
        except BrokenProcessPool:
            self._pool = None
            raise


_worker_progress = None


def _start_worker(progress_queue) -> None:
    global _worker_progress
    _worker_progress = progress_queue
    logging.basicConfig(level=logging.INFO)


def _run_in_worker(fn, args, kwargs):
    return fn(*args, progress=lambda label, fraction: _worker_progress.put((label, fraction)), **kwargs)


class IngestQueue:
    """Indexes submitted files one at a time on a background thread; `run`
    (default run_inline) runs their reading, e.g. a ProcessRunner."""

    def __init__(self, pipeline, run=None):
        self._pipeline = pipeline
        self._run_reading = run or run_inline
        self._jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

    def submit(self, name: str, data: bytes) -> str:
        """Queue a file; returns its job id."""
        job = IngestJob(id=uuid.uuid4().hex, name=name)
        with self._lock:
            self._jobs[job.id] = job
        self._worker.submit(self._run, job.id, UploadedDocument(name, data))
        return job.id

    def jobs(self, ids=None) -> list[IngestJob]:
        """Copies of the jobs (those in `ids`, default all), oldest first."""
        with self._lock:
            return [replace(job) for job in self._jobs.values() if ids is None or job.id in ids]

    def busy(self) -> bool:
        with self._lock:
            return any(job.active for job in self._jobs.values())

    def wait(self, timeout: float | None = None) -> None:
        """Block until every job submitted so far has finished (tests, scripts)."""
        self._worker.submit(lambda: None).result(timeout)

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for name, value in changes.items():
                setattr(job, name, value)

    def _progress(self, job_id: str):
        return lambda label, fraction: self._update(job_id, label=label, fraction=fraction)

    def _run(self, job_id: str, document: UploadedDocument) -> None:
        try:
            self._update(job_id, status=READING, label="Reading the document")
            text = self._pipeline.index_text(document, self._progress(job_id), run=self._run_reading)
            if text.error:
                self._update(job_id, status=FAILED, error=text.error)
                return
            self._update(job_id, status=SEARCHABLE, stats=text.stats, added=text.added,
                         charts_pending=self._pipeline.visuals_pending(document, text),
                         label="Reading charts and tables", fraction=0.0)
            visuals = self._pipeline.index_visuals(document, self._progress(job_id), text, run=self._run_reading)
            self._update(job_id, status=READY, added=text.added + visuals.added, charts_pending=False,
                         visual_passages=visuals.stats.get("passages", 0), warning=visuals.error,
                         label="Ready", fraction=1.0)
        except Exception:
            logger.exception("Indexing %s failed", document.name)
            self._update(job_id, status=FAILED, error="Something went wrong while processing this document.")
