import threading

from app.services.ingest_jobs import FAILED, READY, SEARCHABLE, IngestQueue
from app.services.rag_service import IndexResult


class FakePipeline:
    """index_text / index_visuals stand-ins; `gate` holds the page-image step."""

    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate
        self.order = []

    def index_text(self, file, progress=None, run=None):
        self.order.append(("text", file.name))
        if file.name == "broken.pdf":
            return IndexResult(error="The PDF may be corrupted.")
        progress("Embedding passages (0 of 3)", 0.5)
        return IndexResult(added=3, stats={"pages": 2, "passages": 3, "charts": 0})

    def visuals_pending(self, file, text):
        return file.name.endswith(".pdf")

    def index_visuals(self, file, progress=None, text=None, run=None):
        if self.gate is not None:
            self.gate.wait(5)
        self.order.append(("visuals", file.name))
        return IndexResult(added=2, stats={"passages": 2})


def test_text_is_searchable_before_the_charts_are_read():
    gate = threading.Event()
    queue = IngestQueue(FakePipeline(gate))
    job_id = queue.submit("report.pdf", b"%PDF")

    for _ in range(100):
        (job,) = queue.jobs([job_id])
        if job.status == SEARCHABLE:
            break
        threading.Event().wait(0.02)
    assert job.status == SEARCHABLE and job.charts_pending and job.stats["passages"] == 3
    assert queue.busy()

    gate.set()
    queue.wait(5)
    (job,) = queue.jobs([job_id])
    assert job.status == READY and job.added == 5 and job.visual_passages == 2 and not job.charts_pending
    assert not queue.busy()


def test_files_are_indexed_in_order_and_a_failure_does_not_stop_the_next():
    pipeline = FakePipeline()
    queue = IngestQueue(pipeline)
    ids = [queue.submit(name, b"x") for name in ["broken.pdf", "notes.txt", "report.pdf"]]
    queue.wait(5)

    jobs = {job.name: job for job in queue.jobs(ids)}
    assert jobs["broken.pdf"].status == FAILED and "corrupted" in jobs["broken.pdf"].error
    assert jobs["notes.txt"].status == READY and jobs["report.pdf"].status == READY
    assert pipeline.order == [("text", "broken.pdf"), ("text", "notes.txt"), ("visuals", "notes.txt"),
                              ("text", "report.pdf"), ("visuals", "report.pdf")]


def test_an_unexpected_error_fails_only_that_job():
    pipeline = FakePipeline()
    pipeline.index_text = lambda file, progress=None, run=None: 1 / 0
    queue = IngestQueue(pipeline)
    job_id = queue.submit("report.pdf", b"x")
    queue.wait(5)

    (job,) = queue.jobs([job_id])
    assert job.status == FAILED and job.error


def test_jobs_returns_copies():
    queue = IngestQueue(FakePipeline())
    job_id = queue.submit("notes.txt", b"x")
    queue.wait(5)
    queue.jobs([job_id])[0].status = "changed"
    assert queue.jobs([job_id])[0].status == READY


def _count_pages(data: bytes, progress=None) -> int:
    progress("Counting", 0.5)
    return len(data)


def test_the_process_runner_reads_in_a_worker_process_and_relays_progress():
    from app.services.ingest_jobs import ProcessRunner

    heard = []
    run = ProcessRunner()
    assert run(_count_pages, b"abc", progress=lambda label, fraction: heard.append((label, fraction))) == 3
    assert heard == [("Counting", 0.5)]
    assert run(_count_pages, b"abcd", progress=None) == 4  # the worker is reused
