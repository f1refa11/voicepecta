import threading
import uuid
from queue import Queue
from typing import Dict, Optional, Tuple


class Job:
    def __init__(self, audio_path: str, engine: str, model: str, username: str, language: str | None = None) -> None:
        self.id = str(uuid.uuid4())
        self.audio_path = audio_path
        self.engine = engine
        self.model = model
        self.username = username
        self.language = language
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self._subscribers: list[Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> Queue:
        q: Queue = Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def emit(self, event: str, data: str) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            q.put((event, data))

    def close(self) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            q.put(None)


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
